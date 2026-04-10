"""Tests for imatrix parsing and analysis."""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path

import pytest

from quantbench._types import QuantbenchError
from quantbench.imatrix import (
    ImatrixAnalysis,
    ImatrixData,
    ImatrixEntry,
    analyze_imatrix,
    format_imatrix_report,
    parse_imatrix,
)


def _pack_entry(name: str, num_calls: int, values: list[float]) -> bytes:
    """Build a binary imatrix entry."""
    name_bytes = name.encode("utf-8")
    buf = struct.pack("<i", len(name_bytes))
    buf += name_bytes
    buf += struct.pack("<i", len(values))
    buf += struct.pack("<i", num_calls)
    buf += struct.pack(f"<{len(values)}f", *values)
    return buf


def _write_imatrix(entries: list[tuple[str, int, list[float]]], path: Path) -> None:
    """Write multiple entries to a binary imatrix file."""
    with open(path, "wb") as f:
        for name, num_calls, values in entries:
            f.write(_pack_entry(name, num_calls, values))


# ── Parsing tests ──


def test_parse_single_entry(tmp_path: Path) -> None:
    p = tmp_path / "single.imatrix"
    _write_imatrix([("blk.0.attn_q.weight", 100, [1.0, 2.0, 3.0])], p)
    data = parse_imatrix(p)
    assert len(data.entries) == 1
    assert data.entries[0].name == "blk.0.attn_q.weight"
    assert data.entries[0].num_values == 3
    assert data.entries[0].num_calls == 100
    assert data.entries[0].values == pytest.approx([1.0, 2.0, 3.0])
    assert data.total_calls == 100


def test_parse_multiple_entries(tmp_path: Path) -> None:
    entries = [
        ("layer.0.weight", 50, [0.1, 0.2]),
        ("layer.1.weight", 80, [0.5, 0.6, 0.7]),
        ("layer.2.weight", 80, [1.0]),
    ]
    p = tmp_path / "multi.imatrix"
    _write_imatrix(entries, p)
    data = parse_imatrix(p)
    assert len(data.entries) == 3
    assert data.total_calls == 80
    assert data.entries[1].name == "layer.1.weight"
    assert data.entries[1].num_values == 3


def test_parse_empty_values(tmp_path: Path) -> None:
    p = tmp_path / "empty_vals.imatrix"
    _write_imatrix([("empty_tensor", 10, [])], p)
    data = parse_imatrix(p)
    assert len(data.entries) == 1
    assert data.entries[0].num_values == 0
    assert data.entries[0].values == []


def test_parse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(QuantbenchError, match="not found"):
        parse_imatrix(tmp_path / "nonexistent.imatrix")


def test_parse_invalid_name_length(tmp_path: Path) -> None:
    p = tmp_path / "bad.imatrix"
    with open(p, "wb") as f:
        f.write(struct.pack("<i", -5))  # negative name length
    with pytest.raises(QuantbenchError, match="Invalid name length"):
        parse_imatrix(p)


def test_parse_truncated_file(tmp_path: Path) -> None:
    """Truncate a valid entry mid-values."""
    p = tmp_path / "truncated.imatrix"
    full = _pack_entry("tensor", 10, [1.0, 2.0, 3.0])
    with open(p, "wb") as f:
        f.write(full[: len(full) - 4])  # chop off last float
    with pytest.raises(QuantbenchError, match="Unexpected EOF"):
        parse_imatrix(p)


# ── ImatrixData method tests ──


def test_by_name_found() -> None:
    e1 = ImatrixEntry("a", 2, 10, [1.0, 2.0])
    e2 = ImatrixEntry("b", 1, 10, [5.0])
    data = ImatrixData(entries=[e1, e2], total_calls=10)
    assert data.by_name("b") is e2


def test_by_name_not_found() -> None:
    data = ImatrixData(entries=[], total_calls=0)
    assert data.by_name("missing") is None


def test_top_k() -> None:
    entries = [
        ImatrixEntry("low", 2, 10, [0.1, 0.1]),
        ImatrixEntry("high", 2, 10, [9.0, 8.0]),
        ImatrixEntry("mid", 2, 10, [3.0, 4.0]),
    ]
    data = ImatrixData(entries=entries, total_calls=10)
    top2 = data.top_k(2)
    assert len(top2) == 2
    assert top2[0].name == "high"
    assert top2[1].name == "mid"


def test_top_k_larger_than_entries() -> None:
    entries = [ImatrixEntry("only", 1, 5, [1.0])]
    data = ImatrixData(entries=entries, total_calls=5)
    assert len(data.top_k(10)) == 1


# ── Analysis tests ──


def test_analyze_basic() -> None:
    entries = [
        ImatrixEntry("a", 3, 10, [1.0, 2.0, 3.0]),
        ImatrixEntry("b", 2, 10, [5.0, 5.0]),
    ]
    data = ImatrixData(entries=entries, total_calls=10)
    analysis = analyze_imatrix(data)
    assert analysis.total_tensors == 2
    assert analysis.mean_importance_per_layer["a"] == pytest.approx(2.0)
    assert analysis.mean_importance_per_layer["b"] == pytest.approx(5.0)
    assert analysis.variance_per_layer["a"] == pytest.approx(2 / 3)


def test_analyze_outlier_detection() -> None:
    """One tensor with wildly higher values should be flagged."""
    entries = [
        ImatrixEntry(f"normal_{i}", 4, 10, [1.0, 1.0, 1.0, 1.0])
        for i in range(10)
    ]
    entries.append(ImatrixEntry("outlier", 4, 10, [100.0, 100.0, 100.0, 100.0]))
    data = ImatrixData(entries=entries, total_calls=10)
    analysis = analyze_imatrix(data)
    assert "outlier" in analysis.outlier_layers
    # Normal layers should not be outliers
    for i in range(10):
        assert f"normal_{i}" not in analysis.outlier_layers


def test_analyze_no_outliers() -> None:
    entries = [
        ImatrixEntry("a", 2, 10, [1.0, 1.0]),
        ImatrixEntry("b", 2, 10, [1.0, 1.0]),
    ]
    data = ImatrixData(entries=entries, total_calls=10)
    analysis = analyze_imatrix(data)
    assert analysis.outlier_layers == []


def test_analyze_empty_values() -> None:
    entries = [ImatrixEntry("empty", 0, 10, [])]
    data = ImatrixData(entries=entries, total_calls=10)
    analysis = analyze_imatrix(data)
    assert analysis.mean_importance_per_layer["empty"] == 0.0
    assert analysis.variance_per_layer["empty"] == 0.0


# ── Report formatting tests ──


def test_format_report_contains_sections() -> None:
    analysis = ImatrixAnalysis(
        total_tensors=2,
        mean_importance_per_layer={"a": 1.0, "b": 5.0},
        variance_per_layer={"a": 0.5, "b": 0.1},
        outlier_layers=["b"],
    )
    report = format_imatrix_report(analysis)
    assert "Importance Matrix Analysis Report" in report
    assert "Total tensors: 2" in report
    assert "[OUTLIER]" in report
    assert "b" in report


def test_format_report_no_outliers() -> None:
    analysis = ImatrixAnalysis(
        total_tensors=1,
        mean_importance_per_layer={"x": 2.0},
        variance_per_layer={"x": 0.0},
        outlier_layers=[],
    )
    report = format_imatrix_report(analysis)
    assert "No outlier layers detected." in report
    assert "[OUTLIER]" not in report


# ── Round-trip integration test ──


def test_roundtrip_parse_analyze_report(tmp_path: Path) -> None:
    entries = [
        ("blk.0.attn_q.weight", 200, [0.5, 0.6, 0.7, 0.8]),
        ("blk.0.attn_k.weight", 200, [0.4, 0.5, 0.6, 0.7]),
        ("blk.1.attn_q.weight", 200, [0.5, 0.5, 0.5, 0.5]),
        ("blk.1.attn_k.weight", 200, [0.4, 0.4, 0.4, 0.4]),
        ("blk.2.attn_q.weight", 200, [0.6, 0.6, 0.6, 0.6]),
        ("blk.2.attn_k.weight", 200, [0.3, 0.3, 0.3, 0.3]),
        ("blk.1.ffn_gate.weight", 200, [0.1, 0.1, 0.1, 0.1]),
        ("output.weight", 200, [50.0, 55.0, 52.0, 53.0]),
    ]
    p = tmp_path / "full.imatrix"
    _write_imatrix(entries, p)

    data = parse_imatrix(p)
    assert len(data.entries) == 8

    analysis = analyze_imatrix(data)
    assert analysis.total_tensors == 8
    assert "output.weight" in analysis.outlier_layers

    report = format_imatrix_report(analysis)
    assert "output.weight" in report
    assert len(report) > 100
