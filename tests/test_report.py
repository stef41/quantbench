"""Tests for quantbench.report."""

import json

from quantbench._types import (
    DType,
    ModelProfile,
    QualityEstimate,
    QuantFormat,
    QuantProfile,
    TensorInfo,
)
from quantbench.report import (
    format_markdown,
    format_report_rich,
    format_report_text,
    load_json,
    report_to_dict,
    save_json,
)


def _make_profile() -> ModelProfile:
    tensors = [TensorInfo(name="w", shape=[1000], dtype=DType.Q4_K_M)]
    return ModelProfile(
        name="test-model", format=QuantFormat.GGUF,
        total_params=1000, total_size_bytes=600,
        tensors=tensors,
        quant=QuantProfile(
            avg_bits_per_weight=4.85,
            dtype_distribution={"q4_k_m": 1.0},
            n_quantized_layers=1,
        ),
    )


def _make_quality() -> QualityEstimate:
    return QualityEstimate(
        model_name="test-model", method="ggml",
        avg_bits_per_weight=4.85,
        estimated_perplexity_delta=0.08,
        quality_score=0.85, risk_level="medium",
        recommendations=["Good balance of size and quality"],
    )


class TestReportToDict:
    def test_basic(self):
        d = report_to_dict(_make_profile())
        assert d["name"] == "test-model"
        assert d["format"] == "gguf"

    def test_with_quality(self):
        d = report_to_dict(_make_profile(), _make_quality())
        assert "quality" in d
        assert d["quality"]["risk_level"] == "medium"

    def test_json_serializable(self):
        d = report_to_dict(_make_profile(), _make_quality())
        s = json.dumps(d)
        assert isinstance(s, str)


class TestSaveLoadJson:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "report.json"
        save_json(_make_profile(), path, _make_quality())
        loaded = load_json(path)
        assert loaded["name"] == "test-model"
        assert "quality" in loaded


class TestFormatReportText:
    def test_contains_name(self):
        text = format_report_text(_make_profile())
        assert "test-model" in text

    def test_with_quality(self):
        text = format_report_text(_make_profile(), _make_quality())
        assert "medium" in text.lower() or "MEDIUM" in text

    def test_contains_stats(self):
        text = format_report_text(_make_profile())
        assert "4.85" in text


class TestFormatReportRich:
    def test_returns_string(self):
        result = format_report_rich(_make_profile())
        assert isinstance(result, str)
        assert len(result) > 0


class TestFormatMarkdown:
    def test_has_header(self):
        md = format_markdown(_make_profile())
        assert "test-model" in md
        assert "#" in md

    def test_with_quality(self):
        md = format_markdown(_make_profile(), _make_quality())
        assert "Quality" in md or "quality" in md
