"""GGUF importance matrix (imatrix) parsing and analysis."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from quantbench._types import QuantbenchError


@dataclass
class ImatrixEntry:
    """A single tensor entry in an importance matrix."""

    name: str
    num_values: int
    num_calls: int
    values: list[float]


@dataclass
class ImatrixData:
    """Parsed importance matrix data."""

    entries: list[ImatrixEntry]
    total_calls: int

    def by_name(self, name: str) -> Optional[ImatrixEntry]:
        """Look up an entry by tensor name."""
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def top_k(self, k: int) -> list[ImatrixEntry]:
        """Return the top-k entries ranked by mean importance value."""
        def _mean(entry: ImatrixEntry) -> float:
            if not entry.values:
                return 0.0
            return sum(entry.values) / len(entry.values)

        return sorted(self.entries, key=_mean, reverse=True)[:k]


@dataclass
class ImatrixAnalysis:
    """Result of analyzing an importance matrix."""

    total_tensors: int
    mean_importance_per_layer: Dict[str, float]
    variance_per_layer: Dict[str, float]
    outlier_layers: list[str]


def parse_imatrix(path: Union[str, Path]) -> ImatrixData:
    """Parse a binary imatrix file.

    Format per tensor:
        name_length (int32) + name (bytes) + num_values (int32)
        + num_calls (int32) + values (float32 * num_values)
    """
    path = Path(path)
    if not path.exists():
        raise QuantbenchError(f"imatrix file not found: {path}")

    entries: list[ImatrixEntry] = []
    total_calls = 0

    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    while offset < len(data):
        # name_length (int32)
        if offset + 4 > len(data):
            break
        (name_length,) = struct.unpack_from("<i", data, offset)
        offset += 4

        if name_length <= 0 or offset + name_length > len(data):
            raise QuantbenchError(
                f"Invalid name length {name_length} at offset {offset - 4}"
            )

        # name (bytes)
        name = data[offset : offset + name_length].decode("utf-8")
        offset += name_length

        # num_values (int32)
        if offset + 4 > len(data):
            raise QuantbenchError(f"Unexpected EOF reading num_values for '{name}'")
        (num_values,) = struct.unpack_from("<i", data, offset)
        offset += 4

        if num_values < 0:
            raise QuantbenchError(f"Invalid num_values {num_values} for '{name}'")

        # num_calls (int32)
        if offset + 4 > len(data):
            raise QuantbenchError(f"Unexpected EOF reading num_calls for '{name}'")
        (num_calls,) = struct.unpack_from("<i", data, offset)
        offset += 4

        # values (float32 * num_values)
        values_size = num_values * 4
        if offset + values_size > len(data):
            raise QuantbenchError(
                f"Unexpected EOF reading {num_values} values for '{name}'"
            )
        values = list(struct.unpack_from(f"<{num_values}f", data, offset))
        offset += values_size

        total_calls = max(total_calls, num_calls)
        entries.append(
            ImatrixEntry(
                name=name,
                num_values=num_values,
                num_calls=num_calls,
                values=values,
            )
        )

    return ImatrixData(entries=entries, total_calls=total_calls)


def analyze_imatrix(data: ImatrixData) -> ImatrixAnalysis:
    """Analyze an importance matrix and identify outlier layers."""
    mean_importance: Dict[str, float] = {}
    variance: Dict[str, float] = {}

    for entry in data.entries:
        if not entry.values:
            mean_importance[entry.name] = 0.0
            variance[entry.name] = 0.0
            continue
        n = len(entry.values)
        mu = sum(entry.values) / n
        var = sum((v - mu) ** 2 for v in entry.values) / n
        mean_importance[entry.name] = mu
        variance[entry.name] = var

    # Identify outliers: layers with mean importance > 2 std above global mean
    all_means = list(mean_importance.values())
    outlier_layers: list[str] = []
    if all_means:
        global_mean = sum(all_means) / len(all_means)
        global_var = sum((m - global_mean) ** 2 for m in all_means) / len(all_means)
        global_std = math.sqrt(global_var)
        threshold = global_mean + 2 * global_std
        outlier_layers = [
            name
            for name, mu in mean_importance.items()
            if mu > threshold
        ]

    return ImatrixAnalysis(
        total_tensors=len(data.entries),
        mean_importance_per_layer=mean_importance,
        variance_per_layer=variance,
        outlier_layers=outlier_layers,
    )


def format_imatrix_report(analysis: ImatrixAnalysis) -> str:
    """Format the analysis as a human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Importance Matrix Analysis Report")
    lines.append("=" * 60)
    lines.append(f"Total tensors: {analysis.total_tensors}")
    lines.append("")

    # Sort layers by importance descending
    sorted_layers = sorted(
        analysis.mean_importance_per_layer.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    lines.append("Layer Importance (descending):")
    lines.append("-" * 60)
    lines.append(f"{'Layer':<40} {'Mean':>10} {'Variance':>10}")
    lines.append("-" * 60)
    for name, mu in sorted_layers:
        var = analysis.variance_per_layer.get(name, 0.0)
        tag = " [OUTLIER]" if name in analysis.outlier_layers else ""
        lines.append(f"{name:<40} {mu:>10.6f} {var:>10.6f}{tag}")

    lines.append("")
    if analysis.outlier_layers:
        lines.append(f"Outlier layers ({len(analysis.outlier_layers)}):")
        for name in analysis.outlier_layers:
            lines.append(f"  - {name}")
    else:
        lines.append("No outlier layers detected.")

    lines.append("=" * 60)
    return "\n".join(lines)
