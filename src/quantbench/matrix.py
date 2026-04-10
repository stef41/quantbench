"""Quantization format comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QuantFormatSpec:
    """Specification of a quantization format."""

    name: str
    bits: int
    block_size: Optional[int] = None
    symmetric: bool = True
    description: str = ""

    @property
    def bytes_per_weight(self) -> float:
        """Effective bytes per weight (including overhead for block formats)."""
        base = self.bits / 8.0
        if self.block_size is not None and self.block_size > 0:
            # Block formats store a scale per block
            overhead = 2.0 / self.block_size  # 2 bytes for FP16 scale
            if not self.symmetric:
                overhead += 2.0 / self.block_size  # zero-point
            return base + overhead
        return base


@dataclass
class FormatComparison:
    """Result of comparing two quantization formats."""

    format_a: str
    format_b: str
    size_ratio: float
    accuracy_delta: float
    speed_ratio: float


KNOWN_FORMATS: Dict[str, QuantFormatSpec] = {
    "FP16": QuantFormatSpec(
        name="FP16", bits=16, block_size=None, symmetric=True,
        description="IEEE 754 half-precision floating point",
    ),
    "BF16": QuantFormatSpec(
        name="BF16", bits=16, block_size=None, symmetric=True,
        description="Brain floating point 16-bit",
    ),
    "INT8": QuantFormatSpec(
        name="INT8", bits=8, block_size=None, symmetric=True,
        description="8-bit integer quantization",
    ),
    "INT4": QuantFormatSpec(
        name="INT4", bits=4, block_size=128, symmetric=True,
        description="4-bit integer quantization with 128-element blocks",
    ),
    "GPTQ": QuantFormatSpec(
        name="GPTQ", bits=4, block_size=128, symmetric=False,
        description="GPTQ 4-bit with group size 128",
    ),
    "AWQ": QuantFormatSpec(
        name="AWQ", bits=4, block_size=128, symmetric=False,
        description="Activation-aware Weight Quantization 4-bit",
    ),
    "GGUF_Q4_0": QuantFormatSpec(
        name="GGUF_Q4_0", bits=4, block_size=32, symmetric=True,
        description="GGUF Q4_0: 4-bit quantization, 32-element blocks",
    ),
    "GGUF_Q5_1": QuantFormatSpec(
        name="GGUF_Q5_1", bits=5, block_size=32, symmetric=False,
        description="GGUF Q5_1: 5-bit quantization with zero-point, 32-element blocks",
    ),
    "NF4": QuantFormatSpec(
        name="NF4", bits=4, block_size=64, symmetric=True,
        description="NormalFloat 4-bit (QLoRA)",
    ),
}

# Heuristic accuracy retention scores per bit-width (relative to FP16 = 1.0).
# Higher is better. Accounts for typical perplexity retention on LLMs.
_ACCURACY_BY_BITS: Dict[int, float] = {
    16: 1.0,
    8: 0.995,
    5: 0.98,
    4: 0.96,
    3: 0.92,
    2: 0.82,
}

# Heuristic inference speed multiplier relative to FP16 = 1.0.
# Lower bit counts run faster due to reduced memory bandwidth.
_SPEED_BY_BITS: Dict[int, float] = {
    16: 1.0,
    8: 1.6,
    5: 2.0,
    4: 2.3,
    3: 2.5,
    2: 2.7,
}


def _accuracy_estimate(fmt: QuantFormatSpec) -> float:
    """Heuristic accuracy retention for a format (0-1 scale, FP16=1.0)."""
    bits = fmt.bits
    if bits in _ACCURACY_BY_BITS:
        score = _ACCURACY_BY_BITS[bits]
    else:
        # Linear interpolation between known points
        lower = max(b for b in _ACCURACY_BY_BITS if b <= bits)
        upper = min(b for b in _ACCURACY_BY_BITS if b >= bits)
        if lower == upper:
            score = _ACCURACY_BY_BITS[lower]
        else:
            t = (bits - lower) / (upper - lower)
            score = _ACCURACY_BY_BITS[lower] + t * (
                _ACCURACY_BY_BITS[upper] - _ACCURACY_BY_BITS[lower]
            )
    # Block-quantized formats with asymmetric quantization are slightly more accurate
    if fmt.block_size is not None and not fmt.symmetric:
        score = min(1.0, score + 0.005)
    return score


def _speed_estimate(fmt: QuantFormatSpec) -> float:
    """Heuristic speed multiplier (relative to FP16 = 1.0x)."""
    bits = fmt.bits
    if bits in _SPEED_BY_BITS:
        return _SPEED_BY_BITS[bits]
    lower = max(b for b in _SPEED_BY_BITS if b <= bits)
    upper = min(b for b in _SPEED_BY_BITS if b >= bits)
    if lower == upper:
        return _SPEED_BY_BITS[lower]
    t = (bits - lower) / (upper - lower)
    return _SPEED_BY_BITS[lower] + t * (
        _SPEED_BY_BITS[upper] - _SPEED_BY_BITS[lower]
    )


def _model_size_gb(fmt: QuantFormatSpec, base_size_gb: float) -> float:
    """Estimate model size in GB for a given format, relative to FP16 baseline."""
    return base_size_gb * fmt.bytes_per_weight / 2.0  # FP16 = 2 bytes/weight


class ComparisonMatrix:
    """N×N quantization format comparison matrix."""

    def __init__(self, formats: Optional[List[QuantFormatSpec]] = None) -> None:
        if formats is not None:
            self._formats: Dict[str, QuantFormatSpec] = {f.name: f for f in formats}
        else:
            self._formats = dict(KNOWN_FORMATS)

    @property
    def formats(self) -> Dict[str, QuantFormatSpec]:
        """Return the registered formats."""
        return dict(self._formats)

    def add_format(self, fmt: QuantFormatSpec) -> None:
        """Register a new quantization format."""
        self._formats[fmt.name] = fmt

    def compare_pair(
        self,
        fmt_a: str,
        fmt_b: str,
        model_size_gb: float = 7.0,
    ) -> FormatComparison:
        """Compare two formats and return size/accuracy/speed ratios.

        Ratios are expressed as A / B.  A size_ratio < 1 means A is smaller.
        An accuracy_delta > 0 means A is more accurate.
        A speed_ratio > 1 means A is faster.
        """
        if fmt_a not in self._formats:
            raise KeyError(f"Unknown format: {fmt_a!r}")
        if fmt_b not in self._formats:
            raise KeyError(f"Unknown format: {fmt_b!r}")

        fa = self._formats[fmt_a]
        fb = self._formats[fmt_b]

        size_a = _model_size_gb(fa, model_size_gb)
        size_b = _model_size_gb(fb, model_size_gb)
        size_ratio = size_a / size_b if size_b > 0 else float("inf")

        acc_a = _accuracy_estimate(fa)
        acc_b = _accuracy_estimate(fb)

        speed_a = _speed_estimate(fa)
        speed_b = _speed_estimate(fb)
        speed_ratio = speed_a / speed_b if speed_b > 0 else float("inf")

        return FormatComparison(
            format_a=fmt_a,
            format_b=fmt_b,
            size_ratio=round(size_ratio, 4),
            accuracy_delta=round(acc_a - acc_b, 6),
            speed_ratio=round(speed_ratio, 4),
        )

    def full_matrix(self) -> List[List[FormatComparison]]:
        """Generate an N×N comparison matrix."""
        names = sorted(self._formats)
        return [
            [self.compare_pair(a, b) for b in names]
            for a in names
        ]

    def rank_by(self, criterion: str = "size") -> List[QuantFormatSpec]:
        """Rank formats by a criterion: 'size', 'speed', or 'accuracy'."""
        fmts = list(self._formats.values())
        if criterion == "size":
            fmts.sort(key=lambda f: f.bytes_per_weight)
        elif criterion == "speed":
            fmts.sort(key=lambda f: _speed_estimate(f), reverse=True)
        elif criterion == "accuracy":
            fmts.sort(key=lambda f: _accuracy_estimate(f), reverse=True)
        else:
            raise ValueError(
                f"Unknown criterion: {criterion!r}. Choose 'size', 'speed', or 'accuracy'."
            )
        return fmts

    def recommend(self, constraints: Dict[str, Any]) -> List[QuantFormatSpec]:
        """Recommend formats matching constraints.

        Supported constraint keys:
          max_bits (int): maximum bit-width
          min_accuracy (float): minimum accuracy retention (0-1), FP16=1.0
          min_speed (float): minimum speed multiplier (FP16=1.0)
          max_size_gb (float): maximum model size in GB (requires base_size_gb)
          base_size_gb (float): FP16 model size for size calculations (default 7.0)
        """
        max_bits = constraints.get("max_bits", 32)
        min_accuracy = constraints.get("min_accuracy", 0.0)
        min_speed = constraints.get("min_speed", 0.0)
        max_size_gb = constraints.get("max_size_gb", float("inf"))
        base_size_gb = constraints.get("base_size_gb", 7.0)

        results = []
        for fmt in self._formats.values():
            if fmt.bits > max_bits:
                continue
            if _accuracy_estimate(fmt) < min_accuracy:
                continue
            if _speed_estimate(fmt) < min_speed:
                continue
            if _model_size_gb(fmt, base_size_gb) > max_size_gb:
                continue
            results.append(fmt)

        # Sort by accuracy descending, then speed descending
        results.sort(
            key=lambda f: (_accuracy_estimate(f), _speed_estimate(f)),
            reverse=True,
        )
        return results


def format_comparison_table(matrix: List[List[FormatComparison]]) -> str:
    """Render an N×N comparison matrix as an ASCII table.

    Each cell shows the size ratio of row-format vs column-format.
    """
    if not matrix or not matrix[0]:
        return "(empty matrix)"

    names = [row[0].format_a for row in matrix]
    col_width = max(len(n) for n in names) + 2
    cell_width = 8

    header = " " * col_width + "".join(n.center(cell_width) for n in names)
    separator = "-" * len(header)

    lines = [header, separator]
    for row in matrix:
        label = row[0].format_a.ljust(col_width)
        cells = []
        for comp in row:
            if comp.format_a == comp.format_b:
                cells.append("  ---  ".center(cell_width))
            else:
                cells.append(f"{comp.size_ratio:.2f}x".center(cell_width))
        lines.append(label + "".join(cells))

    return "\n".join(lines)
