"""Quantization recommendation engine — suggest optimal format based on model profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from quantbench._types import DType, ModelProfile
from quantbench.layerwise import layer_sensitivity


@dataclass
class Recommendation:
    """Quantization recommendation for a model."""

    format: str
    estimated_size_gb: float
    estimated_quality: float  # 0-1 quality retention score
    per_layer: Dict[str, str]  # layer name -> recommended format
    explanation: str


# Candidate formats ordered from highest quality to most compressed.
# Each entry: (label, target DType, approx bpw, quality baseline)
_CANDIDATES: List[tuple] = [
    ("Q8_0", DType.Q8_0, 8.5, 0.99),
    ("Q6_K", DType.Q6_K, 6.5625, 0.97),
    ("Q5_K_M", DType.Q5_K_M, 5.5, 0.95),
    ("Q5_K_S", DType.Q5_K_S, 5.5, 0.94),
    ("Q4_K_M", DType.Q4_K_M, 4.85, 0.90),
    ("Q4_K_S", DType.Q4_K_S, 4.5, 0.87),
    ("Q3_K_M", DType.Q3_K_M, 3.9, 0.80),
    ("Q3_K_S", DType.Q3_K_S, 3.5, 0.74),
    ("Q2_K", DType.Q2_K, 3.35, 0.60),
]

# Layers matching these keywords are kept at a higher precision tier.
_HIGH_PRECISION_KEYWORDS = {"embed", "lm_head", "output", "norm", "layernorm", "rmsnorm"}


def recommend(
    profile: ModelProfile,
    target_bits: Optional[float] = None,
    max_quality_loss: float = 0.05,
) -> Recommendation:
    """Recommend a quantization strategy for *profile*.

    Parameters
    ----------
    profile:
        A :class:`ModelProfile` produced by the profiling helpers.
    target_bits:
        If given, pick the format closest to this bits-per-weight.
        Overrides *max_quality_loss*.
    max_quality_loss:
        Maximum tolerable quality loss (0-1).  ``0.05`` means we want
        ≥ 95 % quality retention.  Ignored when *target_bits* is set.

    Returns
    -------
    Recommendation
    """
    total_params = profile.total_params
    if total_params == 0:
        total_params = sum(l.n_params for l in profile.layers)

    # --- pick the overall format ----------------------------------------
    if target_bits is not None:
        chosen = _closest_candidate(target_bits)
    else:
        min_quality = 1.0 - max_quality_loss
        chosen = _best_candidate_for_quality(min_quality)

    label, dtype, bpw, quality_base = chosen

    # --- estimate size --------------------------------------------------
    if total_params > 0:
        estimated_size_bytes = total_params * bpw / 8
        estimated_size_gb = round(estimated_size_bytes / (1024**3), 3)
    else:
        estimated_size_gb = 0.0

    # --- per-layer recommendations (mixed-quant) ------------------------
    sensitivities = layer_sensitivity(profile) if profile.layers else {}
    high_dtype = _one_tier_higher(dtype)
    per_layer: Dict[str, str] = {}
    for layer_name, sens in sensitivities.items():
        if _is_sensitive_layer(layer_name) or sens >= 0.75:
            per_layer[layer_name] = high_dtype.value
        else:
            per_layer[layer_name] = dtype.value

    # --- adjust quality estimate for mixed-quant benefit -----------------
    estimated_quality = _adjust_quality(quality_base, per_layer, sensitivities)

    # --- human-readable explanation -------------------------------------
    explanation = _build_explanation(
        profile, label, bpw, estimated_size_gb, estimated_quality,
        per_layer, sensitivities, target_bits, max_quality_loss,
    )

    return Recommendation(
        format=label,
        estimated_size_gb=estimated_size_gb,
        estimated_quality=round(estimated_quality, 4),
        per_layer=per_layer,
        explanation=explanation,
    )


def format_recommendation(rec: Recommendation) -> str:
    """Return a human-readable multi-line summary of *rec*."""
    lines = [
        f"Recommended format : {rec.format}",
        f"Estimated size     : {rec.estimated_size_gb:.2f} GB",
        f"Quality retention  : {rec.estimated_quality * 100:.1f}%",
        "",
    ]
    if rec.per_layer:
        lines.append("Per-layer strategy:")
        for layer, fmt in rec.per_layer.items():
            lines.append(f"  {layer}: {fmt}")
        lines.append("")
    lines.append(rec.explanation)
    return "\n".join(lines)


# ── internal helpers ─────────────────────────────────────────────────────


def _closest_candidate(target_bpw: float) -> tuple:
    best = _CANDIDATES[0]
    best_dist = abs(best[2] - target_bpw)
    for c in _CANDIDATES[1:]:
        d = abs(c[2] - target_bpw)
        if d < best_dist:
            best = c
            best_dist = d
    return best


def _best_candidate_for_quality(min_quality: float) -> tuple:
    # Walk from most compressed to least; pick the most compressed that
    # still meets the quality threshold.
    for c in reversed(_CANDIDATES):
        if c[3] >= min_quality:
            return c
    # Nothing meets the bar — fall back to highest quality.
    return _CANDIDATES[0]


def _one_tier_higher(dtype: DType) -> DType:
    """Return the next-higher-quality DType tier."""
    order = [c[1] for c in _CANDIDATES]
    try:
        idx = order.index(dtype)
    except ValueError:
        return dtype
    return order[max(0, idx - 1)]


def _is_sensitive_layer(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in _HIGH_PRECISION_KEYWORDS)


def _adjust_quality(
    base: float,
    per_layer: Dict[str, str],
    sensitivities: Dict[str, float],
) -> float:
    """Bump quality estimate when sensitive layers get higher precision."""
    if not per_layer or not sensitivities:
        return base

    total_sens = sum(sensitivities.values())
    if total_sens == 0:
        return base

    # Fraction of total sensitivity weight that got upgraded.
    upgraded_sens = sum(
        sensitivities.get(ln, 0.0)
        for ln, fmt in per_layer.items()
        # "upgraded" if the format differs from the majority
        if fmt != _mode_value(per_layer)
    )
    benefit = 0.02 * (upgraded_sens / total_sens)
    return min(base + benefit, 1.0)


def _mode_value(d: Dict[str, str]) -> str:
    """Most common value in a dict."""
    counts: Dict[str, int] = {}
    for v in d.values():
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else ""


def _build_explanation(
    profile: ModelProfile,
    label: str,
    bpw: float,
    size_gb: float,
    quality: float,
    per_layer: Dict[str, str],
    sensitivities: Dict[str, float],
    target_bits: Optional[float],
    max_quality_loss: float,
) -> str:
    parts: List[str] = []
    param_b = profile.total_params / 1e9 if profile.total_params else 0
    if param_b > 0:
        parts.append(f"For a ~{param_b:.1f}B-parameter model, ")
    else:
        parts.append("Based on the model profile, ")

    if target_bits is not None:
        parts.append(
            f"{label} (≈{bpw:.1f} bpw) is the closest match to the "
            f"requested {target_bits:.1f} bits-per-weight target."
        )
    else:
        parts.append(
            f"{label} (≈{bpw:.1f} bpw) is recommended to stay within "
            f"{max_quality_loss * 100:.0f}% quality loss."
        )

    n_upgraded = sum(
        1 for fmt in per_layer.values() if fmt != _mode_value(per_layer)
    )
    if n_upgraded:
        parts.append(
            f" {n_upgraded} sensitive layer(s) are assigned higher precision "
            f"for mixed-quantization."
        )

    parts.append(
        f" Estimated size: {size_gb:.2f} GB, quality retention: {quality * 100:.1f}%."
    )
    return "".join(parts)
