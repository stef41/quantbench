"""Perplexity-based quality scoring for quantized models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from quantbench._types import ModelProfile


@dataclass
class PerplexityScore:
    """Result of a perplexity computation."""

    value: float
    num_tokens: int
    log_likelihood: float
    normalized: bool


@dataclass
class PerplexityDelta:
    """Estimated perplexity change between original and quantized models."""

    original_bits: float
    quantized_bits: float
    estimated_ppl_increase_pct: float
    quality_retention: float  # 0-1
    per_layer_impact: Dict[str, float] = field(default_factory=dict)


# ── Heuristic constants ───────────────────────────────────────────────

# Attention layers are ~2× more sensitive than FFN layers.
_ATTN_SENSITIVITY = 2.0
_FFN_SENSITIVITY = 1.0

# Empirical base: each halving of bpw roughly doubles the ppl penalty.
_BASE_PPL_RATE = 0.04  # ppl % increase per bit of precision lost


def _layer_sensitivity(name: str) -> float:
    """Return a sensitivity multiplier based on the layer name."""
    low = name.lower()
    if any(kw in low for kw in ("attn", "attention", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        return _ATTN_SENSITIVITY
    if any(kw in low for kw in ("embed", "lm_head", "output", "norm")):
        return _ATTN_SENSITIVITY * 1.5  # most sensitive
    return _FFN_SENSITIVITY


def _bits_for_profile(profile: ModelProfile) -> float:
    bpw = profile.quant.avg_bits_per_weight
    return bpw if bpw > 0 else 16.0


def _per_layer_impact(profile: ModelProfile, ref_bits: float) -> Dict[str, float]:
    impacts: Dict[str, float] = {}
    for layer in profile.layers:
        lbpw = layer.avg_bits_per_weight
        if lbpw <= 0:
            lbpw = ref_bits
        bit_drop = max(0.0, ref_bits - lbpw)
        sens = _layer_sensitivity(layer.name)
        impacts[layer.name] = round(bit_drop * sens * _BASE_PPL_RATE, 6)
    return impacts


def _ppl_increase_pct(original_bits: float, quantized_bits: float) -> float:
    """Estimate perplexity % increase from precision loss.

    Uses an exponential heuristic: ppl_pct ≈ base_rate * 2^(bit_drop / scale) - base_rate
    so small drops are nearly linear while large drops blow up.
    """
    bit_drop = max(0.0, original_bits - quantized_bits)
    if bit_drop == 0:
        return 0.0
    # scale factor: a 4-bit drop ≈ 1 doubling
    return _BASE_PPL_RATE * (2.0 ** (bit_drop / 4.0) - 1.0) * 100.0


# ── Public API ────────────────────────────────────────────────────────


def estimate_perplexity_delta(
    profile_original: ModelProfile,
    profile_quantized: ModelProfile,
) -> PerplexityDelta:
    """Estimate the perplexity delta between an original and quantized model.

    Uses a heuristic: lower bit-widths increase perplexity roughly
    exponentially, and attention layers are more sensitive than FFN layers.
    """
    orig_bits = _bits_for_profile(profile_original)
    quant_bits = _bits_for_profile(profile_quantized)

    ppl_pct = _ppl_increase_pct(orig_bits, quant_bits)
    retention = max(0.0, min(1.0, 1.0 - ppl_pct / 100.0))

    impacts = _per_layer_impact(profile_quantized, orig_bits)

    return PerplexityDelta(
        original_bits=round(orig_bits, 4),
        quantized_bits=round(quant_bits, 4),
        estimated_ppl_increase_pct=round(ppl_pct, 4),
        quality_retention=round(retention, 6),
        per_layer_impact=impacts,
    )


def quality_score(profile: ModelProfile, reference_bits: float = 16.0) -> float:
    """Score model quality relative to a reference precision.

    Returns a float in [0, 1].  1.0 means no quality loss (same precision
    as reference), 0.0 means severe degradation.
    """
    bpw = _bits_for_profile(profile)
    if bpw >= reference_bits:
        return 1.0
    bit_drop = reference_bits - bpw
    # Exponential decay: exp(-k * bit_drop^1.5) gives a smooth curve
    score = math.exp(-0.06 * (bit_drop ** 1.5))
    return round(max(0.0, min(1.0, score)), 6)


def perplexity_from_logprobs(logprobs: Sequence[float]) -> PerplexityScore:
    """Compute perplexity from a list of token log-probabilities.

    Standard formula:  ppl = exp( -1/N * Σ log p_i )
    """
    n = len(logprobs)
    if n == 0:
        return PerplexityScore(value=float("inf"), num_tokens=0, log_likelihood=0.0, normalized=True)

    total_ll = sum(logprobs)
    avg_neg_ll = -total_ll / n
    ppl = math.exp(avg_neg_ll)

    return PerplexityScore(
        value=ppl,
        num_tokens=n,
        log_likelihood=total_ll,
        normalized=True,
    )


def format_quality_report(delta: PerplexityDelta) -> str:
    """Format a PerplexityDelta as a human-readable report."""
    lines = [
        "Perplexity Quality Report",
        "=" * 40,
        f"Original precision : {delta.original_bits:.2f} bits",
        f"Quantized precision: {delta.quantized_bits:.2f} bits",
        f"Est. PPL increase  : {delta.estimated_ppl_increase_pct:.2f}%",
        f"Quality retention  : {delta.quality_retention:.4f}",
    ]

    if delta.per_layer_impact:
        lines.append("")
        lines.append("Per-layer impact:")
        sorted_layers = sorted(delta.per_layer_impact.items(), key=lambda x: x[1], reverse=True)
        for name, impact in sorted_layers[:10]:
            lines.append(f"  {name:40s} {impact:.6f}")
        if len(sorted_layers) > 10:
            lines.append(f"  ... and {len(sorted_layers) - 10} more layers")

    return "\n".join(lines)
