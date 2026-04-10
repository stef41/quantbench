"""Quality prediction — estimate perplexity delta and quality score from quantization params."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from quantbench._types import DType, ModelProfile, QualityEstimate


# Empirical perplexity delta estimates per bits-per-weight
# Based on published quantization benchmarks (llama.cpp, GPTQ papers)
_PERPLEXITY_CURVE = {
    16.0: 0.0,
    8.5: 0.01,
    6.5: 0.03,
    5.5: 0.05,
    4.85: 0.08,
    4.5: 0.12,
    4.0: 0.18,
    3.9: 0.22,
    3.5: 0.35,
    3.35: 0.45,
    3.0: 0.65,
    2.5: 1.0,
    2.0: 1.8,
    1.5: 3.5,
}


def _interpolate_perplexity(bpw: float) -> float:
    """Interpolate expected perplexity delta from bits-per-weight."""
    points = sorted(_PERPLEXITY_CURVE.items(), reverse=True)

    if bpw >= points[0][0]:
        return points[0][1]
    if bpw <= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x2 <= bpw <= x1:
            t = (bpw - x2) / (x1 - x2) if x1 != x2 else 0.0
            return y1 * t + y2 * (1 - t)

    return 0.5  # fallback


def estimate_quality(profile: ModelProfile) -> QualityEstimate:
    """Estimate quantization quality from a model profile.

    Returns a QualityEstimate with predicted perplexity delta,
    quality score, risk level, and recommendations.
    """
    bpw = profile.quant.avg_bits_per_weight
    if bpw == 0:
        bpw = 16.0  # assume full precision if unknown

    ppl_delta = _interpolate_perplexity(bpw)

    # Quality score: 1.0 = perfect (no loss), 0.0 = terrible
    # Exponential decay from bpw
    quality = math.exp(-0.15 * max(0, 16.0 - bpw))
    quality = max(0.0, min(1.0, quality))

    # Risk level
    if ppl_delta < 0.05:
        risk = "low"
    elif ppl_delta < 0.15:
        risk = "medium"
    elif ppl_delta < 0.5:
        risk = "high"
    else:
        risk = "critical"

    # Find sensitive layers
    sensitive: List[str] = []
    for layer in profile.layers:
        name_lower = layer.name.lower()
        if any(kw in name_lower for kw in ("embed", "lm_head", "output", "norm")):
            if layer.avg_bits_per_weight < 6.0:
                sensitive.append(layer.name)

    # Recommendations
    recs: List[str] = []
    if bpw < 3.0:
        recs.append("Very aggressive quantization — expect significant quality loss")
        recs.append("Consider Q4_K_M or higher for production use")
    elif bpw < 4.0:
        recs.append("Aggressive quantization — test on your specific workload")
    elif bpw < 5.0:
        recs.append("Good balance of size and quality for most use cases")

    if sensitive:
        recs.append(f"Sensitive layers at low precision: {', '.join(sensitive[:5])}")
        recs.append("Consider mixed-quant with higher precision for embed/output layers")

    if profile.quant.n_full_precision_layers == 0 and len(profile.tensors) > 10:
        recs.append("No full-precision layers detected — norm layers may benefit from FP16")

    total_params = profile.total_params
    if total_params > 0:
        param_b = total_params / 1e9
        size_gb = profile.size_gb
        if param_b > 0:
            recs.append(f"Model: ~{param_b:.1f}B params, {size_gb:.1f} GB at {bpw:.1f} bpw")

    return QualityEstimate(
        model_name=profile.name,
        method=profile.quant.method.value,
        avg_bits_per_weight=round(bpw, 2),
        estimated_perplexity_delta=round(ppl_delta, 4),
        quality_score=round(quality, 4),
        risk_level=risk,
        sensitive_layers=sensitive,
        recommendations=recs,
    )


def perplexity_delta(bpw: float) -> float:
    """Estimate perplexity increase for a given bits-per-weight.

    Based on published quantization benchmarks.
    """
    return round(_interpolate_perplexity(bpw), 4)
