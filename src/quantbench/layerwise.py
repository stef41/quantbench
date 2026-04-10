"""Layerwise analysis — sensitivity estimation and mixed-quant recommendations."""

from __future__ import annotations

from typing import Any, Dict, List

from quantbench._types import DType, ModelProfile

# Sensitivity heuristics based on layer position and type
_SENSITIVITY_KEYWORDS = {
    "embed": 0.9,
    "lm_head": 0.9,
    "output": 0.8,
    "norm": 0.85,
    "ln": 0.85,
    "layernorm": 0.85,
    "rmsnorm": 0.85,
    "attn_q": 0.7,
    "attn_k": 0.7,
    "attn_v": 0.65,
    "attn_output": 0.6,
    "q_proj": 0.7,
    "k_proj": 0.7,
    "v_proj": 0.65,
    "o_proj": 0.6,
    "gate": 0.5,
    "up": 0.5,
    "down": 0.45,
    "ffn": 0.45,
    "mlp": 0.45,
}

# Position-based sensitivity multiplier (first and last layers are more sensitive)
_POSITION_CURVE = {
    "first_10pct": 1.3,
    "last_10pct": 1.2,
    "middle": 1.0,
}


def analyze_layers(profile: ModelProfile) -> List[Dict[str, Any]]:
    """Analyze each layer's quantization characteristics.

    Returns a list of dicts with layer name, param count, avg bpw,
    dominant dtype, and estimated sensitivity.
    """
    results: List[Dict[str, Any]] = []
    n_layers = len(profile.layers)

    for i, layer in enumerate(profile.layers):
        sensitivity = _estimate_layer_sensitivity(layer.name, i, n_layers)
        results.append({
            "name": layer.name,
            "n_params": layer.n_params,
            "n_tensors": len(layer.tensors),
            "size_bytes": layer.size_bytes,
            "avg_bits_per_weight": round(layer.avg_bits_per_weight, 2),
            "dominant_dtype": layer.dominant_dtype.value,
            "sensitivity": round(sensitivity, 3),
        })

    return results


def layer_sensitivity(profile: ModelProfile) -> Dict[str, float]:
    """Return a mapping of layer name → estimated sensitivity score (0-1)."""
    n_layers = len(profile.layers)
    return {
        layer.name: round(_estimate_layer_sensitivity(layer.name, i, n_layers), 3)
        for i, layer in enumerate(profile.layers)
    }


def recommend_mixed_quant(
    profile: ModelProfile,
    target_bpw: float = 4.5,
    high_quant: DType = DType.Q5_K_M,
    low_quant: DType = DType.Q4_K_M,
) -> Dict[str, Any]:
    """Recommend a mixed-precision quantization strategy.

    Assigns higher precision to more sensitive layers to meet a target
    average bits-per-weight while maximizing quality.
    """
    n_layers = len(profile.layers)
    if n_layers == 0:
        return {"strategy": [], "estimated_avg_bpw": 0.0}

    # Score layers by sensitivity
    scored: List[tuple] = []  # (sensitivity, layer_name, n_params)
    for i, layer in enumerate(profile.layers):
        sens = _estimate_layer_sensitivity(layer.name, i, n_layers)
        scored.append((sens, layer.name, layer.n_params))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Binary search for the cutoff: how many layers get high_quant
    total_params = sum(s[2] for s in scored)
    if total_params == 0:
        return {"strategy": [], "estimated_avg_bpw": 0.0}

    best_k = 0
    for k in range(len(scored) + 1):
        high_params = sum(scored[j][2] for j in range(k))
        low_params = total_params - high_params
        avg = (high_params * high_quant.bits_per_weight + low_params * low_quant.bits_per_weight) / total_params
        if avg <= target_bpw:
            best_k = k
            break
    else:
        best_k = len(scored)

    # Build strategy
    high_set = {scored[j][1] for j in range(best_k)}
    strategy = []
    for sens, name, n_params in scored:
        chosen = high_quant if name in high_set else low_quant
        strategy.append({
            "layer": name,
            "dtype": chosen.value,
            "sensitivity": round(sens, 3),
            "n_params": n_params,
        })

    # Compute final average
    high_params = sum(s[2] for s in scored[:best_k])
    low_params = total_params - high_params
    est_avg = (high_params * high_quant.bits_per_weight + low_params * low_quant.bits_per_weight) / total_params

    return {
        "strategy": strategy,
        "estimated_avg_bpw": round(est_avg, 2),
        "target_bpw": target_bpw,
        "high_quant": high_quant.value,
        "low_quant": low_quant.value,
        "n_high_precision_layers": best_k,
        "n_low_precision_layers": len(scored) - best_k,
    }


def _estimate_layer_sensitivity(name: str, position: int, total_layers: int) -> float:
    """Estimate how sensitive a layer is to quantization.

    Score 0-1 where 1 = most sensitive (should keep high precision).
    Uses name-based heuristics and position in the network.
    """
    # Base sensitivity from name
    base = 0.5
    name_lower = name.lower()
    for keyword, score in _SENSITIVITY_KEYWORDS.items():
        if keyword in name_lower:
            base = max(base, score)
            break

    # Position multiplier
    if total_layers <= 1:
        mult = 1.0
    else:
        frac = position / (total_layers - 1) if total_layers > 1 else 0.0
        if frac < 0.1:
            mult = _POSITION_CURVE["first_10pct"]
        elif frac > 0.9:
            mult = _POSITION_CURVE["last_10pct"]
        else:
            mult = _POSITION_CURVE["middle"]

    return min(base * mult, 1.0)
