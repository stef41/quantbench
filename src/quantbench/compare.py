"""Compare quantized model profiles across formats and methods."""

from __future__ import annotations

from typing import Any, Dict, List

from quantbench._types import ModelProfile


def compare_profiles(profile_a: ModelProfile, profile_b: ModelProfile) -> Dict[str, Any]:
    """Compare two model profiles and return a structured diff."""
    size_delta = profile_b.total_size_bytes - profile_a.total_size_bytes
    size_ratio = (
        profile_b.total_size_bytes / profile_a.total_size_bytes
        if profile_a.total_size_bytes > 0
        else 0.0
    )

    bpw_a = profile_a.quant.avg_bits_per_weight
    bpw_b = profile_b.quant.avg_bits_per_weight
    bpw_delta = bpw_b - bpw_a

    # Find tensors that differ in dtype
    tensors_a = {t.name: t for t in profile_a.tensors}
    tensors_b = {t.name: t for t in profile_b.tensors}
    common = set(tensors_a.keys()) & set(tensors_b.keys())

    dtype_changes: List[Dict[str, str]] = []
    for name in sorted(common):
        ta, tb = tensors_a[name], tensors_b[name]
        if ta.dtype != tb.dtype:
            dtype_changes.append({
                "tensor": name,
                "dtype_a": ta.dtype.value,
                "dtype_b": tb.dtype.value,
                "bpw_a": ta.bits_per_weight,
                "bpw_b": tb.bits_per_weight,
            })

    only_a = sorted(set(tensors_a.keys()) - set(tensors_b.keys()))
    only_b = sorted(set(tensors_b.keys()) - set(tensors_a.keys()))

    return {
        "model_a": profile_a.name,
        "model_b": profile_b.name,
        "format_a": profile_a.format.value,
        "format_b": profile_b.format.value,
        "size_a_gb": round(profile_a.size_gb, 3),
        "size_b_gb": round(profile_b.size_gb, 3),
        "size_delta_bytes": size_delta,
        "size_ratio": round(size_ratio, 3),
        "bpw_a": round(bpw_a, 2),
        "bpw_b": round(bpw_b, 2),
        "bpw_delta": round(bpw_delta, 2),
        "method_a": profile_a.quant.method.value,
        "method_b": profile_b.quant.method.value,
        "n_dtype_changes": len(dtype_changes),
        "dtype_changes": dtype_changes[:50],  # limit output
        "tensors_only_in_a": only_a[:20],
        "tensors_only_in_b": only_b[:20],
        "params_a": profile_a.total_params,
        "params_b": profile_b.total_params,
    }


def compare_formats(profiles: List[ModelProfile]) -> Dict[str, Any]:
    """Compare multiple quantization formats of the same base model.

    Returns a summary table and ranking by size vs quality trade-off.
    """
    if not profiles:
        return {"models": [], "ranking": []}

    rows: List[Dict[str, Any]] = []
    for p in profiles:
        rows.append({
            "name": p.name,
            "format": p.format.value,
            "method": p.quant.method.value,
            "size_gb": round(p.size_gb, 3),
            "avg_bpw": round(p.quant.avg_bits_per_weight, 2),
            "compression_ratio": round(p.compression_ratio, 2),
            "n_params": p.total_params,
            "n_quantized": p.quant.n_quantized_layers,
            "n_full_precision": p.quant.n_full_precision_layers,
        })

    # Rank by compression efficiency (higher compression at higher bpw = better quality)
    ranked = sorted(
        rows,
        key=lambda r: (r["avg_bpw"], -r["compression_ratio"]),
        reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "models": rows,
        "ranking": ranked,
        "n_models": len(profiles),
    }
