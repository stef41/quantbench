"""Edge-case and hardened tests for quantbench modules."""

import pytest

from quantbench._types import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    TensorInfo,
)
from quantbench.compare import compare_formats, compare_profiles
from quantbench.layerwise import (
    _estimate_layer_sensitivity,
    analyze_layers,
    layer_sensitivity,
    recommend_mixed_quant,
)
from quantbench.predict import _interpolate_perplexity, estimate_quality, perplexity_delta


def _profile(
    name="test",
    n_layers=3,
    dtype=DType.Q4_K_M,
    add_embed=False,
) -> ModelProfile:
    tensors = []
    layers = []
    for i in range(n_layers):
        t = TensorInfo(name=f"blk.{i}.weight", shape=[128, 128], dtype=dtype)
        tensors.append(t)
        layers.append(LayerInfo(name=f"blk.{i}", tensors=[t]))
    if add_embed:
        embed = TensorInfo(name="token_embd.weight", shape=[32000, 128], dtype=DType.F16)
        tensors.append(embed)
        layers.append(LayerInfo(name="token_embd", tensors=[embed]))
    total = sum(t.n_elements for t in tensors)
    size = sum(t.size_bytes for t in tensors)
    qp = QuantProfile(
        method=QuantMethod.GGML,
        avg_bits_per_weight=dtype.bits_per_weight,
        n_quantized_layers=n_layers,
    )
    return ModelProfile(
        name=name, format=QuantFormat.GGUF,
        total_params=total, total_size_bytes=size,
        tensors=tensors, layers=layers, quant=qp,
    )


# ---- types edge cases ----


class TestDTypeEdgeCases:
    def test_all_dtypes_have_bpw(self):
        for dt in DType:
            assert dt.bits_per_weight > 0

    def test_f32_bpw(self):
        assert DType.F32.bits_per_weight == 32.0

    def test_unknown_bpw(self):
        assert DType.UNKNOWN.bits_per_weight == 16.0

    def test_q2_k_is_small(self):
        assert DType.Q2_K.bits_per_weight < 4.0


class TestTensorInfoEdgeCases:
    def test_auto_computes_elements(self):
        t = TensorInfo(name="x", shape=[10, 20], dtype=DType.F16)
        assert t.n_elements == 200

    def test_auto_computes_size(self):
        t = TensorInfo(name="x", shape=[100], dtype=DType.F32)
        assert t.size_bytes == 100 * 4  # 32 bits = 4 bytes

    def test_empty_shape(self):
        t = TensorInfo(name="x", shape=[], dtype=DType.F16)
        assert t.n_elements == 0

    def test_compression_ratio(self):
        t = TensorInfo(name="x", shape=[100], dtype=DType.Q4_0)
        assert t.compression_ratio > 1.0

    def test_compression_ratio_f32(self):
        t = TensorInfo(name="x", shape=[100], dtype=DType.F32)
        assert t.compression_ratio == 1.0


class TestLayerInfoEdgeCases:
    def test_empty_layer(self):
        l = LayerInfo(name="empty")
        assert l.n_params == 0
        assert l.size_bytes == 0
        assert l.avg_bits_per_weight == 0.0
        assert l.dominant_dtype == DType.UNKNOWN

    def test_mixed_dtypes(self):
        t1 = TensorInfo(name="a", shape=[100], dtype=DType.F16)
        t2 = TensorInfo(name="b", shape=[1000], dtype=DType.Q4_0)
        l = LayerInfo(name="mixed", tensors=[t1, t2])
        # Q4_0 dominates since 1000 > 100 elements
        assert l.dominant_dtype == DType.Q4_0
        # avg bpw should be weighted
        assert l.avg_bits_per_weight < 16.0


class TestModelProfileEdgeCases:
    def test_empty_profile(self):
        p = ModelProfile(name="empty", format=QuantFormat.UNKNOWN)
        assert p.size_gb == 0.0
        assert p.compression_ratio == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        p = _profile("rt_test", 3)
        d = p.to_dict()
        p2 = ModelProfile.from_dict(d)
        assert p2.name == "rt_test"
        assert p2.format == QuantFormat.GGUF
        assert p2.total_params == p.total_params

    def test_from_dict_missing_fields(self):
        d = {"name": "min", "format": "unknown"}
        p = ModelProfile.from_dict(d)
        assert p.name == "min"
        assert p.total_params == 0


# ---- compare edge cases ----


class TestCompareEdgeCases:
    def test_compare_same_profile(self):
        p = _profile("same")
        result = compare_profiles(p, p)
        assert result["size_delta_bytes"] == 0
        assert result["size_ratio"] == 1.0
        assert result["bpw_delta"] == 0.0
        assert result["n_dtype_changes"] == 0

    def test_compare_different_dtypes(self):
        p_a = _profile("a", dtype=DType.F16)
        p_b = _profile("b", dtype=DType.Q4_0)
        result = compare_profiles(p_a, p_b)
        assert result["bpw_delta"] < 0  # Q4 < F16
        assert result["n_dtype_changes"] > 0

    def test_compare_empty_profiles(self):
        p_a = ModelProfile(name="a", format=QuantFormat.GGUF)
        p_b = ModelProfile(name="b", format=QuantFormat.GGUF)
        result = compare_profiles(p_a, p_b)
        assert result["size_delta_bytes"] == 0

    def test_compare_formats_empty(self):
        result = compare_formats([])
        assert result["models"] == []

    def test_compare_formats_single(self):
        result = compare_formats([_profile("single")])
        assert result["n_models"] == 1
        assert len(result["ranking"]) == 1

    def test_compare_formats_ranking(self):
        profiles = [
            _profile("fp16", dtype=DType.F16),
            _profile("q4", dtype=DType.Q4_K_M),
            _profile("q2", dtype=DType.Q2_K),
        ]
        result = compare_formats(profiles)
        assert result["n_models"] == 3
        # Ranking should be sorted by bpw descending
        bpws = [r["avg_bpw"] for r in result["ranking"]]
        assert bpws == sorted(bpws, reverse=True)


# ---- layerwise edge cases ----


class TestLayerwiseEdgeCases:
    def test_single_layer_sensitivity(self):
        t = TensorInfo(name="blk.0.weight", shape=[100], dtype=DType.Q4_K_M)
        layers = [LayerInfo(name="blk.0", tensors=[t])]
        profile = ModelProfile(
            name="single", format=QuantFormat.GGUF,
            total_params=100, total_size_bytes=t.size_bytes,
            tensors=[t], layers=layers,
        )
        sens = layer_sensitivity(profile)
        assert len(sens) == 1
        assert 0 <= sens["blk.0"] <= 1.0

    def test_sensitivity_keywords(self):
        # embed should be more sensitive than ffn
        s_embed = _estimate_layer_sensitivity("embed_tokens", 0, 10)
        s_ffn = _estimate_layer_sensitivity("blk.5.ffn", 5, 10)
        assert s_embed > s_ffn

    def test_position_curve_first_layer(self):
        s_first = _estimate_layer_sensitivity("blk.0.weight", 0, 100)
        s_mid = _estimate_layer_sensitivity("blk.50.weight", 50, 100)
        assert s_first > s_mid

    def test_position_curve_last_layer(self):
        s_last = _estimate_layer_sensitivity("blk.99.weight", 99, 100)
        s_mid = _estimate_layer_sensitivity("blk.50.weight", 50, 100)
        assert s_last > s_mid

    def test_recommend_mixed_quant_many_layers(self):
        profile = _profile("big", n_layers=20, add_embed=True)
        result = recommend_mixed_quant(profile, target_bpw=5.0)
        assert len(result["strategy"]) == len(profile.layers)

    def test_analyze_layers_sensitivity_bounded(self):
        profile = _profile("bounded", n_layers=10, add_embed=True)
        result = analyze_layers(profile)
        for row in result:
            assert 0.0 <= row["sensitivity"] <= 1.0


# ---- predict edge cases ----


class TestPredictEdgeCases:
    def test_perplexity_delta_boundary_values(self):
        # Test exact curve points
        assert perplexity_delta(16.0) == pytest.approx(0.0)
        assert perplexity_delta(4.0) > 0

    def test_interpolation_between_points(self):
        # Between 4.0 and 4.5 should interpolate
        delta = _interpolate_perplexity(4.25)
        assert _interpolate_perplexity(4.0) >= delta >= _interpolate_perplexity(4.5)

    def test_quality_score_bounded(self):
        for bpw_dtype in [DType.F16, DType.Q4_K_M, DType.Q2_K, DType.IQ1_S]:
            profile = _profile("qs", dtype=bpw_dtype)
            qe = estimate_quality(profile)
            assert 0.0 <= qe.quality_score <= 1.0

    def test_risk_levels(self):
        risk_levels = set()
        for dt in [DType.F16, DType.Q5_K_M, DType.Q4_K_M, DType.Q2_K, DType.IQ1_S]:
            profile = _profile("risk", dtype=dt)
            qe = estimate_quality(profile)
            risk_levels.add(qe.risk_level)
        assert risk_levels.issubset({"low", "medium", "high", "critical"})

    def test_zero_bpw_fallback(self):
        profile = ModelProfile(
            name="zero", format=QuantFormat.GGUF,
            quant=QuantProfile(avg_bits_per_weight=0.0),
        )
        qe = estimate_quality(profile)
        # Should fallback to 16.0 bpw
        assert qe.risk_level == "low"

    def test_recommendations_aggressive(self):
        profile = _profile("agg", dtype=DType.IQ2_XXS)
        qe = estimate_quality(profile)
        assert any("aggressive" in r.lower() or "quality loss" in r.lower() for r in qe.recommendations)

    def test_sensitive_layers_with_embed(self):
        embed = TensorInfo(name="embed_tokens.weight", shape=[32000, 128], dtype=DType.Q4_0)
        lm_head = TensorInfo(name="lm_head.weight", shape=[32000, 128], dtype=DType.Q4_0)
        layers = [
            LayerInfo(name="embed_tokens", tensors=[embed]),
            LayerInfo(name="lm_head", tensors=[lm_head]),
        ]
        total = embed.n_elements + lm_head.n_elements
        size = embed.size_bytes + lm_head.size_bytes
        profile = ModelProfile(
            name="sensitive", format=QuantFormat.GGUF,
            total_params=total, total_size_bytes=size,
            tensors=[embed, lm_head], layers=layers,
            quant=QuantProfile(avg_bits_per_weight=4.5),
        )
        qe = estimate_quality(profile)
        assert len(qe.sensitive_layers) >= 1
