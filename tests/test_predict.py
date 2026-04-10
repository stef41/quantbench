"""Tests for quantbench.predict."""

import pytest

from quantbench._types import DType, LayerInfo, ModelProfile, QuantFormat, QuantProfile, TensorInfo
from quantbench.predict import estimate_quality, perplexity_delta


def _make_profile(bpw_dtype: DType = DType.Q4_K_M) -> ModelProfile:
    tensors = [
        TensorInfo(name=f"blk.{i}.weight", shape=[256, 256], dtype=bpw_dtype)
        for i in range(5)
    ]
    layers = [LayerInfo(name=f"blk.{i}", tensors=[t]) for i, t in enumerate(tensors)]
    total = sum(t.n_elements for t in tensors)
    size = sum(t.size_bytes for t in tensors)
    qp = QuantProfile(
        avg_bits_per_weight=bpw_dtype.bits_per_weight,
        n_quantized_layers=5,
    )
    return ModelProfile(
        name="test", format=QuantFormat.GGUF,
        total_params=total, total_size_bytes=size,
        tensors=tensors, layers=layers, quant=qp,
    )


class TestEstimateQuality:
    def test_high_precision(self):
        profile = _make_profile(DType.F16)
        qe = estimate_quality(profile)
        assert qe.quality_score > 0.9
        assert qe.risk_level == "low"
        assert qe.estimated_perplexity_delta < 0.05

    def test_aggressive_quant(self):
        profile = _make_profile(DType.Q2_K)
        qe = estimate_quality(profile)
        assert qe.quality_score < 0.8
        assert qe.risk_level in ("high", "critical")

    def test_moderate_quant(self):
        profile = _make_profile(DType.Q4_K_M)
        qe = estimate_quality(profile)
        assert qe.risk_level in ("low", "medium", "high")
        assert qe.estimated_perplexity_delta > 0.0

    def test_has_recommendations(self):
        profile = _make_profile(DType.Q2_K)
        qe = estimate_quality(profile)
        assert len(qe.recommendations) > 0

    def test_model_name(self):
        profile = _make_profile()
        qe = estimate_quality(profile)
        assert qe.model_name == "test"

    def test_sensitive_layers_detected(self):
        # Create profile with embed layer at low precision
        embed = TensorInfo(name="embed.weight", shape=[32000, 128], dtype=DType.Q4_0)
        layers = [LayerInfo(name="embed", tensors=[embed])]
        profile = ModelProfile(
            name="test", format=QuantFormat.GGUF,
            total_params=embed.n_elements, total_size_bytes=embed.size_bytes,
            tensors=[embed], layers=layers,
            quant=QuantProfile(avg_bits_per_weight=4.5),
        )
        qe = estimate_quality(profile)
        assert "embed" in qe.sensitive_layers


class TestPerplexityDelta:
    def test_fp16(self):
        assert perplexity_delta(16.0) == pytest.approx(0.0)

    def test_q4(self):
        delta = perplexity_delta(4.5)
        assert delta > 0.0
        assert delta < 1.0

    def test_monotonic(self):
        """Lower bpw should give higher perplexity delta."""
        deltas = [perplexity_delta(bpw) for bpw in [16.0, 8.0, 4.0, 2.0]]
        for i in range(len(deltas) - 1):
            assert deltas[i] <= deltas[i + 1]

    def test_extreme_low(self):
        delta = perplexity_delta(1.0)
        assert delta > 1.0

    def test_extreme_high(self):
        delta = perplexity_delta(32.0)
        assert delta == pytest.approx(0.0)
