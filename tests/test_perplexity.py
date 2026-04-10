"""Tests for quantbench.perplexity — perplexity-based quality scoring."""

from __future__ import annotations

import math

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
from quantbench.perplexity import (
    PerplexityDelta,
    PerplexityScore,
    estimate_perplexity_delta,
    format_quality_report,
    perplexity_from_logprobs,
    quality_score,
)


# ── helpers ────────────────────────────────────────────────────────────


def _make_profile(
    name: str = "model",
    avg_bpw: float = 16.0,
    method: QuantMethod = QuantMethod.FP16,
    layers: list[LayerInfo] | None = None,
) -> ModelProfile:
    tensors = [TensorInfo(name="w", shape=[100, 100], dtype=DType.F16)]
    return ModelProfile(
        name=name,
        format=QuantFormat.SAFETENSORS,
        total_params=10_000,
        tensors=tensors,
        layers=layers or [],
        quant=QuantProfile(method=method, avg_bits_per_weight=avg_bpw),
    )


def _make_layers() -> list[LayerInfo]:
    return [
        LayerInfo(name="model.layers.0.self_attn.q_proj", tensors=[
            TensorInfo(name="q", shape=[128, 128], dtype=DType.Q4_K_M),
        ]),
        LayerInfo(name="model.layers.0.mlp.gate_proj", tensors=[
            TensorInfo(name="g", shape=[128, 128], dtype=DType.Q4_K_M),
        ]),
        LayerInfo(name="model.embed_tokens", tensors=[
            TensorInfo(name="e", shape=[32000, 128], dtype=DType.Q4_K_M),
        ]),
    ]


# ── PerplexityScore dataclass ──────────────────────────────────────────


class TestPerplexityScore:
    def test_fields(self):
        s = PerplexityScore(value=5.2, num_tokens=100, log_likelihood=-165.0, normalized=True)
        assert s.value == 5.2
        assert s.num_tokens == 100
        assert s.log_likelihood == -165.0
        assert s.normalized is True


# ── perplexity_from_logprobs ───────────────────────────────────────────


class TestPerplexityFromLogprobs:
    def test_uniform_logprobs(self):
        # All tokens equally likely  ⇒  ppl = exp(-log(p))
        lp = [math.log(0.1)] * 50
        result = perplexity_from_logprobs(lp)
        assert isinstance(result, PerplexityScore)
        assert math.isclose(result.value, 10.0, rel_tol=1e-6)
        assert result.num_tokens == 50
        assert result.normalized is True

    def test_perfect_predictions(self):
        # log(1.0) = 0 ⇒ ppl = 1
        result = perplexity_from_logprobs([0.0] * 10)
        assert math.isclose(result.value, 1.0, rel_tol=1e-9)

    def test_single_token(self):
        result = perplexity_from_logprobs([math.log(0.5)])
        assert math.isclose(result.value, 2.0, rel_tol=1e-6)
        assert result.num_tokens == 1

    def test_empty_logprobs(self):
        result = perplexity_from_logprobs([])
        assert result.value == float("inf")
        assert result.num_tokens == 0

    def test_log_likelihood_sum(self):
        lps = [math.log(0.2), math.log(0.3), math.log(0.5)]
        result = perplexity_from_logprobs(lps)
        assert math.isclose(result.log_likelihood, sum(lps), rel_tol=1e-9)


# ── estimate_perplexity_delta ──────────────────────────────────────────


class TestEstimatePerplexityDelta:
    def test_same_precision_no_change(self):
        orig = _make_profile(avg_bpw=16.0)
        quant = _make_profile(avg_bpw=16.0)
        delta = estimate_perplexity_delta(orig, quant)
        assert isinstance(delta, PerplexityDelta)
        assert delta.estimated_ppl_increase_pct == 0.0
        assert delta.quality_retention == 1.0

    def test_lower_bits_increases_ppl(self):
        orig = _make_profile(avg_bpw=16.0)
        quant = _make_profile(avg_bpw=4.0)
        delta = estimate_perplexity_delta(orig, quant)
        assert delta.estimated_ppl_increase_pct > 0
        assert delta.quality_retention < 1.0

    def test_very_low_bits_high_penalty(self):
        orig = _make_profile(avg_bpw=16.0)
        q2 = _make_profile(avg_bpw=2.0)
        q4 = _make_profile(avg_bpw=4.0)
        d2 = estimate_perplexity_delta(orig, q2)
        d4 = estimate_perplexity_delta(orig, q4)
        assert d2.estimated_ppl_increase_pct > d4.estimated_ppl_increase_pct

    def test_per_layer_impact_present(self):
        layers = _make_layers()
        orig = _make_profile(avg_bpw=16.0)
        quant = _make_profile(avg_bpw=4.85, layers=layers)
        delta = estimate_perplexity_delta(orig, quant)
        assert len(delta.per_layer_impact) == len(layers)

    def test_attention_more_sensitive_than_ffn(self):
        layers = _make_layers()
        orig = _make_profile(avg_bpw=16.0)
        quant = _make_profile(avg_bpw=4.85, layers=layers)
        delta = estimate_perplexity_delta(orig, quant)
        attn_impact = delta.per_layer_impact["model.layers.0.self_attn.q_proj"]
        ffn_impact = delta.per_layer_impact["model.layers.0.mlp.gate_proj"]
        assert attn_impact > ffn_impact

    def test_embed_most_sensitive(self):
        layers = _make_layers()
        orig = _make_profile(avg_bpw=16.0)
        quant = _make_profile(avg_bpw=4.85, layers=layers)
        delta = estimate_perplexity_delta(orig, quant)
        embed_impact = delta.per_layer_impact["model.embed_tokens"]
        attn_impact = delta.per_layer_impact["model.layers.0.self_attn.q_proj"]
        assert embed_impact > attn_impact

    def test_quantized_better_than_original_clamps(self):
        # Edge case: quantized has MORE bits  ⇒  no degradation
        orig = _make_profile(avg_bpw=4.0)
        quant = _make_profile(avg_bpw=16.0)
        delta = estimate_perplexity_delta(orig, quant)
        assert delta.estimated_ppl_increase_pct == 0.0
        assert delta.quality_retention == 1.0


# ── quality_score ──────────────────────────────────────────────────────


class TestQualityScore:
    def test_full_precision_is_one(self):
        p = _make_profile(avg_bpw=16.0)
        assert quality_score(p) == 1.0

    def test_higher_precision_than_ref(self):
        p = _make_profile(avg_bpw=32.0)
        assert quality_score(p, reference_bits=16.0) == 1.0

    def test_lower_precision_below_one(self):
        p = _make_profile(avg_bpw=4.0)
        s = quality_score(p)
        assert 0.0 < s < 1.0

    def test_monotonically_decreasing(self):
        scores = []
        for bpw in [16, 8, 6, 4, 3, 2]:
            scores.append(quality_score(_make_profile(avg_bpw=float(bpw))))
        for a, b in zip(scores, scores[1:]):
            assert a >= b

    def test_custom_reference(self):
        p = _make_profile(avg_bpw=8.0)
        s8 = quality_score(p, reference_bits=8.0)
        assert s8 == 1.0

    def test_returns_float_in_range(self):
        for bpw in [1.0, 2.0, 4.0, 8.0, 16.0]:
            s = quality_score(_make_profile(avg_bpw=bpw))
            assert 0.0 <= s <= 1.0


# ── format_quality_report ─────────────────────────────────────────────


class TestFormatQualityReport:
    def test_contains_key_fields(self):
        delta = PerplexityDelta(
            original_bits=16.0,
            quantized_bits=4.85,
            estimated_ppl_increase_pct=3.14,
            quality_retention=0.9686,
            per_layer_impact={"layer.0.attn": 0.001, "layer.0.mlp": 0.0005},
        )
        report = format_quality_report(delta)
        assert "16.00" in report
        assert "4.85" in report
        assert "3.14" in report
        assert "Quality retention" in report
        assert "layer.0.attn" in report

    def test_empty_layers(self):
        delta = PerplexityDelta(
            original_bits=16.0,
            quantized_bits=8.0,
            estimated_ppl_increase_pct=1.0,
            quality_retention=0.99,
        )
        report = format_quality_report(delta)
        assert "Per-layer" not in report

    def test_many_layers_truncated(self):
        impacts = {f"layer.{i}": float(i) for i in range(20)}
        delta = PerplexityDelta(
            original_bits=16.0,
            quantized_bits=4.0,
            estimated_ppl_increase_pct=5.0,
            quality_retention=0.95,
            per_layer_impact=impacts,
        )
        report = format_quality_report(delta)
        assert "... and 10 more layers" in report
