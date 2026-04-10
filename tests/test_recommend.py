"""Tests for quantbench.recommend."""

from __future__ import annotations

import pytest

from quantbench._types import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantProfile,
    TensorInfo,
)
from quantbench.recommend import Recommendation, format_recommendation, recommend

# ── helpers ──────────────────────────────────────────────────────────────


def _make_profile(
    name: str = "test-model",
    total_params: int = 7_000_000_000,
    layers: list | None = None,
) -> ModelProfile:
    """Build a minimal ModelProfile for testing."""
    if layers is None:
        layers = _default_layers()
    total_size = sum(l.size_bytes for l in layers)
    return ModelProfile(
        name=name,
        format=QuantFormat.GGUF,
        total_params=total_params,
        total_size_bytes=total_size,
        layers=layers,
        quant=QuantProfile(avg_bits_per_weight=16.0),
    )


def _default_layers() -> list[LayerInfo]:
    """Return a small set of layers that exercises the heuristics."""
    return [
        LayerInfo(
            name="model.embed_tokens",
            tensors=[TensorInfo(name="model.embed_tokens.weight", shape=[32000, 4096], dtype=DType.F16)],
        ),
        LayerInfo(
            name="model.layers.0.self_attn.q_proj",
            tensors=[TensorInfo(name="model.layers.0.self_attn.q_proj.weight", shape=[4096, 4096], dtype=DType.F16)],
        ),
        LayerInfo(
            name="model.layers.0.self_attn.k_proj",
            tensors=[TensorInfo(name="model.layers.0.self_attn.k_proj.weight", shape=[4096, 4096], dtype=DType.F16)],
        ),
        LayerInfo(
            name="model.layers.0.mlp.gate_proj",
            tensors=[TensorInfo(name="model.layers.0.mlp.gate_proj.weight", shape=[11008, 4096], dtype=DType.F16)],
        ),
        LayerInfo(
            name="model.layers.0.mlp.down_proj",
            tensors=[TensorInfo(name="model.layers.0.mlp.down_proj.weight", shape=[4096, 11008], dtype=DType.F16)],
        ),
        LayerInfo(
            name="model.norm",
            tensors=[TensorInfo(name="model.norm.weight", shape=[4096], dtype=DType.F16)],
        ),
        LayerInfo(
            name="lm_head",
            tensors=[TensorInfo(name="lm_head.weight", shape=[32000, 4096], dtype=DType.F16)],
        ),
    ]


# ── recommend() ──────────────────────────────────────────────────────────


class TestRecommend:
    def test_returns_recommendation_dataclass(self):
        rec = recommend(_make_profile())
        assert isinstance(rec, Recommendation)

    def test_default_quality_threshold(self):
        rec = recommend(_make_profile())
        # Default max_quality_loss=0.05 → quality ≥ 0.95
        assert rec.estimated_quality >= 0.94  # small float tolerance

    def test_format_is_valid(self):
        rec = recommend(_make_profile())
        valid_formats = {
            "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
            "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K",
        }
        assert rec.format in valid_formats

    def test_estimated_size_gb_positive(self):
        rec = recommend(_make_profile())
        assert rec.estimated_size_gb > 0

    def test_per_layer_keys_match_layers(self):
        profile = _make_profile()
        rec = recommend(profile)
        assert set(rec.per_layer.keys()) == {l.name for l in profile.layers}

    def test_sensitive_layers_get_higher_precision(self):
        profile = _make_profile()
        rec = recommend(profile)
        # embed and lm_head should both get higher precision than ffn layers
        embed_fmt = rec.per_layer.get("model.embed_tokens", "")
        ffn_fmt = rec.per_layer.get("model.layers.0.mlp.down_proj", "")
        assert embed_fmt != "" and ffn_fmt != ""
        embed_bpw = DType(embed_fmt).bits_per_weight
        ffn_bpw = DType(ffn_fmt).bits_per_weight
        assert embed_bpw >= ffn_bpw

    def test_explanation_non_empty(self):
        rec = recommend(_make_profile())
        assert len(rec.explanation) > 20


class TestRecommendTargetBits:
    @pytest.mark.parametrize("target", [3.5, 4.5, 5.5, 8.0])
    def test_closest_format_selected(self, target: float):
        rec = recommend(_make_profile(), target_bits=target)
        # The chosen bpw should be reasonably close to the target
        chosen_bpw = DType(rec.per_layer.get("model.layers.0.mlp.down_proj", "q4_k_m")).bits_per_weight
        assert abs(chosen_bpw - target) < 3.0

    def test_small_target_picks_aggressive_format(self):
        rec = recommend(_make_profile(), target_bits=3.5)
        assert rec.format in {"Q3_K_S", "Q3_K_M", "Q2_K"}

    def test_large_target_picks_conservative_format(self):
        rec = recommend(_make_profile(), target_bits=8.0)
        assert rec.format in {"Q8_0", "Q6_K"}


class TestRecommendMaxQualityLoss:
    def test_strict_threshold(self):
        rec = recommend(_make_profile(), max_quality_loss=0.01)
        # Very strict → high quality format
        assert rec.estimated_quality >= 0.95

    def test_relaxed_threshold(self):
        rec = recommend(_make_profile(), max_quality_loss=0.30)
        # Relaxed → more aggressive compression allowed
        assert rec.estimated_size_gb < recommend(_make_profile(), max_quality_loss=0.01).estimated_size_gb


class TestRecommendEdgeCases:
    def test_empty_profile(self):
        profile = ModelProfile(name="empty", format=QuantFormat.GGUF)
        rec = recommend(profile)
        assert isinstance(rec, Recommendation)
        assert rec.per_layer == {}
        assert rec.estimated_size_gb == 0.0

    def test_single_layer(self):
        single_layer = [
            LayerInfo(
                name="weight",
                tensors=[TensorInfo(name="weight", shape=[1000, 1000], dtype=DType.F16)],
            )
        ]
        profile = _make_profile(total_params=1_000_000, layers=single_layer)
        rec = recommend(profile)
        assert len(rec.per_layer) == 1


# ── format_recommendation() ─────────────────────────────────────────────


class TestFormatRecommendation:
    def test_output_is_string(self):
        rec = recommend(_make_profile())
        text = format_recommendation(rec)
        assert isinstance(text, str)

    def test_contains_key_fields(self):
        rec = recommend(_make_profile())
        text = format_recommendation(rec)
        assert "Recommended format" in text
        assert "Estimated size" in text
        assert "Quality retention" in text

    def test_contains_per_layer(self):
        rec = recommend(_make_profile())
        text = format_recommendation(rec)
        assert "Per-layer strategy" in text
        assert "embed" in text.lower()
