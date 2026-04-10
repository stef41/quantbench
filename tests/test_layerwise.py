"""Tests for quantbench.layerwise."""

import pytest
from quantbench._types import DType, LayerInfo, ModelProfile, QuantFormat, TensorInfo
from quantbench.layerwise import analyze_layers, layer_sensitivity, recommend_mixed_quant


def _make_profile(n_layers: int = 5, dtype: DType = DType.Q4_K_M) -> ModelProfile:
    """Create a test profile with n numbered layers."""
    tensors = []
    layers = []
    for i in range(n_layers):
        layer_tensors = [
            TensorInfo(name=f"blk.{i}.attn_q.weight", shape=[128, 128], dtype=dtype),
            TensorInfo(name=f"blk.{i}.attn_v.weight", shape=[128, 128], dtype=dtype),
            TensorInfo(name=f"blk.{i}.ffn.weight", shape=[128, 512], dtype=dtype),
        ]
        tensors.extend(layer_tensors)
        layers.append(LayerInfo(name=f"blk.{i}", tensors=layer_tensors))

    # Add embed/output
    embed = TensorInfo(name="token_embd.weight", shape=[32000, 128], dtype=DType.F16)
    output = TensorInfo(name="output.weight", shape=[32000, 128], dtype=DType.F16)
    tensors.extend([embed, output])
    layers.append(LayerInfo(name="other", tensors=[embed, output]))

    total = sum(t.n_elements for t in tensors)
    size = sum(t.size_bytes for t in tensors)

    return ModelProfile(
        name="test-model", format=QuantFormat.GGUF,
        total_params=total, total_size_bytes=size,
        tensors=tensors, layers=layers,
    )


class TestAnalyzeLayers:
    def test_returns_all_layers(self):
        profile = _make_profile(3)
        result = analyze_layers(profile)
        assert len(result) == len(profile.layers)

    def test_fields(self):
        profile = _make_profile(2)
        result = analyze_layers(profile)
        row = result[0]
        assert "name" in row
        assert "n_params" in row
        assert "avg_bits_per_weight" in row
        assert "sensitivity" in row
        assert "dominant_dtype" in row

    def test_sensitivity_range(self):
        profile = _make_profile(10)
        result = analyze_layers(profile)
        for row in result:
            assert 0.0 <= row["sensitivity"] <= 1.0

    def test_empty(self):
        profile = ModelProfile(name="empty", format=QuantFormat.GGUF)
        result = analyze_layers(profile)
        assert result == []


class TestLayerSensitivity:
    def test_returns_all(self):
        profile = _make_profile(3)
        sens = layer_sensitivity(profile)
        assert len(sens) == len(profile.layers)

    def test_values_bounded(self):
        profile = _make_profile(10)
        sens = layer_sensitivity(profile)
        for name, score in sens.items():
            assert 0.0 <= score <= 1.0

    def test_embed_sensitive(self):
        tensors = [TensorInfo(name="embed.weight", shape=[100], dtype=DType.F16)]
        layers = [LayerInfo(name="embed", tensors=tensors)]
        profile = ModelProfile(
            name="test", format=QuantFormat.GGUF,
            tensors=tensors, layers=layers,
        )
        sens = layer_sensitivity(profile)
        assert sens["embed"] > 0.5


class TestRecommendMixedQuant:
    def test_basic(self):
        profile = _make_profile(5)
        result = recommend_mixed_quant(profile, target_bpw=5.0)
        assert "strategy" in result
        assert "estimated_avg_bpw" in result
        assert len(result["strategy"]) > 0

    def test_target_respected(self):
        profile = _make_profile(10)
        result = recommend_mixed_quant(profile, target_bpw=5.0)
        # The estimated avg should be close to or below target
        assert result["estimated_avg_bpw"] <= 6.0  # generous bound

    def test_empty(self):
        profile = ModelProfile(name="empty", format=QuantFormat.GGUF)
        result = recommend_mixed_quant(profile)
        assert result["strategy"] == []
        assert result["estimated_avg_bpw"] == 0.0

    def test_high_low_count(self):
        profile = _make_profile(5)
        result = recommend_mixed_quant(profile, target_bpw=5.0)
        assert "n_high_precision_layers" in result
        assert "n_low_precision_layers" in result
        total = result["n_high_precision_layers"] + result["n_low_precision_layers"]
        assert total == len(profile.layers)
