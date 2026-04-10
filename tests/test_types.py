"""Tests for quantbench._types."""

import pytest
from quantbench._types import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    QuantbenchError,
    QualityEstimate,
    TensorInfo,
)


class TestDType:
    def test_bits_per_weight_f32(self):
        assert DType.F32.bits_per_weight == 32.0

    def test_bits_per_weight_f16(self):
        assert DType.F16.bits_per_weight == 16.0

    def test_bits_per_weight_q4_0(self):
        assert DType.Q4_0.bits_per_weight == 4.5

    def test_bits_per_weight_q8_0(self):
        assert DType.Q8_0.bits_per_weight == 8.5

    def test_bits_per_weight_q2_k(self):
        assert DType.Q2_K.bits_per_weight == 3.35

    def test_bits_per_weight_unknown(self):
        assert DType.UNKNOWN.bits_per_weight == 16.0


class TestTensorInfo:
    def test_auto_elements(self):
        t = TensorInfo(name="w", shape=[4, 8], dtype=DType.F16)
        assert t.n_elements == 32

    def test_auto_size_bytes(self):
        t = TensorInfo(name="w", shape=[100], dtype=DType.F32)
        assert t.n_elements == 100
        assert t.size_bytes == 400

    def test_compression_ratio(self):
        t = TensorInfo(name="w", shape=[100], dtype=DType.Q4_0)
        assert t.compression_ratio > 1.0

    def test_explicit_elements(self):
        t = TensorInfo(name="w", shape=[2, 3], dtype=DType.F16, n_elements=10)
        assert t.n_elements == 10

    def test_empty_shape(self):
        t = TensorInfo(name="w", shape=[], dtype=DType.F16)
        assert t.n_elements == 0


class TestLayerInfo:
    def test_n_params(self):
        tensors = [
            TensorInfo(name="a", shape=[10], dtype=DType.F16),
            TensorInfo(name="b", shape=[20], dtype=DType.F16),
        ]
        layer = LayerInfo(name="layer0", tensors=tensors)
        assert layer.n_params == 30

    def test_size_bytes(self):
        tensors = [TensorInfo(name="a", shape=[100], dtype=DType.F32)]
        layer = LayerInfo(name="layer0", tensors=tensors)
        assert layer.size_bytes == 400

    def test_avg_bits(self):
        tensors = [
            TensorInfo(name="a", shape=[100], dtype=DType.F16),
            TensorInfo(name="b", shape=[100], dtype=DType.Q4_0),
        ]
        layer = LayerInfo(name="layer0", tensors=tensors)
        assert 4.0 < layer.avg_bits_per_weight < 16.0

    def test_dominant_dtype(self):
        tensors = [
            TensorInfo(name="a", shape=[1000], dtype=DType.Q4_K_M),
            TensorInfo(name="b", shape=[10], dtype=DType.F16),
        ]
        layer = LayerInfo(name="layer0", tensors=tensors)
        assert layer.dominant_dtype == DType.Q4_K_M

    def test_empty(self):
        layer = LayerInfo(name="empty")
        assert layer.n_params == 0
        assert layer.dominant_dtype == DType.UNKNOWN


class TestModelProfile:
    def test_size_gb(self):
        p = ModelProfile(name="test", format=QuantFormat.GGUF, total_size_bytes=1024**3)
        assert p.size_gb == pytest.approx(1.0)

    def test_compression_ratio(self):
        p = ModelProfile(
            name="test", format=QuantFormat.GGUF,
            total_params=1000, total_size_bytes=500,
        )
        assert p.compression_ratio == 8.0  # 4000 / 500

    def test_to_dict(self):
        p = ModelProfile(name="test", format=QuantFormat.GGUF, total_params=1000)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["format"] == "gguf"

    def test_from_dict(self):
        d = {
            "name": "test",
            "format": "gguf",
            "total_params": 1000,
            "total_size_bytes": 500,
            "quant": {"method": "ggml", "avg_bits_per_weight": 4.5},
        }
        p = ModelProfile.from_dict(d)
        assert p.name == "test"
        assert p.quant.method == QuantMethod.GGML

    def test_roundtrip(self):
        p = ModelProfile(name="test", format=QuantFormat.GGUF, total_params=1000)
        d = p.to_dict()
        p2 = ModelProfile.from_dict(d)
        assert p2.name == p.name


class TestQualityEstimate:
    def test_basic(self):
        qe = QualityEstimate(
            model_name="test", method="ggml", avg_bits_per_weight=4.5,
            estimated_perplexity_delta=0.12, quality_score=0.85,
            risk_level="medium",
        )
        assert qe.model_name == "test"
        assert qe.risk_level == "medium"


class TestQuantbenchError:
    def test_is_exception(self):
        assert issubclass(QuantbenchError, Exception)
