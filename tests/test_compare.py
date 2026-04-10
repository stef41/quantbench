"""Tests for quantbench.compare."""

from quantbench._types import DType, ModelProfile, QuantFormat, QuantProfile, TensorInfo
from quantbench.compare import compare_formats, compare_profiles


def _make_profile(name: str, dtype: DType = DType.Q4_K_M, n_tensors: int = 3) -> ModelProfile:
    tensors = [
        TensorInfo(name=f"blk.0.w{i}", shape=[100, 100], dtype=dtype)
        for i in range(n_tensors)
    ]
    total = sum(t.n_elements for t in tensors)
    size = sum(t.size_bytes for t in tensors)
    qp = QuantProfile(
        avg_bits_per_weight=dtype.bits_per_weight,
        n_quantized_layers=n_tensors if dtype not in (DType.F16, DType.F32) else 0,
        n_full_precision_layers=n_tensors if dtype in (DType.F16, DType.F32) else 0,
    )
    return ModelProfile(
        name=name, format=QuantFormat.GGUF,
        total_params=total, total_size_bytes=size,
        tensors=tensors, quant=qp,
    )


class TestCompareProfiles:
    def test_same_model(self):
        p = _make_profile("model")
        result = compare_profiles(p, p)
        assert result["size_delta_bytes"] == 0
        assert result["bpw_delta"] == 0.0
        assert result["n_dtype_changes"] == 0

    def test_different_quant(self):
        pa = _make_profile("q4", DType.Q4_K_M)
        pb = _make_profile("q8", DType.Q8_0)
        result = compare_profiles(pa, pb)
        assert result["bpw_delta"] != 0.0
        assert result["size_delta_bytes"] != 0

    def test_dtype_changes_detected(self):
        pa = _make_profile("a", DType.Q4_K_M, n_tensors=2)
        pb = _make_profile("b", DType.Q8_0, n_tensors=2)
        result = compare_profiles(pa, pb)
        assert result["n_dtype_changes"] == 2

    def test_model_names(self):
        pa = _make_profile("alpha")
        pb = _make_profile("beta")
        result = compare_profiles(pa, pb)
        assert result["model_a"] == "alpha"
        assert result["model_b"] == "beta"

    def test_missing_tensors(self):
        pa = _make_profile("a", n_tensors=3)
        pb = _make_profile("b", n_tensors=2)
        # tensor names differ by count, so some are only in A
        result = compare_profiles(pa, pb)
        assert isinstance(result["tensors_only_in_a"], list)


class TestCompareFormats:
    def test_basic(self):
        profiles = [
            _make_profile("q4", DType.Q4_K_M),
            _make_profile("q8", DType.Q8_0),
            _make_profile("fp16", DType.F16),
        ]
        result = compare_formats(profiles)
        assert result["n_models"] == 3
        assert len(result["models"]) == 3
        assert len(result["ranking"]) == 3

    def test_ranking_has_rank(self):
        profiles = [
            _make_profile("q4", DType.Q4_K_M),
            _make_profile("q8", DType.Q8_0),
        ]
        result = compare_formats(profiles)
        for row in result["ranking"]:
            assert "rank" in row

    def test_empty(self):
        result = compare_formats([])
        assert result["models"] == []
        assert result["ranking"] == []

    def test_single(self):
        result = compare_formats([_make_profile("only")])
        assert result["n_models"] == 1
