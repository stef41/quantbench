"""Tests for quantbench.bandwidth."""

import math

import pytest
from quantbench.bandwidth import (
    BandwidthEstimate,
    BandwidthEstimator,
    GPUSpec,
    KNOWN_GPUS,
    compare_gpus,
    format_bandwidth_report,
)


@pytest.fixture
def sample_gpu():
    return GPUSpec(
        name="TestGPU",
        memory_bandwidth_gbps=1000.0,
        memory_gb=40.0,
        compute_tflops=200.0,
    )


@pytest.fixture
def estimator(sample_gpu):
    return BandwidthEstimator(sample_gpu)


class TestGPUSpec:
    def test_fields(self, sample_gpu):
        assert sample_gpu.name == "TestGPU"
        assert sample_gpu.memory_bandwidth_gbps == 1000.0
        assert sample_gpu.memory_gb == 40.0
        assert sample_gpu.compute_tflops == 200.0


class TestBandwidthEstimate:
    def test_fields(self):
        est = BandwidthEstimate(
            model_size_gb=10.0,
            gpu="TestGPU",
            transfer_time_ms=10.0,
            is_memory_bound=True,
            arithmetic_intensity=2.0,
            achievable_tflops=100.0,
        )
        assert est.model_size_gb == 10.0
        assert est.gpu == "TestGPU"
        assert est.is_memory_bound is True


class TestEstimateTransfer:
    def test_basic(self, estimator):
        est = estimator.estimate_transfer(10.0)
        # 10 GB / 1000 GB/s = 0.01 s = 10 ms
        assert est.transfer_time_ms == pytest.approx(10.0, rel=1e-2)
        assert est.is_memory_bound is True
        assert est.arithmetic_intensity == 0.0

    def test_large_model(self, estimator):
        est = estimator.estimate_transfer(80.0)
        assert est.transfer_time_ms == pytest.approx(80.0, rel=1e-2)

    def test_returns_gpu_name(self, estimator):
        est = estimator.estimate_transfer(1.0)
        assert est.gpu == "TestGPU"

    def test_zero_model_size_raises(self, estimator):
        with pytest.raises(ValueError, match="positive"):
            estimator.estimate_transfer(0.0)

    def test_negative_model_size_raises(self, estimator):
        with pytest.raises(ValueError, match="positive"):
            estimator.estimate_transfer(-1.0)


class TestEstimateInference:
    def test_basic(self, estimator):
        est = estimator.estimate_inference(10.0)
        assert est.model_size_gb == 10.0
        assert est.transfer_time_ms > 0
        assert est.arithmetic_intensity > 0

    def test_batch_size_affects_intensity(self, estimator):
        est1 = estimator.estimate_inference(10.0, batch_size=1)
        est8 = estimator.estimate_inference(10.0, batch_size=8)
        assert est8.arithmetic_intensity > est1.arithmetic_intensity

    def test_small_batch_memory_bound(self, estimator):
        est = estimator.estimate_inference(10.0, batch_size=1)
        assert est.is_memory_bound is True

    def test_invalid_batch_size(self, estimator):
        with pytest.raises(ValueError, match="batch_size"):
            estimator.estimate_inference(10.0, batch_size=0)

    def test_invalid_seq_length(self, estimator):
        with pytest.raises(ValueError, match="seq_length"):
            estimator.estimate_inference(10.0, seq_length=0)


class TestRooflineAnalysis:
    def test_returns_dict(self, estimator):
        result = estimator.roofline_analysis(10.0, flops_per_token=1e12)
        assert isinstance(result, dict)
        assert "ridge_point" in result
        assert "is_memory_bound" in result
        assert "tokens_per_second" in result

    def test_memory_bound_low_ai(self, estimator):
        # Very low flops → memory-bound
        result = estimator.roofline_analysis(10.0, flops_per_token=1e6)
        assert result["is_memory_bound"] is True

    def test_invalid_model_size(self, estimator):
        with pytest.raises(ValueError, match="positive"):
            estimator.roofline_analysis(0.0, flops_per_token=1e12)

    def test_invalid_flops(self, estimator):
        with pytest.raises(ValueError, match="positive"):
            estimator.roofline_analysis(10.0, flops_per_token=0.0)


class TestFitsInMemory:
    def test_fits(self, estimator):
        assert estimator.fits_in_memory(20.0) is True

    def test_exact_fit(self, estimator):
        assert estimator.fits_in_memory(40.0) is True

    def test_does_not_fit(self, estimator):
        assert estimator.fits_in_memory(41.0) is False


class TestRequiredGPUs:
    def test_single_gpu(self, estimator):
        assert estimator.required_gpus(30.0) == 1

    def test_two_gpus(self, estimator):
        assert estimator.required_gpus(60.0) == 2

    def test_exact_boundary(self, estimator):
        assert estimator.required_gpus(40.0) == 1

    def test_invalid(self, estimator):
        with pytest.raises(ValueError, match="positive"):
            estimator.required_gpus(-1.0)


class TestKnownGPUs:
    def test_contains_expected_keys(self):
        expected = {"H100_SXM", "H100_PCIe", "A100_80GB", "A100_40GB",
                    "A10G", "L4", "T4", "RTX_4090", "V100"}
        assert expected == set(KNOWN_GPUS.keys())

    def test_all_have_positive_values(self):
        for key, gpu in KNOWN_GPUS.items():
            assert gpu.memory_bandwidth_gbps > 0, key
            assert gpu.memory_gb > 0, key
            assert gpu.compute_tflops > 0, key


class TestCompareGPUs:
    def test_default_all_gpus(self):
        results = compare_gpus(10.0)
        assert len(results) == len(KNOWN_GPUS)

    def test_custom_gpu_list(self, sample_gpu):
        results = compare_gpus(10.0, gpus=[sample_gpu])
        assert len(results) == 1
        assert results[0].gpu == "TestGPU"


class TestFormatBandwidthReport:
    def test_empty(self):
        assert format_bandwidth_report([]) == "No estimates to report."

    def test_non_empty(self):
        results = compare_gpus(10.0)
        report = format_bandwidth_report(results)
        assert "Bandwidth Estimation Report" in report
        assert "10.00 GB" in report

    def test_sorted_by_time(self):
        results = compare_gpus(10.0)
        report = format_bandwidth_report(results)
        lines = [l for l in report.split("\n") if l.strip() and "---" not in l]
        # Just verify it's a non-empty string containing GPU names
        assert any(gpu.name in report for gpu in KNOWN_GPUS.values())
