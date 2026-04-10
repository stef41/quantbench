"""Memory bandwidth estimation and roofline analysis for quantized models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GPUSpec:
    """Specification of a GPU's memory and compute capabilities."""

    name: str
    memory_bandwidth_gbps: float  # GB/s
    memory_gb: float  # Total VRAM in GB
    compute_tflops: float  # Peak FP16/BF16 TFLOPS


@dataclass
class BandwidthEstimate:
    """Result of a bandwidth estimation."""

    model_size_gb: float
    gpu: str
    transfer_time_ms: float
    is_memory_bound: bool
    arithmetic_intensity: float  # FLOP/byte
    achievable_tflops: float


# Pre-defined GPU specs (peak values from vendor datasheets)
KNOWN_GPUS: Dict[str, GPUSpec] = {
    "H100_SXM": GPUSpec(
        name="H100 SXM",
        memory_bandwidth_gbps=3350.0,
        memory_gb=80.0,
        compute_tflops=989.5,
    ),
    "H100_PCIe": GPUSpec(
        name="H100 PCIe",
        memory_bandwidth_gbps=2039.0,
        memory_gb=80.0,
        compute_tflops=756.5,
    ),
    "A100_80GB": GPUSpec(
        name="A100 80GB SXM",
        memory_bandwidth_gbps=2039.0,
        memory_gb=80.0,
        compute_tflops=312.0,
    ),
    "A100_40GB": GPUSpec(
        name="A100 40GB",
        memory_bandwidth_gbps=1555.0,
        memory_gb=40.0,
        compute_tflops=312.0,
    ),
    "A10G": GPUSpec(
        name="A10G",
        memory_bandwidth_gbps=600.0,
        memory_gb=24.0,
        compute_tflops=125.0,
    ),
    "L4": GPUSpec(
        name="L4",
        memory_bandwidth_gbps=300.0,
        memory_gb=24.0,
        compute_tflops=121.0,
    ),
    "T4": GPUSpec(
        name="T4",
        memory_bandwidth_gbps=300.0,
        memory_gb=16.0,
        compute_tflops=65.0,
    ),
    "RTX_4090": GPUSpec(
        name="RTX 4090",
        memory_bandwidth_gbps=1008.0,
        memory_gb=24.0,
        compute_tflops=330.0,
    ),
    "V100": GPUSpec(
        name="V100",
        memory_bandwidth_gbps=900.0,
        memory_gb=16.0,
        compute_tflops=125.0,
    ),
}


class BandwidthEstimator:
    """Estimate memory bandwidth requirements for model inference."""

    def __init__(self, gpu: GPUSpec) -> None:
        self.gpu = gpu

    def estimate_transfer(self, model_size_gb: float) -> BandwidthEstimate:
        """Estimate time to transfer model weights from HBM once.

        This is the lower bound for a single forward pass in a memory-bound
        regime — every weight must be read at least once.
        """
        if model_size_gb <= 0:
            raise ValueError("model_size_gb must be positive")

        transfer_time_s = model_size_gb / self.gpu.memory_bandwidth_gbps
        transfer_time_ms = transfer_time_s * 1000.0

        # A pure weight-read has zero compute → always memory-bound
        return BandwidthEstimate(
            model_size_gb=model_size_gb,
            gpu=self.gpu.name,
            transfer_time_ms=round(transfer_time_ms, 4),
            is_memory_bound=True,
            arithmetic_intensity=0.0,
            achievable_tflops=0.0,
        )

    def estimate_inference(
        self,
        model_size_gb: float,
        batch_size: int = 1,
        seq_length: int = 512,
    ) -> BandwidthEstimate:
        """Estimate inference bandwidth characteristics.

        Uses a simplified model:
        - Bytes transferred ≈ model_size (weights read once per token)
        - FLOPs per token ≈ 2 × parameters (one multiply + one add per param)
        - Parameters estimated from model_size assuming 2 bytes/param (fp16 baseline).
        """
        if model_size_gb <= 0:
            raise ValueError("model_size_gb must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if seq_length < 1:
            raise ValueError("seq_length must be >= 1")

        model_size_bytes = model_size_gb * (1024 ** 3)

        # Estimate parameter count (assume 2 bytes / param as fp16 baseline)
        est_params = model_size_bytes / 2.0

        # FLOPs per token ≈ 2 × params  (matmul dominated)
        flops_per_token = 2.0 * est_params
        total_flops = flops_per_token * batch_size

        # Bytes read from memory per token generation step
        bytes_read = model_size_bytes  # weights read once

        # Arithmetic intensity (FLOP / byte)
        arithmetic_intensity = total_flops / bytes_read if bytes_read > 0 else 0.0

        # Roofline: compute-bound when AI > ridge point
        ridge_point = (self.gpu.compute_tflops * 1e12) / (
            self.gpu.memory_bandwidth_gbps * 1e9
        )
        is_memory_bound = arithmetic_intensity < ridge_point

        if is_memory_bound:
            # Achievable throughput limited by bandwidth
            achievable_tflops = (
                arithmetic_intensity * self.gpu.memory_bandwidth_gbps * 1e9
            ) / 1e12
        else:
            achievable_tflops = self.gpu.compute_tflops

        # Time for one decoding step (generating one token)
        if is_memory_bound:
            time_s = bytes_read / (self.gpu.memory_bandwidth_gbps * 1e9)
        else:
            time_s = total_flops / (self.gpu.compute_tflops * 1e12)

        transfer_time_ms = round(time_s * 1000.0, 4)

        return BandwidthEstimate(
            model_size_gb=model_size_gb,
            gpu=self.gpu.name,
            transfer_time_ms=transfer_time_ms,
            is_memory_bound=is_memory_bound,
            arithmetic_intensity=round(arithmetic_intensity, 4),
            achievable_tflops=round(achievable_tflops, 4),
        )

    def roofline_analysis(
        self,
        model_size_gb: float,
        flops_per_token: float,
    ) -> Dict[str, object]:
        """Perform roofline analysis for a given workload.

        Returns a dict with ridge point, operational intensity, and
        whether the workload is compute- or memory-bound.
        """
        if model_size_gb <= 0:
            raise ValueError("model_size_gb must be positive")
        if flops_per_token <= 0:
            raise ValueError("flops_per_token must be positive")

        model_size_bytes = model_size_gb * (1024 ** 3)
        arithmetic_intensity = flops_per_token / model_size_bytes

        ridge_point = (self.gpu.compute_tflops * 1e12) / (
            self.gpu.memory_bandwidth_gbps * 1e9
        )

        is_memory_bound = arithmetic_intensity < ridge_point

        if is_memory_bound:
            achievable_tflops = (
                arithmetic_intensity * self.gpu.memory_bandwidth_gbps * 1e9
            ) / 1e12
        else:
            achievable_tflops = self.gpu.compute_tflops

        # Time per token
        if is_memory_bound:
            time_s = model_size_bytes / (self.gpu.memory_bandwidth_gbps * 1e9)
        else:
            time_s = flops_per_token / (self.gpu.compute_tflops * 1e12)

        tokens_per_second = 1.0 / time_s if time_s > 0 else 0.0

        return {
            "gpu": self.gpu.name,
            "model_size_gb": model_size_gb,
            "ridge_point": round(ridge_point, 4),
            "arithmetic_intensity": round(arithmetic_intensity, 4),
            "is_memory_bound": is_memory_bound,
            "achievable_tflops": round(achievable_tflops, 4),
            "time_per_token_ms": round(time_s * 1000.0, 4),
            "tokens_per_second": round(tokens_per_second, 2),
        }

    def fits_in_memory(self, model_size_gb: float) -> bool:
        """Check if the model fits in a single GPU's memory."""
        return model_size_gb <= self.gpu.memory_gb

    def required_gpus(self, model_size_gb: float) -> int:
        """Minimum number of GPUs required to hold the model weights."""
        if model_size_gb <= 0:
            raise ValueError("model_size_gb must be positive")
        return math.ceil(model_size_gb / self.gpu.memory_gb)


def compare_gpus(
    model_size_gb: float,
    gpus: Optional[List[GPUSpec]] = None,
) -> List[BandwidthEstimate]:
    """Compare bandwidth estimates across multiple GPUs.

    If *gpus* is ``None``, all :data:`KNOWN_GPUS` are used.
    """
    if gpus is None:
        gpus = list(KNOWN_GPUS.values())

    results: List[BandwidthEstimate] = []
    for gpu in gpus:
        estimator = BandwidthEstimator(gpu)
        est = estimator.estimate_inference(model_size_gb)
        results.append(est)
    return results


def format_bandwidth_report(estimates: List[BandwidthEstimate]) -> str:
    """Format a list of bandwidth estimates as a human-readable report."""
    if not estimates:
        return "No estimates to report."

    lines: List[str] = []
    lines.append("Bandwidth Estimation Report")
    lines.append("=" * 60)

    model_size = estimates[0].model_size_gb
    lines.append(f"Model size: {model_size:.2f} GB")
    lines.append("")

    header = f"{'GPU':<20} {'Time/tok(ms)':>13} {'Bound':>8} {'AI':>8} {'TFLOPS':>8}"
    lines.append(header)
    lines.append("-" * 60)

    for est in sorted(estimates, key=lambda e: e.transfer_time_ms):
        bound = "MEM" if est.is_memory_bound else "COMP"
        lines.append(
            f"{est.gpu:<20} {est.transfer_time_ms:>13.4f} {bound:>8} "
            f"{est.arithmetic_intensity:>8.2f} {est.achievable_tflops:>8.2f}"
        )

    lines.append("")
    return "\n".join(lines)
