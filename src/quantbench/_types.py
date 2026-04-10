"""Core types for quantbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class QuantbenchError(Exception):
    """Base exception for quantbench."""


class QuantFormat(Enum):
    """Quantization file format."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    UNKNOWN = "unknown"


class QuantMethod(Enum):
    """Quantization method/algorithm."""
    GPTQ = "gptq"
    AWQ = "awq"
    GGML = "ggml"
    BITSANDBYTES = "bitsandbytes"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"
    UNKNOWN = "unknown"


class DType(Enum):
    """Data type for tensors."""
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q5_1 = "q5_1"
    Q5_0 = "q5_0"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q4_1 = "q4_1"
    Q4_0 = "q4_0"
    Q3_K_M = "q3_k_m"
    Q3_K_S = "q3_k_s"
    Q3_K_L = "q3_k_l"
    Q2_K = "q2_k"
    IQ4_XS = "iq4_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"
    IQ1_S = "iq1_s"
    UNKNOWN = "unknown"

    @property
    def bits_per_weight(self) -> float:
        """Approximate bits per weight for this dtype."""
        _bpw = {
            "f32": 32.0, "f16": 16.0, "bf16": 16.0,
            "q8_0": 8.5, "q6_k": 6.5625, "q5_k_m": 5.5,
            "q5_k_s": 5.5, "q5_1": 5.5, "q5_0": 5.5,
            "q4_k_m": 4.85, "q4_k_s": 4.5, "q4_1": 4.5,
            "q4_0": 4.5, "q3_k_m": 3.9, "q3_k_s": 3.5,
            "q3_k_l": 4.3, "q2_k": 3.35,
            "iq4_xs": 4.25, "iq3_xxs": 3.06,
            "iq2_xxs": 2.06, "iq1_s": 1.56,
            "unknown": 16.0,
        }
        return _bpw.get(self.value, 16.0)


@dataclass
class TensorInfo:
    """Information about a single tensor in a model."""
    name: str
    shape: List[int]
    dtype: DType
    n_elements: int = 0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.n_elements == 0 and self.shape:
            prod = 1
            for s in self.shape:
                prod *= s
            self.n_elements = prod
        if self.size_bytes == 0 and self.n_elements > 0:
            self.size_bytes = int(self.n_elements * self.dtype.bits_per_weight / 8)

    @property
    def bits_per_weight(self) -> float:
        return self.dtype.bits_per_weight

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP32."""
        return 32.0 / self.bits_per_weight if self.bits_per_weight > 0 else 1.0


@dataclass
class LayerInfo:
    """Information about a model layer (group of tensors)."""
    name: str
    tensors: List[TensorInfo] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def size_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    @property
    def avg_bits_per_weight(self) -> float:
        total_elements = sum(t.n_elements for t in self.tensors)
        if total_elements == 0:
            return 0.0
        weighted = sum(t.n_elements * t.bits_per_weight for t in self.tensors)
        return weighted / total_elements

    @property
    def dominant_dtype(self) -> DType:
        if not self.tensors:
            return DType.UNKNOWN
        dtype_counts: Dict[DType, int] = {}
        for t in self.tensors:
            dtype_counts[t.dtype] = dtype_counts.get(t.dtype, 0) + t.n_elements
        return max(dtype_counts, key=lambda d: dtype_counts[d])


@dataclass
class QuantProfile:
    """Quantization profile summary."""
    method: QuantMethod = QuantMethod.UNKNOWN
    avg_bits_per_weight: float = 0.0
    dtype_distribution: Dict[str, float] = field(default_factory=dict)
    n_quantized_layers: int = 0
    n_full_precision_layers: int = 0
    group_size: int = 0


@dataclass
class ModelProfile:
    """Full model profile with all tensor/layer information."""
    name: str
    format: QuantFormat
    total_params: int = 0
    total_size_bytes: int = 0
    tensors: List[TensorInfo] = field(default_factory=list)
    layers: List[LayerInfo] = field(default_factory=list)
    quant: QuantProfile = field(default_factory=QuantProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.total_size_bytes / (1024**3) if self.total_size_bytes > 0 else 0.0

    @property
    def compression_ratio(self) -> float:
        fp32_size = self.total_params * 4
        return fp32_size / self.total_size_bytes if self.total_size_bytes > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format.value,
            "total_params": self.total_params,
            "total_size_bytes": self.total_size_bytes,
            "size_gb": round(self.size_gb, 3),
            "compression_ratio": round(self.compression_ratio, 2),
            "quant": {
                "method": self.quant.method.value,
                "avg_bits_per_weight": round(self.quant.avg_bits_per_weight, 2),
                "dtype_distribution": self.quant.dtype_distribution,
                "n_quantized_layers": self.quant.n_quantized_layers,
                "n_full_precision_layers": self.quant.n_full_precision_layers,
            },
            "n_tensors": len(self.tensors),
            "n_layers": len(self.layers),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModelProfile:
        quant_d = d.get("quant", {})
        qp = QuantProfile(
            method=QuantMethod(quant_d.get("method", "unknown")),
            avg_bits_per_weight=quant_d.get("avg_bits_per_weight", 0.0),
            dtype_distribution=quant_d.get("dtype_distribution", {}),
            n_quantized_layers=quant_d.get("n_quantized_layers", 0),
            n_full_precision_layers=quant_d.get("n_full_precision_layers", 0),
        )
        return cls(
            name=d["name"],
            format=QuantFormat(d.get("format", "unknown")),
            total_params=d.get("total_params", 0),
            total_size_bytes=d.get("total_size_bytes", 0),
            quant=qp,
            metadata=d.get("metadata", {}),
        )


@dataclass
class QualityEstimate:
    """Estimated quality loss from quantization."""
    model_name: str
    method: str
    avg_bits_per_weight: float
    estimated_perplexity_delta: float
    quality_score: float  # 0-1, higher = better
    risk_level: str  # low, medium, high, critical
    sensitive_layers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
