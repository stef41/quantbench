"""quantbench — Quantization quality analyzer."""

from __future__ import annotations

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
from quantbench.profile import profile_gguf, profile_safetensors, profile_from_dict
from quantbench.layerwise import (
    analyze_layers,
    layer_sensitivity,
    recommend_mixed_quant,
)
from quantbench.compare import (
    compare_profiles,
    compare_formats,
)
from quantbench.predict import (
    estimate_quality,
    perplexity_delta,
)
from quantbench.recommend import (
    Recommendation,
    recommend,
    format_recommendation,
)
from quantbench.report import (
    format_report_text,
    format_report_rich,
    format_markdown,
    report_to_dict,
    save_json,
    load_json,
)

__all__ = [
    "DType",
    "LayerInfo",
    "ModelProfile",
    "QuantFormat",
    "QuantMethod",
    "QuantProfile",
    "QuantbenchError",
    "QualityEstimate",
    "Recommendation",
    "TensorInfo",
    "analyze_layers",
    "compare_formats",
    "compare_profiles",
    "estimate_quality",
    "format_markdown",
    "format_recommendation",
    "format_report_rich",
    "format_report_text",
    "layer_sensitivity",
    "load_json",
    "perplexity_delta",
    "profile_from_dict",
    "profile_gguf",
    "profile_safetensors",
    "recommend",
    "recommend_mixed_quant",
    "report_to_dict",
    "save_json",
]
