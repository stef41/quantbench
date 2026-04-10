"""quantbench — Quantization quality analyzer."""
from __future__ import annotations

__version__ = "0.3.0"

from quantbench._types import (
    DType,
    LayerInfo,
    ModelProfile,
    QualityEstimate,
    QuantbenchError,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    TensorInfo,
)
from quantbench.bandwidth import (
    KNOWN_GPUS,
    BandwidthEstimate,
    BandwidthEstimator,
    GPUSpec,
    compare_gpus,
    format_bandwidth_report,
)
from quantbench.compare import (
    compare_formats,
    compare_profiles,
)
from quantbench.imatrix import (
    ImatrixAnalysis,
    ImatrixData,
    ImatrixEntry,
    analyze_imatrix,
    format_imatrix_report,
    parse_imatrix,
)
from quantbench.layerwise import (
    analyze_layers,
    layer_sensitivity,
    recommend_mixed_quant,
)
from quantbench.matrix import (
    KNOWN_FORMATS,
    ComparisonMatrix,
    FormatComparison,
    QuantFormatSpec,
    format_comparison_table,
)
from quantbench.perplexity import (
    PerplexityDelta,
    PerplexityScore,
    estimate_perplexity_delta,
    format_quality_report,
    perplexity_from_logprobs,
)
from quantbench.perplexity import (
    quality_score as perplexity_quality_score,
)
from quantbench.predict import (
    estimate_quality,
    perplexity_delta,
)
from quantbench.profile import profile_from_dict, profile_gguf, profile_safetensors
from quantbench.recommend import (
    Recommendation,
    format_recommendation,
    recommend,
)
from quantbench.report import (
    format_markdown,
    format_report_rich,
    format_report_text,
    load_json,
    report_to_dict,
    save_json,
)

__all__ = [
    "DType",
    "ImatrixAnalysis",
    "ImatrixData",
    "ImatrixEntry",
    "LayerInfo",
    "ModelProfile",
    "QuantFormat",
    "QuantMethod",
    "QuantProfile",
    "QuantbenchError",
    "QualityEstimate",
    "PerplexityDelta",
    "PerplexityScore",
    "Recommendation",
    "TensorInfo",
    "analyze_imatrix",
    "analyze_layers",
    "compare_formats",
    "compare_profiles",
    "estimate_perplexity_delta",
    "estimate_quality",
    "format_imatrix_report",
    "format_markdown",
    "format_quality_report",
    "format_recommendation",
    "format_report_rich",
    "format_report_text",
    "layer_sensitivity",
    "load_json",
    "parse_imatrix",
    "perplexity_delta",
    "perplexity_from_logprobs",
    "perplexity_quality_score",
    "profile_from_dict",
    "profile_gguf",
    "profile_safetensors",
    "recommend",
    "recommend_mixed_quant",
    "report_to_dict",
    "save_json",
    # matrix
    "ComparisonMatrix",
    "FormatComparison",
    "KNOWN_FORMATS",
    "QuantFormatSpec",
    "format_comparison_table",
    # bandwidth
    "BandwidthEstimate",
    "BandwidthEstimator",
    "GPUSpec",
    "KNOWN_GPUS",
    "compare_gpus",
    "format_bandwidth_report",
]
