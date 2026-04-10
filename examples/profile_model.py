"""Profile a model from a dictionary and display the report.

Demonstrates: profile_from_dict(), format_report_text().
No GGUF file needed — builds a profile from metadata.
"""

from quantbench import (
    DType,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    TensorInfo,
    LayerInfo,
    profile_from_dict,
    format_report_text,
)

if __name__ == "__main__":
    # Build a model profile from a dictionary (e.g. loaded from JSON)
    profile_data = {
        "name": "llama-7b-q4_k_m",
        "format": "gguf",
        "total_params": 6_738_415_616,
        "total_size_bytes": 4_081_004_000,
        "quant": {
            "method": "ggml",
            "avg_bits_per_weight": 4.85,
            "n_quantized_layers": 224,
            "n_full_precision_layers": 7,
            "dtype_distribution": {"q4_k_m": 0.85, "q6_k": 0.10, "f16": 0.05},
        },
        "tensors": [],
        "layers": [],
        "metadata": {"general.name": "llama-7b-q4_k_m", "general.architecture": "llama"},
    }

    profile = profile_from_dict(profile_data)
    print(format_report_text(profile))

    # You can also build a profile directly in code
    tensors = [
        TensorInfo(name="blk.0.attn_q.weight", shape=[4096, 4096], dtype=DType.Q4_K_M),
        TensorInfo(name="blk.0.attn_k.weight", shape=[4096, 4096], dtype=DType.Q4_K_M),
        TensorInfo(name="blk.0.ffn_down.weight", shape=[4096, 11008], dtype=DType.Q4_K_M),
        TensorInfo(name="output.weight", shape=[4096, 32000], dtype=DType.Q6_K),
    ]
    layers = [LayerInfo(name="blk.0", tensors=tensors[:3]), LayerInfo(name="output", tensors=[tensors[3]])]

    manual_profile = ModelProfile(
        name="custom-model",
        format=QuantFormat.GGUF,
        total_params=sum(t.n_elements for t in tensors),
        total_size_bytes=sum(t.size_bytes for t in tensors),
        tensors=tensors,
        layers=layers,
        quant=QuantProfile(
            method=QuantMethod.GGML,
            avg_bits_per_weight=4.85,
            n_quantized_layers=3,
            n_full_precision_layers=1,
        ),
    )

    print("\n" + format_report_text(manual_profile))
