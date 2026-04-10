"""Get quantization recommendations for a model profile.

Demonstrates: recommend(), format_recommendation().
"""

from quantbench import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    TensorInfo,
    recommend,
    format_recommendation,
)

if __name__ == "__main__":
    # Build a realistic FP16 model profile to get recommendations for
    tensors = [
        TensorInfo(name="blk.0.attn_q.weight", shape=[4096, 4096], dtype=DType.F16),
        TensorInfo(name="blk.0.attn_k.weight", shape=[4096, 4096], dtype=DType.F16),
        TensorInfo(name="blk.0.ffn_down.weight", shape=[4096, 11008], dtype=DType.F16),
        TensorInfo(name="embed.weight", shape=[32000, 4096], dtype=DType.F16),
        TensorInfo(name="lm_head.weight", shape=[32000, 4096], dtype=DType.F16),
    ]
    layers = [
        LayerInfo(name="blk.0", tensors=tensors[:3]),
        LayerInfo(name="embed", tensors=[tensors[3]]),
        LayerInfo(name="lm_head", tensors=[tensors[4]]),
    ]

    profile = ModelProfile(
        name="llama-7b-fp16",
        format=QuantFormat.SAFETENSORS,
        total_params=sum(t.n_elements for t in tensors),
        total_size_bytes=sum(t.size_bytes for t in tensors),
        tensors=tensors,
        layers=layers,
        quant=QuantProfile(method=QuantMethod.FP16, avg_bits_per_weight=16.0),
    )

    # Recommend with max 5% quality loss (default)
    rec = recommend(profile, max_quality_loss=0.05)
    print("=== Default recommendation (≤5% quality loss) ===")
    print(format_recommendation(rec))

    # Recommend targeting ~4 bits per weight
    rec_4bit = recommend(profile, target_bits=4.0)
    print("\n=== Targeting ~4 bits per weight ===")
    print(format_recommendation(rec_4bit))

    # Recommend aggressive compression
    rec_aggressive = recommend(profile, max_quality_loss=0.25)
    print("\n=== Aggressive (≤25% quality loss) ===")
    print(format_recommendation(rec_aggressive))
