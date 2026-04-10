"""Analyze quantization quality using perplexity heuristics.

Demonstrates: perplexity_from_logprobs(), quality_score(),
estimate_perplexity_delta(), format_quality_report().
"""

import math

from quantbench import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    TensorInfo,
    estimate_perplexity_delta,
    format_quality_report,
    perplexity_from_logprobs,
    perplexity_quality_score as quality_score,
)

if __name__ == "__main__":
    # --- 1. Compute perplexity from log-probabilities ---
    # Simulate token log-probs from a language model evaluation
    logprobs = [-2.3, -1.8, -3.1, -0.5, -1.2, -2.7, -1.9, -0.8, -2.1, -1.5]

    ppl = perplexity_from_logprobs(logprobs)
    print("=== Perplexity from Log-Probabilities ===")
    print(f"  Perplexity   : {ppl.value:.2f}")
    print(f"  Tokens       : {ppl.num_tokens}")
    print(f"  Log-likelihood: {ppl.log_likelihood:.4f}")
    print()

    # --- 2. Build original and quantized profiles for comparison ---
    base_tensors = [
        TensorInfo(name="blk.0.attn_q.weight", shape=[4096, 4096], dtype=DType.F16),
        TensorInfo(name="blk.0.ffn_down.weight", shape=[4096, 11008], dtype=DType.F16),
        TensorInfo(name="lm_head.weight", shape=[32000, 4096], dtype=DType.F16),
    ]
    original = ModelProfile(
        name="llama-7b-fp16",
        format=QuantFormat.SAFETENSORS,
        total_params=sum(t.n_elements for t in base_tensors),
        total_size_bytes=sum(t.size_bytes for t in base_tensors),
        tensors=base_tensors,
        layers=[LayerInfo(name="blk.0", tensors=base_tensors[:2]),
                LayerInfo(name="lm_head", tensors=[base_tensors[2]])],
        quant=QuantProfile(method=QuantMethod.FP16, avg_bits_per_weight=16.0),
    )

    q4_tensors = [
        TensorInfo(name="blk.0.attn_q.weight", shape=[4096, 4096], dtype=DType.Q4_K_M),
        TensorInfo(name="blk.0.ffn_down.weight", shape=[4096, 11008], dtype=DType.Q4_K_M),
        TensorInfo(name="lm_head.weight", shape=[32000, 4096], dtype=DType.Q6_K),
    ]
    quantized = ModelProfile(
        name="llama-7b-q4_k_m",
        format=QuantFormat.GGUF,
        total_params=sum(t.n_elements for t in q4_tensors),
        total_size_bytes=sum(t.size_bytes for t in q4_tensors),
        tensors=q4_tensors,
        layers=[LayerInfo(name="blk.0", tensors=q4_tensors[:2]),
                LayerInfo(name="lm_head", tensors=[q4_tensors[2]])],
        quant=QuantProfile(method=QuantMethod.GGML, avg_bits_per_weight=4.85),
    )

    # --- 3. Estimate perplexity delta ---
    delta = estimate_perplexity_delta(original, quantized)
    print(format_quality_report(delta))

    # --- 4. Quick quality score ---
    score = quality_score(quantized)
    print(f"\nQuality score (vs FP16): {score:.4f}")
    print(f"Interpretation: {'Excellent' if score > 0.95 else 'Good' if score > 0.85 else 'Acceptable' if score > 0.7 else 'Degraded'}")
