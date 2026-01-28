"""Analyze kernel launch count: vLLM vs llama.cpp for decode."""

print("""
================================================================================
DECODE STEP KERNEL ANALYSIS
================================================================================

For gpt-oss-120b with 60 transformer layers, 8 experts per MoE, top-k=8:

vLLM/FlashInfer Decode (per layer):
-----------------------------------
1. QKV projection (Marlin GEMM)                    : 1 kernel
2. RoPE                                            : 1 kernel
3. Attention (FlashInfer decode)                   : 1 kernel
4. O projection (Marlin GEMM)                      : 1 kernel
5. RMSNorm                                         : 1 kernel
6. Router (dense GEMM + softmax + topk)            : 3+ kernels
7. MoE FC1 (CUTLASS grouped GEMM)                  : 1 kernel
8. SwiGLU activation                               : 1 kernel
9. MoE FC2 (CUTLASS grouped GEMM)                  : 1 kernel
10. Token combination (scatter-add)                : 1 kernel
11. Residual add                                   : 1 kernel
12. RMSNorm                                        : 1 kernel
-----------------------------------
SUBTOTAL per layer:                                ~14 kernels
TOTAL for 60 layers:                               ~840 kernels
Plus: embedding lookup, LM head                    ~10 kernels
-----------------------------------
GRAND TOTAL:                                       ~850 kernels


llama.cpp Decode (per layer):
-----------------------------
1. QKV projection (fused GEMV, pre-quantized)      : 2 kernels (quant + GEMV)
2. RoPE                                            : fused with attention
3. Attention                                       : 1 kernel
4. O projection (GEMV, reuses quantized x)         : 1 kernel (GEMV only!)
5. RMSNorm                                         : 1 kernel
6. Router (fused softmax+topk+routing)             : 1 kernel
7. MoE FC1+SwiGLU (FUSED!)                         : 1 kernel per expert
8. MoE FC2                                         : 1 kernel per expert
9. Token combination                               : fused
10. Residual add                                   : fused
-----------------------------------
With expert fusion:
- 8 experts × 2 kernels (FC1+GLU fused, FC2) = 16 kernels
- OR with batched MoE: 2 kernels total

Optimistic estimate per layer:                     ~6-8 kernels
TOTAL for 60 layers:                               ~360-480 kernels
-----------------------------------
SAVINGS:                                           ~2x fewer kernels


KEY DIFFERENCES:
================

1. ACTIVATION QUANTIZATION
   - vLLM:      None (Marlin dequants weights, uses BF16 activations)
   - llama.cpp: ONCE per GEMV call, reused across all rows
   - Impact:    llama.cpp saves N redundant operations

2. GEMV+GLU FUSION
   - vLLM:      Separate SwiGLU kernel
   - llama.cpp: SwiGLU computed inside GEMV kernel
   - Impact:    Saves 1 kernel + 1 memory round-trip

3. TOP-K+SOFTMAX FUSION
   - vLLM:      Multiple kernels (softmax, topk, etc.)
   - llama.cpp: Single fused kernel
   - Impact:    Saves 2-3 kernels + memory traffic

4. PRE-QUANTIZED ACTIVATIONS
   - vLLM:      Not applicable (Marlin uses BF16)
   - llama.cpp: Quantize once, reuse for Q, K, V, O projections
   - Impact:    4x reuse of quantized activations

5. FRAMEWORK OVERHEAD
   - vLLM:      Python dispatch, tensor metadata, etc.
   - llama.cpp: Direct C++ kernel calls
   - Impact:    ~1-5μs per kernel call

ESTIMATED OVERHEAD PER DECODE STEP:
===================================
vLLM:
  - 850 kernels × 5μs overhead = 4.25ms
  - Actual compute time        = ~15ms (estimated)
  - Total:                      ~19ms
  - Throughput:                 ~52 tok/s (1000ms / 19ms)

llama.cpp:
  - 400 kernels × 2μs overhead = 0.8ms
  - Actual compute time        = ~15ms (similar compute)
  - Total:                      ~16ms
  - Throughput:                 ~62 tok/s (1000ms / 16ms)

Wait, this doesn't fully explain the gap. Let's look at compute too...


COMPUTE DIFFERENCES:
====================

Dense layers (QKV, O, LM head) for gpt-oss-120b:
  - K = 2880, N_QKV = 5120, N_O = 2880, N_LM = 201088
  - For M=1 decode:

  Marlin (dequant + BF16 tensor cores):
    - Memory: Read FP4 weights + BF16 input + write BF16 output
    - Compute: BF16 tensor core (fast, but weight dequant overhead)
    
  llama.cpp DP4A:
    - Memory: Read FP4 weights + Q8 input (smaller!) + write F32 output
    - Compute: DP4A on CUDA cores (slower than tensor cores)
    - BUT: Input is 4x smaller (INT8 vs BF16)!

Memory traffic comparison for QKV projection:
  - Marlin:    (5120 × 2880 / 2) + (1 × 2880 × 2) = 7.4MB + 5.7KB = 7.4MB
  - llama.cpp: (5120 × 2880 / 2) + (1 × 2880 × 1) = 7.4MB + 2.9KB = 7.4MB
  - Input is negligible, weights dominate

So why is llama.cpp faster?

THE REAL ANSWER:
================

1. FEWER KERNEL LAUNCHES (2x fewer)
2. FUSED OPERATIONS (GLU, top-k, etc.)
3. LOWER FRAMEWORK OVERHEAD (C++ vs Python)
4. COMPUTE GRAPH OPTIMIZATION (whole-graph view)

Marlin is faster for a single GEMM, but llama.cpp is faster end-to-end
because it optimizes the ENTIRE inference pipeline, not just one kernel.
""")
