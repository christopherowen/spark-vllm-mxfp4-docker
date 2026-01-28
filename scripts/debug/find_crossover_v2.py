#!/usr/bin/env python3
"""
Find actual GEMV vs Marlin GEMM crossover point.
"""

import torch
import time

print("=" * 70)
print("Finding GEMV vs Marlin GEMM Crossover Point")
print("=" * 70)

K = 2880  # gpt-oss-120b hidden size
N = 2880  # Q/O projection size

from flashinfer.gemv import quantize_activations_q8, gemv_mxfp4_dp4a_prequant

# Check if Marlin is available
try:
    from vllm._custom_ops import marlin_gemm_mxfp4
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_mxfp4_weights_pack,
    )
    has_marlin = True
    print("✓ Marlin GEMM available")
except Exception as e:
    has_marlin = False
    print(f"✗ Marlin not available: {e}")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(f"\nDimensions: K={K}, N={N}")

# Prepare Marlin weights once
if has_marlin:
    print("Preparing Marlin-packed weights...")
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Quantize to MXFP4
    from flashinfer import mxfp4_quantize
    weight_fp4, weight_scale = mxfp4_quantize(weight_bf16)
    
    # Pack for Marlin
    marlin_weight, marlin_scale = marlin_mxfp4_weights_pack(
        weight_fp4, weight_scale
    )
    print(f"  Marlin weight shape: {marlin_weight.shape}")
    print(f"  Marlin scale shape: {marlin_scale.shape}")

# Also prepare GEMV weights
weight_gemv = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
scale_gemv = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127

print(f"\nTesting M from 1 to 64...\n")
print(f"{'M':>4} {'GEMV (μs)':>12} {'Marlin (μs)':>12} {'Winner':>10} {'Ratio':>10}")
print("-" * 52)

crossover_m = None

for M in [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]:
    # Prepare inputs
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    # GEMV path
    n_k_blocks = K // 32
    q8_buffer = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
    quantize_activations_q8(input_bf16, q8_buffer)
    
    # Benchmark GEMV
    for _ in range(10):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight_gemv, scale_gemv, K)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight_gemv, scale_gemv, K)
    end.record()
    torch.cuda.synchronize()
    gemv_us = start.elapsed_time(end) * 1000 / 50
    
    # Benchmark Marlin GEMM
    if has_marlin:
        output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        
        for _ in range(10):
            marlin_gemm_mxfp4(
                input_bf16.contiguous(),
                marlin_weight,
                output,
                marlin_scale,
                N,  # size_n
                K,  # size_k
            )
        
        torch.cuda.synchronize()
        start.record()
        for _ in range(50):
            marlin_gemm_mxfp4(
                input_bf16.contiguous(),
                marlin_weight,
                output,
                marlin_scale,
                N,
                K,
            )
        end.record()
        torch.cuda.synchronize()
        marlin_us = start.elapsed_time(end) * 1000 / 50
    else:
        marlin_us = float('inf')
    
    # Determine winner
    if marlin_us < gemv_us:
        winner = "Marlin"
        ratio = gemv_us / marlin_us
        if crossover_m is None:
            crossover_m = M
    else:
        winner = "GEMV"
        ratio = marlin_us / gemv_us if marlin_us != float('inf') else 0
    
    marlin_str = f"{marlin_us:.2f}" if marlin_us != float('inf') else "N/A"
    print(f"{M:>4} {gemv_us:>12.2f} {marlin_str:>12} {winner:>10} {ratio:>10.2f}x")

print("\n" + "=" * 70)
if crossover_m:
    print(f"CROSSOVER POINT: M = {crossover_m}")
    print(f"Use GEMV for M < {crossover_m}, use GEMM for M >= {crossover_m}")
else:
    print("No crossover found - GEMV wins for all tested M values")
print("=" * 70)
