#!/usr/bin/env python3
"""
Find GEMV vs GEMM crossover using BF16 matmul as GEMM baseline.
This represents the best possible GEMM performance (before FP4 overhead).
"""

import torch

print("=" * 70)
print("GEMV vs BF16 GEMM Crossover Analysis")
print("=" * 70)

K = 2880
N = 2880

from flashinfer.gemv import quantize_activations_q8, gemv_mxfp4_dp4a_prequant

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Prepare weights
weight_gemv = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
scale_gemv = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127
weight_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")  # For matmul

print(f"Dimensions: K={K}, N={N}")
print(f"\nComparing MXFP4 GEMV vs BF16 matmul (best-case GEMM)")
print(f"Note: Real FP4 GEMM would be between these two.\n")

print(f"{'M':>4} {'GEMV (μs)':>12} {'BF16 mm (μs)':>12} {'Faster':>10} {'Ratio':>10}")
print("-" * 52)

crossover_m = None
results = []

for M in [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 128, 256]:
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    # GEMV benchmark
    n_k_blocks = K // 32
    q8_buffer = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
    quantize_activations_q8(input_bf16, q8_buffer)
    
    for _ in range(10):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight_gemv, scale_gemv, K)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight_gemv, scale_gemv, K)
    end.record()
    torch.cuda.synchronize()
    gemv_us = start.elapsed_time(end) * 1000 / 50
    
    # BF16 matmul benchmark (best-case GEMM)
    for _ in range(10):
        _ = torch.mm(input_bf16, weight_bf16)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = torch.mm(input_bf16, weight_bf16)
    end.record()
    torch.cuda.synchronize()
    bf16_us = start.elapsed_time(end) * 1000 / 50
    
    # Compare
    if bf16_us < gemv_us:
        faster = "BF16 mm"
        ratio = gemv_us / bf16_us
        if crossover_m is None:
            crossover_m = M
    else:
        faster = "GEMV"
        ratio = bf16_us / gemv_us
    
    results.append((M, gemv_us, bf16_us, faster))
    print(f"{M:>4} {gemv_us:>12.2f} {bf16_us:>12.2f} {faster:>10} {ratio:>10.2f}x")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if crossover_m:
    print(f"\nBF16 matmul beats GEMV at M = {crossover_m}")
    print(f"\nSince real FP4 GEMM is slower than BF16 matmul:")
    print(f"  - Actual crossover is likely at M = {crossover_m * 2} to {crossover_m * 4}")
    print(f"  - Conservative recommendation: M >= {max(8, crossover_m * 2)}")
else:
    print("\nGEMV wins for all tested M values!")
    print("This means GEMV is very well optimized.")

# Calculate theoretical crossover
print("\n" + "-" * 50)
print("Theoretical Analysis:")
print("-" * 50)

# GEMV: ~12.5 μs for M=1, scales linearly
gemv_per_m = results[0][1]  # μs per M

# BF16 matmul: has fixed overhead + compute
# From results, estimate the overhead
if len(results) > 1:
    bf16_m1 = results[0][2]
    bf16_m64 = next((r[2] for r in results if r[0] == 64), None)
    if bf16_m64:
        # Overhead ≈ time at M=1
        # Per-M cost ≈ (time at M=64 - time at M=1) / 63
        bf16_overhead = bf16_m1
        bf16_per_m = (bf16_m64 - bf16_m1) / 63
        
        print(f"GEMV: {gemv_per_m:.2f} μs × M (linear)")
        print(f"BF16: {bf16_overhead:.2f} μs overhead + {bf16_per_m:.2f} μs × M")
        
        # Crossover: gemv_per_m * M = bf16_overhead + bf16_per_m * M
        # M * (gemv_per_m - bf16_per_m) = bf16_overhead
        if gemv_per_m > bf16_per_m:
            theoretical_crossover = bf16_overhead / (gemv_per_m - bf16_per_m)
            print(f"\nTheoretical crossover: M ≈ {theoretical_crossover:.1f}")
