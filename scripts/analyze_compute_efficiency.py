#!/usr/bin/env python3
"""
Analyze compute efficiency of GEMV kernels.

Key metrics:
1. Achieved vs theoretical throughput
2. Compute vs memory time breakdown
3. Instruction mix analysis
"""

import torch
import time

print("=" * 70)
print("Compute Efficiency Analysis")
print("=" * 70)

# gpt-oss-120b dimensions
K = 2880
M = 1

from flashinfer.gemv import (
    quantize_activations_q8,
    gemv_mxfp4_dp4a_prequant,
)

# Hardware specs for GB10 (SM121)
# These are estimates - actual specs may vary
HBM_BW = 273  # GB/s
SM_COUNT = 22  # Approximate for GB10
CUDA_CORES_PER_SM = 128  # Blackwell
DP4A_PER_CYCLE = 1  # DP4A throughput per SM (conservative)
CLOCK_GHZ = 2.0  # Approximate boost clock

# Test cases
test_cases = [
    ("K/V projection", 360),
    ("Q/O projection", 2880),
    ("LM Head", 201088),
]

# Pre-allocate
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
q8_buffer = torch.empty((M, K // 32, 36), dtype=torch.uint8, device="cuda")
quantize_activations_q8(input_bf16, q8_buffer)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("\n1. Per-layer Analysis:")
print("-" * 70)
print(f"{'Layer':<20} {'N':>8} {'Time':>10} {'Mem BW':>10} {'DP4A ops':>12} {'Compute':>10}")
print(f"{'':20} {'':>8} {'(μs)':>10} {'(GB/s)':>10} {'(Gops)':>12} {'Bound?':>10}")
print("-" * 70)

for name, N in test_cases:
    weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127
    
    # Warmup
    for _ in range(10):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    
    # Measure
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    end.record()
    torch.cuda.synchronize()
    
    time_us = start.elapsed_time(end) * 1000 / 100
    time_s = time_us / 1e6
    
    # Memory analysis
    weight_bytes = N * K // 2  # FP4 packed
    scale_bytes = N * K // 32  # E8M0
    act_bytes = M * K // 32 * 36  # Q8_1 blocks
    output_bytes = M * N * 2  # BF16
    total_bytes = weight_bytes + scale_bytes + act_bytes + output_bytes
    
    achieved_bw = total_bytes / time_s / 1e9
    bw_efficiency = achieved_bw / HBM_BW * 100
    
    # Compute analysis
    # For each output element: K/32 blocks, each block = 32 FP4×INT8 multiplies
    # DP4A does 4 multiplies per instruction, so K/32 * 32 / 4 = K/4 DP4A per output
    # Total DP4A ops = M * N * K / 4
    dp4a_ops = M * N * K / 4
    dp4a_gops = dp4a_ops / 1e9
    dp4a_per_sec = dp4a_ops / time_s / 1e12  # TOPS
    
    # Theoretical DP4A throughput (very rough estimate)
    # Each SM can do some DP4A per cycle
    theoretical_dp4a_tops = SM_COUNT * DP4A_PER_CYCLE * CLOCK_GHZ * 4  # 4 ops per DP4A
    compute_efficiency = dp4a_per_sec / theoretical_dp4a_tops * 100 if theoretical_dp4a_tops > 0 else 0
    
    # Determine bottleneck
    if bw_efficiency > 70:
        bottleneck = "Memory"
    elif compute_efficiency > 50:
        bottleneck = "Compute"
    else:
        bottleneck = "Latency"
    
    print(f"{name:<20} {N:>8} {time_us:>10.2f} {achieved_bw:>10.1f} {dp4a_gops:>12.3f} {bottleneck:>10}")

print("\n2. Instruction Analysis:")
print("-" * 70)

# Analyze instruction mix for Q/O projection (most balanced)
N = 2880
weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127

# Count operations per output element
n_k_blocks = K // 32

print(f"Per output element (K={K}, blocks={n_k_blocks}):")
print(f"  Memory loads:")
print(f"    Weight: {K//2} bytes (16 bytes per K-block)")
print(f"    Scale: {n_k_blocks} bytes (1 per K-block)")
print(f"    Activations: {n_k_blocks * 36} bytes (36 per Q8_1 block, cached)")
print(f"  Compute:")
print(f"    LUT lookups: {n_k_blocks * 2} (get_int_from_table_16 per 16 bytes)")
print(f"    DP4A: {n_k_blocks * 8} ops (8 per K-block)")
print(f"    FP multiply-add: {n_k_blocks * 2} (scale application)")
print(f"    Warp reduction: 32 adds")

# Ratio analysis
total_dp4a = n_k_blocks * 8
total_lut = n_k_blocks * 2
total_fma = n_k_blocks * 2
total_reduction = 32

print(f"\n  Instruction ratio:")
print(f"    DP4A : LUT : FMA : Reduction = {total_dp4a} : {total_lut} : {total_fma} : {total_reduction}")
print(f"    DP4A represents {total_dp4a / (total_dp4a + total_lut + total_fma + total_reduction) * 100:.1f}% of compute")

print("\n3. Potential Improvements:")
print("-" * 70)

improvements = [
    ("LUT optimization", "get_int_from_table_16 uses 16 LDS.128 + shuffles. Could use PRMT?"),
    ("Register blocking", "Process multiple K-blocks per thread to hide latency"),
    ("Shared memory for weights", "Cache weight tiles in smem for reuse (only helps M>1)"),
    ("Unroll K loop", "Explicit unrolling to increase ILP"),
    ("Vectorized output", "Write multiple outputs at once (for large N)"),
    ("Fused scale application", "Combine weight_scale * act_scale earlier"),
]

for name, desc in improvements:
    print(f"  • {name}: {desc}")

print("\n4. Roofline Analysis:")
print("-" * 70)

# For each layer, compute arithmetic intensity
for name, N in test_cases:
    weight_bytes = N * K // 2
    scale_bytes = N * K // 32
    act_bytes = M * K // 32 * 36  # Loaded once, reused
    output_bytes = M * N * 2
    
    # Bytes transferred (activations amortized if reused)
    bytes_per_output = (weight_bytes + scale_bytes) / N + output_bytes / N
    
    # Ops per output: K/32 blocks * 8 DP4A * 4 ops = K ops
    ops_per_output = K
    
    # Arithmetic intensity
    ai = ops_per_output / bytes_per_output
    
    # Roofline intersection
    # Memory roof: BW * AI
    # Compute roof: peak TOPS
    mem_limited_tops = HBM_BW * ai / 1e3  # Convert to TOPS
    
    print(f"{name:<20}: AI = {ai:.2f} ops/byte, Memory-limited peak = {mem_limited_tops:.2f} TOPS")

print("\n  Note: Low arithmetic intensity (<10 ops/byte) means memory-bound.")
print("  Focus on memory efficiency, not compute optimization.")
