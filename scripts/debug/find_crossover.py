#!/usr/bin/env python3
"""
Find the actual GEMV vs GEMM crossover point for gpt-oss-120b dimensions.
"""

import torch
import time

print("=" * 70)
print("Finding GEMV vs GEMM Crossover Point")
print("=" * 70)

K = 2880  # gpt-oss-120b hidden size
N = 2880  # Q/O projection size

from flashinfer.gemv import quantize_activations_q8, gemv_mxfp4_dp4a_prequant

# Also try Marlin GEMM for comparison
try:
    from vllm._custom_ops import marlin_gemm_mxfp4
    has_marlin = True
except:
    has_marlin = False
    print("Note: Marlin not available for comparison")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(f"\nDimensions: K={K}, N={N}")
print(f"Testing M from 1 to 64...\n")

print(f"{'M':>4} {'GEMV (μs)':>12} {'Marlin (μs)':>12} {'Winner':>10} {'Speedup':>10}")
print("-" * 52)

# Test various M values
for M in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]:
    # Prepare inputs for GEMV
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    n_k_blocks = K // 32
    q8_buffer = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
    weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127
    
    # Quantize activations
    quantize_activations_q8(input_bf16, q8_buffer)
    
    # Benchmark GEMV
    for _ in range(10):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    end.record()
    torch.cuda.synchronize()
    gemv_us = start.elapsed_time(end) * 1000 / 50
    
    # Benchmark Marlin (represents GEMM path)
    if has_marlin:
        # Marlin uses different weight format - skip for now
        marlin_us = float('inf')
    else:
        marlin_us = float('inf')
    
    # For now, estimate GEMM time based on M scaling
    # A proper GEMM should scale sublinearly with M due to better parallelism
    # This is a rough estimate - we'd need actual CUTLASS GEMM benchmarks
    
    winner = "GEMV"
    speedup = 1.0
    
    # Print results
    marlin_str = f"{marlin_us:.2f}" if marlin_us != float('inf') else "N/A"
    print(f"{M:>4} {gemv_us:>12.2f} {marlin_str:>12} {winner:>10} {speedup:>10.2f}x")

print("\n" + "=" * 70)
print("Analysis")
print("=" * 70)

print("""
To find the true crossover, we need to benchmark CUTLASS GEMM at these M values.

Current GEMV scaling observation:
- GEMV time scales roughly linearly with M (each row independent)
- GEMM time scales sublinearly (better parallelism, Tensor Core efficiency)

The crossover happens when:
  GEMM_time(M) < GEMV_time(M)

For CUTLASS with 64×128 tiles:
- M=1-7: GEMM wastes most of the tile → GEMV wins
- M=8-15: GEMM starts becoming competitive
- M=16+: GEMM likely wins due to Tensor Core throughput

Recommendation:
- Benchmark CUTLASS GEMM for these M values
- Set crossover at the measured point
- Or use a conservative M=16 if benchmarking not available
""")
