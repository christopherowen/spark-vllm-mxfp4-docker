#!/usr/bin/env python3
"""Measure kernel launch overhead precisely."""

import torch
import time

print("=" * 70)
print("Kernel Launch Overhead Analysis")
print("=" * 70)

K = 2880
M = 1

from flashinfer.gemv import (
    quantize_activations_q8,
    gemv_mxfp4_dp4a_prequant,
    gemv_mxfp4_dp4a_fused_qkv,
)

# Pre-allocate everything
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
n_k_blocks = K // 32
q8 = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
quantize_activations_q8(input_bf16, q8)

# Test different N sizes
test_cases = [
    ("Tiny (N=32)", 32),
    ("K/V proj (N=360)", 360),
    ("Q/O proj (N=2880)", 2880),
    ("Medium (N=10000)", 10000),
    ("Large (N=50000)", 50000),
    ("LM Head (N=201088)", 201088),
]

print("\n1. Single GEMV Launch Times:")
print("-" * 70)

for name, N in test_cases:
    weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127
    
    # Warmup
    for _ in range(10):
        result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
    
    # Measure with sync after each call (worst case)
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1e6
    min_time = min(times) * 1e6
    
    # Theoretical compute time (rough estimate)
    bytes_moved = N * K // 2 + N * K // 32 + M * N * 2
    theoretical = bytes_moved / 273e9 * 1e6
    overhead = avg_time - theoretical
    
    print(f"{name:25s}: avg={avg_time:8.2f}μs, min={min_time:8.2f}μs, "
          f"theoretical={theoretical:8.2f}μs, overhead≈{overhead:6.2f}μs")

print("\n2. Batched vs Separate Launches:")
print("-" * 70)

# Compare 3 separate K/V/O launches vs hypothetical batched
N_q, N_k, N_v = 2880, 360, 360

weight_q = torch.randint(0, 128, (N_q, K // 2), dtype=torch.uint8, device="cuda")
weight_k = torch.randint(0, 128, (N_k, K // 2), dtype=torch.uint8, device="cuda")
weight_v = torch.randint(0, 128, (N_v, K // 2), dtype=torch.uint8, device="cuda")
scale_q = torch.ones((N_q, K // 32), dtype=torch.uint8, device="cuda") * 127
scale_k = torch.ones((N_k, K // 32), dtype=torch.uint8, device="cuda") * 127
scale_v = torch.ones((N_v, K // 32), dtype=torch.uint8, device="cuda") * 127

# Warmup
for _ in range(10):
    r1 = gemv_mxfp4_dp4a_prequant(q8, weight_q, scale_q, K)
    r2 = gemv_mxfp4_dp4a_prequant(q8, weight_k, scale_k, K)
    r3 = gemv_mxfp4_dp4a_prequant(q8, weight_v, scale_v, K)

# Measure separate launches
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    r1 = gemv_mxfp4_dp4a_prequant(q8, weight_q, scale_q, K)
    r2 = gemv_mxfp4_dp4a_prequant(q8, weight_k, scale_k, K)
    r3 = gemv_mxfp4_dp4a_prequant(q8, weight_v, scale_v, K)
torch.cuda.synchronize()
separate_time = (time.perf_counter() - start) / 100 * 1e6

# Warmup fused
out_q = torch.empty(M, N_q, dtype=torch.bfloat16, device="cuda")
out_k = torch.empty(M, N_k, dtype=torch.bfloat16, device="cuda")
out_v = torch.empty(M, N_v, dtype=torch.bfloat16, device="cuda")

for _ in range(10):
    gemv_mxfp4_dp4a_fused_qkv(q8, weight_q, scale_q, weight_k, scale_k, 
                              weight_v, scale_v, K, out_q, out_k, out_v)

# Measure fused
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    gemv_mxfp4_dp4a_fused_qkv(q8, weight_q, scale_q, weight_k, scale_k,
                              weight_v, scale_v, K, out_q, out_k, out_v)
torch.cuda.synchronize()
fused_time = (time.perf_counter() - start) / 100 * 1e6

print(f"3 Separate (Q+K+V):  {separate_time:.2f} μs")
print(f"1 Fused QKV:         {fused_time:.2f} μs")
print(f"Saved per fusion:    {separate_time - fused_time:.2f} μs ({(1-fused_time/separate_time)*100:.1f}%)")

print("\n3. CUDA Graph Potential:")
print("-" * 70)

# Test CUDA graph overhead
try:
    # Create a simple graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(s):
        # Warmup in stream
        for _ in range(3):
            r = gemv_mxfp4_dp4a_prequant(q8, weight_q, scale_q, K)
    
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        r = gemv_mxfp4_dp4a_prequant(q8, weight_q, scale_q, K)
    
    # Warmup replay
    for _ in range(10):
        g.replay()
    
    # Measure graph replay
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - start) / 100 * 1e6
    
    # Compare to regular launch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        r = gemv_mxfp4_dp4a_prequant(q8, weight_q, scale_q, K)
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) / 100 * 1e6
    
    print(f"Regular launch (Q proj): {regular_time:.2f} μs")
    print(f"Graph replay (Q proj):   {graph_time:.2f} μs")
    print(f"Launch overhead saved:   {regular_time - graph_time:.2f} μs")
    
except Exception as e:
    print(f"CUDA Graph test failed: {e}")

print("\n4. Estimated Impact on Full Decode:")
print("-" * 70)

# Estimate for 60 layers
layers = 60
qkv_time = fused_time  # Already fused
o_time = 12  # Estimated
lm_head_time = 1300  # Measured

current_total = layers * (qkv_time + o_time) + lm_head_time
print(f"Current dense layer time: {current_total/1000:.2f} ms")

# With graph (assuming ~2μs launch overhead saved per kernel)
launch_overhead_per_kernel = 3  # μs estimate
kernels_per_layer = 2  # fused_qkv + o
total_launches = layers * kernels_per_layer + 1  # +1 for LM head

if graph_time < regular_time:
    overhead_saved = (regular_time - graph_time) * total_launches / 1000
    print(f"If all in CUDA graph: save ~{overhead_saved:.2f} ms ({overhead_saved/current_total*100:.1f}%)")
