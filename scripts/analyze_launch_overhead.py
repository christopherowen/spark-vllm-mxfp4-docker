#!/usr/bin/env python3
"""Analyze launch overhead more carefully."""

import torch
import time

print("=" * 70)
print("Launch Overhead Analysis")
print("=" * 70)

K = 2880
N = 2880
M = 1

from flashinfer.gemv import gemv_mxfp4_dp4a_prequant, quantize_activations_q8

# Allocate tensors
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
q8_buffer = torch.empty((M, K // 32, 36), dtype=torch.uint8, device="cuda")
weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127

# Pre-quantize
quantize_activations_q8(input_bf16, q8_buffer)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range(10):
    _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
torch.cuda.synchronize()

print("\n1. Single kernel timing (CUDA events):")
start.record()
for _ in range(100):
    _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
end.record()
torch.cuda.synchronize()
single_kernel_us = start.elapsed_time(end) * 1000 / 100
print(f"  Single kernel: {single_kernel_us:.2f} μs")

print("\n2. Back-to-back kernel timing (varying count):")
for count in [1, 10, 50, 100, 200]:
    start.record()
    for _ in range(count):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    per_kernel_us = total_ms * 1000 / count
    print(f"  {count:3d} kernels: {total_ms:.3f} ms total, {per_kernel_us:.2f} μs each")

print("\n3. Python overhead measurement:")
# Measure pure Python overhead without any CUDA work
import time
py_times = []
for _ in range(100):
    t0 = time.perf_counter_ns()
    # Just call the function but measure from Python
    _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    py_times.append(time.perf_counter_ns() - t0)
torch.cuda.synchronize()
avg_py_ns = sum(py_times) / len(py_times)
print(f"  Python call overhead: {avg_py_ns/1000:.1f} μs (includes kernel + return)")

print("\n4. With synchronization after each call:")
sync_times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
    torch.cuda.synchronize()
    sync_times.append(time.perf_counter_ns() - t0)
avg_sync_ns = sum(sync_times) / len(sync_times)
print(f"  With sync: {avg_sync_ns/1000:.1f} μs (true end-to-end)")

print("\n5. Graph capture test:")
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    graph_result = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)

# Warmup
for _ in range(10):
    graph.replay()
torch.cuda.synchronize()

start.record()
for _ in range(100):
    graph.replay()
end.record()
torch.cuda.synchronize()
graph_us = start.elapsed_time(end) * 1000 / 100
print(f"  Graph replay: {graph_us:.2f} μs")
print(f"  Normal:       {single_kernel_us:.2f} μs")
print(f"  Savings:      {single_kernel_us - graph_us:.2f} μs")

# Compare batched
print("\n6. Batched comparison (100 kernels):")
start.record()
for _ in range(100):
    _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)
end.record()
torch.cuda.synchronize()
normal_100 = start.elapsed_time(end)

graph100 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph100):
    for _ in range(100):
        _ = gemv_mxfp4_dp4a_prequant(q8_buffer, weight, scale, K)

for _ in range(3):
    graph100.replay()
torch.cuda.synchronize()

start.record()
for _ in range(10):  # 10 x 100 = 1000 kernels
    graph100.replay()
end.record()
torch.cuda.synchronize()
graph_100 = start.elapsed_time(end) / 10  # Per 100 kernels

print(f"  Normal 100:  {normal_100:.3f} ms")
print(f"  Graph 100:   {graph_100:.3f} ms")
print(f"  Savings:     {normal_100 - graph_100:.3f} ms ({(normal_100 - graph_100) * 10:.1f} μs per kernel)")
