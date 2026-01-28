"""Analyze memory bandwidth requirements for gpt-oss-120b decode."""

# Model parameters
n_layers = 60
hidden_dim = 2880
n_heads = 64
head_dim = 64  # 64 * 64 = 4096 attention dim
n_kv_heads = 8
intermediate_dim = 5888  # per expert
n_experts = 8
top_k = 8
vocab_size = 201088

# Derived dimensions
qkv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim  # Q + K + V
attn_dim = n_heads * head_dim  # 4096

print("=" * 70)
print("gpt-oss-120b DECODE MEMORY ANALYSIS (M=1)")
print("=" * 70)

# Weight sizes per layer (FP4 = 0.5 bytes per element)
qkv_weights = hidden_dim * qkv_dim * 0.5  # [2880, 5120] packed
qkv_scales = hidden_dim * qkv_dim // 32   # [2880, 160] 
o_weights = attn_dim * hidden_dim * 0.5   # [4096, 2880] packed
o_scales = attn_dim * hidden_dim // 32

# MoE weights per expert
fc1_weights_per_expert = hidden_dim * intermediate_dim * 2 * 0.5  # gate + up
fc1_scales_per_expert = hidden_dim * intermediate_dim * 2 // 32
fc2_weights_per_expert = intermediate_dim * hidden_dim * 0.5
fc2_scales_per_expert = intermediate_dim * hidden_dim // 32

# Total MoE weights (8 experts)
moe_fc1_weights = fc1_weights_per_expert * n_experts
moe_fc1_scales = fc1_scales_per_expert * n_experts
moe_fc2_weights = fc2_weights_per_expert * n_experts
moe_fc2_scales = fc2_scales_per_expert * n_experts

# Router weights
router_weights = hidden_dim * n_experts * 0.5
router_scales = hidden_dim * n_experts // 32

# Per-layer total
layer_weights = (
    qkv_weights + qkv_scales +
    o_weights + o_scales +
    moe_fc1_weights + moe_fc1_scales +
    moe_fc2_weights + moe_fc2_scales +
    router_weights + router_scales
)

# LM head
lm_head_weights = hidden_dim * vocab_size * 0.5
lm_head_scales = hidden_dim * vocab_size // 32

# Total model weights
total_weights = n_layers * layer_weights + lm_head_weights + lm_head_scales

print(f"\nPer-layer weights:")
print(f"  QKV:    {(qkv_weights + qkv_scales) / 1e6:.2f} MB")
print(f"  O:      {(o_weights + o_scales) / 1e6:.2f} MB")
print(f"  Router: {(router_weights + router_scales) / 1e6:.2f} MB")
print(f"  MoE FC1 (8 experts): {(moe_fc1_weights + moe_fc1_scales) / 1e6:.2f} MB")
print(f"  MoE FC2 (8 experts): {(moe_fc2_weights + moe_fc2_scales) / 1e6:.2f} MB")
print(f"  Layer total: {layer_weights / 1e6:.2f} MB")

print(f"\nModel totals:")
print(f"  60 layers: {n_layers * layer_weights / 1e9:.2f} GB")
print(f"  LM head: {(lm_head_weights + lm_head_scales) / 1e6:.2f} MB")
print(f"  Total weights: {total_weights / 1e9:.2f} GB")

# For M=1 decode, we read ALL weights once
# (ignoring activation memory which is negligible for M=1)

print("\n" + "=" * 70)
print("DECODE THROUGHPUT ANALYSIS")
print("=" * 70)

peak_bandwidth_gbs = 273  # GB/s for GB10

# Time to read all weights
weight_read_time_ms = (total_weights / 1e9) / peak_bandwidth_gbs * 1000

print(f"\nTheoretical minimum decode time (memory-bound):")
print(f"  Weight read at {peak_bandwidth_gbs} GB/s: {weight_read_time_ms:.2f} ms")
print(f"  Max throughput: {1000 / weight_read_time_ms:.1f} tok/s")

# With realistic efficiency
for efficiency in [0.5, 0.6, 0.7, 0.8]:
    actual_bw = peak_bandwidth_gbs * efficiency
    actual_time = (total_weights / 1e9) / actual_bw * 1000
    print(f"  At {efficiency*100:.0f}% efficiency ({actual_bw:.0f} GB/s): {1000/actual_time:.1f} tok/s")

print("\n" + "=" * 70)
print("KERNEL LAUNCH OVERHEAD IMPACT")
print("=" * 70)

n_kernels = 900  # From our analysis

for launch_overhead_us in [2, 5, 10, 20]:
    overhead_ms = n_kernels * launch_overhead_us / 1000
    total_time = weight_read_time_ms + overhead_ms
    print(f"\n{launch_overhead_us}μs per kernel ({n_kernels} kernels):")
    print(f"  Overhead: {overhead_ms:.2f} ms")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Throughput: {1000/total_time:.1f} tok/s")
    print(f"  Overhead fraction: {overhead_ms/total_time*100:.1f}%")

print("\n" + "=" * 70)
print("llama.cpp COMPARISON")
print("=" * 70)

# llama.cpp with fewer kernels and lower overhead
llama_kernels = 400
llama_overhead_us = 2  # C++ is faster

print(f"\nllama.cpp estimates:")
llama_overhead_ms = llama_kernels * llama_overhead_us / 1000
llama_total = weight_read_time_ms + llama_overhead_ms
print(f"  Kernels: {llama_kernels}")
print(f"  Overhead: {llama_overhead_ms:.2f} ms")
print(f"  Total time: {llama_total:.2f} ms")
print(f"  Throughput: {1000/llama_total:.1f} tok/s")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The performance gap between llama.cpp (58 tok/s) and vLLM (30 tok/s) is NOT
primarily about GEMV vs Marlin kernel performance. It's about:

1. KERNEL COUNT: ~900 vs ~400 kernels (2.25x more launches)
2. LAUNCH OVERHEAD: ~5-10μs (Python) vs ~2μs (C++) per kernel
3. KERNEL FUSION: llama.cpp fuses GEMV+GLU, top-k+softmax, etc.
4. MEMORY REUSE: Pre-quantized activations used across multiple GEMVs

The individual kernel performance (Marlin vs DP4A) matters less than
the system-level optimization (fewer launches, better fusion, lower overhead).

To match llama.cpp, vLLM needs:
1. Kernel fusion (reduce kernel count by 2x)
2. CUDA graphs (eliminate Python launch overhead)
3. Fused GEMV+activation where applicable
""")
