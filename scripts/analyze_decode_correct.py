"""Correct analysis with actual gpt-oss-120b config."""

# ACTUAL gpt-oss-120b config (from config.json)
n_layers = 36  # NOT 60!
hidden_dim = 2880
n_heads = 64
head_dim = 64  # 64 * 64 = 4096 attention dim
n_kv_heads = 8
intermediate_dim = 2880  # NOT 5888! (per expert)
n_experts = 128  # NOT 8!
top_k = 4  # NOT 8!
vocab_size = 201088

# Derived dimensions
qkv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim  # Q + K + V = 4096 + 512 + 512 = 5120
attn_dim = n_heads * head_dim  # 4096

print("=" * 70)
print("gpt-oss-120b CORRECT CONFIG")
print("=" * 70)
print(f"Layers:          {n_layers}")
print(f"Hidden dim:      {hidden_dim}")
print(f"Experts:         {n_experts} (top-k={top_k})")
print(f"Intermediate:    {intermediate_dim} per expert")
print(f"Attention heads: {n_heads} (KV heads: {n_kv_heads})")

# Weight sizes per layer (FP4 = 0.5 bytes per element)
qkv_weights = hidden_dim * qkv_dim * 0.5  # [2880, 5120] packed
qkv_scales = hidden_dim * qkv_dim // 32   
o_weights = attn_dim * hidden_dim * 0.5   # [4096, 2880] packed
o_scales = attn_dim * hidden_dim // 32

# MoE weights PER EXPERT (gate_up fused + down)
# gate_up: [hidden, 2*intermediate] = [2880, 5760]
# down: [intermediate, hidden] = [2880, 2880]
fc1_weights_per_expert = hidden_dim * intermediate_dim * 2 * 0.5  # gate + up
fc1_scales_per_expert = hidden_dim * intermediate_dim * 2 // 32
fc2_weights_per_expert = intermediate_dim * hidden_dim * 0.5
fc2_scales_per_expert = intermediate_dim * hidden_dim // 32

# For TOP-K=4, we only read 4 experts per decode
active_fc1_weights = fc1_weights_per_expert * top_k
active_fc1_scales = fc1_scales_per_expert * top_k
active_fc2_weights = fc2_weights_per_expert * top_k
active_fc2_scales = fc2_scales_per_expert * top_k

# Router weights (small - routes to 128 experts)
router_weights = hidden_dim * n_experts * 0.5
router_scales = hidden_dim * n_experts // 32

# Per-layer weights READ (not stored) - only active experts!
layer_weights_read = (
    qkv_weights + qkv_scales +
    o_weights + o_scales +
    active_fc1_weights + active_fc1_scales +
    active_fc2_weights + active_fc2_scales +
    router_weights + router_scales
)

# Total MoE weights STORED (all 128 experts)
total_moe_weights = (fc1_weights_per_expert + fc1_scales_per_expert + 
                     fc2_weights_per_expert + fc2_scales_per_expert) * n_experts

# LM head
lm_head_weights = hidden_dim * vocab_size * 0.5
lm_head_scales = hidden_dim * vocab_size // 32

# Total weights READ per decode step
total_weights_read = n_layers * layer_weights_read + lm_head_weights + lm_head_scales

print(f"\n" + "=" * 70)
print("Per-layer weights READ (top-k={} of {} experts):".format(top_k, n_experts))
print("=" * 70)
print(f"  QKV:    {(qkv_weights + qkv_scales) / 1e6:.2f} MB")
print(f"  O:      {(o_weights + o_scales) / 1e6:.2f} MB")
print(f"  Router: {(router_weights + router_scales) / 1e6:.3f} MB")
print(f"  MoE FC1 ({top_k} experts): {(active_fc1_weights + active_fc1_scales) / 1e6:.2f} MB")
print(f"  MoE FC2 ({top_k} experts): {(active_fc2_weights + active_fc2_scales) / 1e6:.2f} MB")
print(f"  Layer total: {layer_weights_read / 1e6:.2f} MB")

print(f"\nTotal weights READ per decode step:")
print(f"  {n_layers} layers: {n_layers * layer_weights_read / 1e6:.2f} MB")
print(f"  LM head: {(lm_head_weights + lm_head_scales) / 1e6:.2f} MB")
print(f"  TOTAL: {total_weights_read / 1e6:.2f} MB = {total_weights_read / 1e9:.3f} GB")

print(f"\n" + "=" * 70)
print("DECODE THROUGHPUT ANALYSIS (CORRECTED)")
print("=" * 70)

peak_bandwidth_gbs = 273  # GB/s for GB10

# Time to read weights for one decode step
weight_read_time_ms = (total_weights_read / 1e9) / peak_bandwidth_gbs * 1000

print(f"\nTheoretical minimum decode time (memory-bound):")
print(f"  Weight read at {peak_bandwidth_gbs} GB/s: {weight_read_time_ms:.2f} ms")
print(f"  Max throughput: {1000 / weight_read_time_ms:.1f} tok/s")

# With realistic efficiency
print(f"\nWith realistic bandwidth efficiency:")
for efficiency in [0.6, 0.7, 0.8, 0.9]:
    actual_bw = peak_bandwidth_gbs * efficiency
    actual_time = (total_weights_read / 1e9) / actual_bw * 1000
    print(f"  At {efficiency*100:.0f}% efficiency ({actual_bw:.0f} GB/s): {1000/actual_time:.1f} tok/s")

print(f"\n" + "=" * 70)
print("KERNEL OVERHEAD IMPACT")
print("=" * 70)

# Corrected kernel count for 36 layers
n_kernels_vllm = 36 * 15 + 10  # ~15 kernels per layer + LM head etc
n_kernels_llama = 36 * 6 + 5   # ~6 kernels per layer (fused)

for name, n_kernels, overhead_us in [("vLLM", n_kernels_vllm, 5), ("llama.cpp", n_kernels_llama, 2)]:
    overhead_ms = n_kernels * overhead_us / 1000
    total_time = weight_read_time_ms / 0.8 + overhead_ms  # 80% efficiency
    print(f"\n{name}:")
    print(f"  Kernels: {n_kernels}")
    print(f"  Launch overhead: {overhead_ms:.2f} ms")
    print(f"  Compute time (80% BW): {weight_read_time_ms / 0.8:.2f} ms")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Throughput: {1000/total_time:.1f} tok/s")

print(f"\n" + "=" * 70)
print("COMPARISON WITH BENCHMARKS")
print("=" * 70)
print(f"""
Actual measured throughput:
  - llama.cpp: ~58 tok/s
  - vLLM:      ~30 tok/s

Our analysis (with corrected model config):
  - Theoretical max: {1000/weight_read_time_ms:.1f} tok/s
  - llama.cpp estimate: ~{1000/(weight_read_time_ms/0.8 + n_kernels_llama*2/1000):.0f} tok/s
  - vLLM estimate: ~{1000/(weight_read_time_ms/0.8 + n_kernels_vllm*5/1000):.0f} tok/s

The gap is explained by:
1. Kernel fusion (llama.cpp has ~{n_kernels_vllm - n_kernels_llama} fewer kernels)
2. Lower launch overhead (C++ vs Python: 2μs vs 5μs)
3. Better memory access patterns in fused kernels
""")

print("=" * 70)
print("KEY INSIGHT: MoE SPARSITY")
print("=" * 70)
print(f"""
With 128 experts but top-k=4:
  - Only {top_k}/{n_experts} = {top_k/n_experts*100:.1f}% of MoE weights read per step
  - This DRAMATICALLY reduces memory bandwidth requirement
  - Makes the model much faster than a dense model of similar size
  
If we had to read ALL expert weights:
  - Would need to read {n_layers * total_moe_weights / 1e9:.2f} GB of MoE weights
  - Total would be ~{(n_layers * (total_moe_weights + qkv_weights + o_weights) + lm_head_weights) / 1e9:.1f} GB
  - Max throughput would be ~{1000 / ((n_layers * (total_moe_weights + qkv_weights + o_weights) + lm_head_weights) / 1e9 / peak_bandwidth_gbs * 1000):.1f} tok/s (much slower!)
""")
