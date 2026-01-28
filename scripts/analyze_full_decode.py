"""Full decode analysis including KV cache, activations, and all memory traffic."""

# gpt-oss-120b config
n_layers = 36
hidden_dim = 2880
n_heads = 64
head_dim = 64
n_kv_heads = 8
intermediate_dim = 2880
n_experts = 128
top_k = 4
vocab_size = 201088

# Assume context length of 2048 for decode
context_len = 2048

print("=" * 70)
print("FULL DECODE MEMORY TRAFFIC ANALYSIS")
print("=" * 70)

# Per-layer memory traffic (all in bytes)

# 1. QKV projection: read weights, read input, write output
qkv_weight_read = hidden_dim * (n_heads + 2*n_kv_heads) * head_dim * 0.5
qkv_scale_read = hidden_dim * (n_heads + 2*n_kv_heads) * head_dim // 32
qkv_input_read = hidden_dim * 2  # BF16
qkv_output_write = (n_heads + 2*n_kv_heads) * head_dim * 2  # BF16

# 2. RoPE: read/write Q and K
rope_traffic = (n_heads + n_kv_heads) * head_dim * 2 * 2  # read + write

# 3. Attention: read Q, read K cache, read V cache, write output
# KV cache size per layer: context_len × n_kv_heads × head_dim
kv_cache_read = context_len * n_kv_heads * head_dim * 2 * 2  # K and V, FP8 cache = 1 byte
attention_q_read = n_heads * head_dim * 2
attention_output_write = n_heads * head_dim * 2

# 4. O projection: read weights, read input, write output
o_weight_read = n_heads * head_dim * hidden_dim * 0.5
o_scale_read = n_heads * head_dim * hidden_dim // 32
o_input_read = n_heads * head_dim * 2
o_output_write = hidden_dim * 2

# 5. Residual + RMSNorm
residual_traffic = hidden_dim * 2 * 4  # read 2, write 2

# 6. Router: read weights, compute routing
router_weight_read = hidden_dim * n_experts * 0.5
router_scale_read = hidden_dim * n_experts // 32
router_traffic = hidden_dim * 2 + n_experts * 4  # input + logits

# 7. MoE FC1 (top-k experts): read weights, read input, write output
moe_fc1_weight_read = hidden_dim * intermediate_dim * 2 * 0.5 * top_k  # gate+up, top-k experts
moe_fc1_scale_read = hidden_dim * intermediate_dim * 2 // 32 * top_k
moe_fc1_input_read = hidden_dim * 2 * top_k
moe_fc1_output_write = intermediate_dim * 2 * 2 * top_k  # gate and up

# 8. SwiGLU: read gate and up, write activated
swiglu_traffic = intermediate_dim * 2 * 3 * top_k  # read 2, write 1

# 9. MoE FC2: read weights, read input, write output
moe_fc2_weight_read = intermediate_dim * hidden_dim * 0.5 * top_k
moe_fc2_scale_read = intermediate_dim * hidden_dim // 32 * top_k
moe_fc2_input_read = intermediate_dim * 2 * top_k
moe_fc2_output_write = hidden_dim * 2 * top_k

# 10. Token combine (scatter-add): read top-k outputs, write combined
token_combine_traffic = hidden_dim * 2 * (top_k + 1)

# 11. Final residual + RMSNorm
final_residual = hidden_dim * 2 * 4

# Per-layer total
layer_weight_traffic = (
    qkv_weight_read + qkv_scale_read +
    o_weight_read + o_scale_read +
    router_weight_read + router_scale_read +
    moe_fc1_weight_read + moe_fc1_scale_read +
    moe_fc2_weight_read + moe_fc2_scale_read
)

layer_activation_traffic = (
    qkv_input_read + qkv_output_write +
    rope_traffic +
    kv_cache_read + attention_q_read + attention_output_write +
    o_input_read + o_output_write +
    residual_traffic +
    router_traffic +
    moe_fc1_input_read + moe_fc1_output_write +
    swiglu_traffic +
    moe_fc2_input_read + moe_fc2_output_write +
    token_combine_traffic +
    final_residual
)

layer_total = layer_weight_traffic + layer_activation_traffic

# LM head
lm_head_weight = hidden_dim * vocab_size * 0.5
lm_head_scale = hidden_dim * vocab_size // 32
lm_head_traffic = lm_head_weight + lm_head_scale + hidden_dim * 2 + vocab_size * 4

# Total per decode step
total_traffic = n_layers * layer_total + lm_head_traffic

print(f"\nPer-layer traffic breakdown:")
print(f"  Weight reads:     {layer_weight_traffic / 1e6:.2f} MB")
print(f"  Activation I/O:   {layer_activation_traffic / 1e6:.2f} MB")
print(f"    - KV cache read: {kv_cache_read / 1e6:.2f} MB (context={context_len})")
print(f"  Layer total:      {layer_total / 1e6:.2f} MB")

print(f"\nTotal per decode step:")
print(f"  {n_layers} layers:  {n_layers * layer_total / 1e6:.2f} MB")
print(f"  LM head:    {lm_head_traffic / 1e6:.2f} MB")
print(f"  TOTAL:      {total_traffic / 1e6:.2f} MB = {total_traffic / 1e9:.3f} GB")

print(f"\n" + "=" * 70)
print("THROUGHPUT ANALYSIS")
print("=" * 70)

peak_bw = 273  # GB/s

time_ms = (total_traffic / 1e9) / peak_bw * 1000
print(f"\nAt {peak_bw} GB/s peak bandwidth:")
print(f"  Time per decode: {time_ms:.2f} ms")
print(f"  Max throughput:  {1000/time_ms:.1f} tok/s")

# Add kernel overhead
n_kernels = 550  # vLLM estimate
overhead_us = 10  # More realistic
overhead_ms = n_kernels * overhead_us / 1000

for eff in [0.5, 0.6, 0.7]:
    actual_bw = peak_bw * eff
    compute_time = (total_traffic / 1e9) / actual_bw * 1000
    total_time = compute_time + overhead_ms
    print(f"\nAt {eff*100:.0f}% efficiency + {overhead_us}μs/kernel:")
    print(f"  Compute: {compute_time:.2f} ms")
    print(f"  Overhead: {overhead_ms:.2f} ms")
    print(f"  Total: {total_time:.2f} ms")
    print(f"  Throughput: {1000/total_time:.1f} tok/s")

print(f"\n" + "=" * 70)
print("KV CACHE IMPACT ANALYSIS")
print("=" * 70)

for ctx in [256, 512, 1024, 2048, 4096]:
    kv_per_layer = ctx * n_kv_heads * head_dim * 2 * 2  # K + V, FP8
    kv_total = n_layers * kv_per_layer
    total_with_kv = n_layers * (layer_weight_traffic + layer_activation_traffic - kv_cache_read + kv_per_layer) + lm_head_traffic
    time_with_kv = (total_with_kv / 1e9) / (peak_bw * 0.6) * 1000 + overhead_ms
    print(f"Context {ctx:4d}: KV cache = {kv_total/1e6:6.1f} MB, "
          f"Total = {total_with_kv/1e9:.2f} GB, "
          f"Time = {time_with_kv:.1f} ms, "
          f"Throughput = {1000/time_with_kv:.1f} tok/s")

print(f"\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
With full memory traffic accounting (weights + activations + KV cache):
- Total memory traffic: ~{total_traffic/1e9:.1f} GB per decode step
- At 60% bandwidth efficiency + kernel overhead: ~{1000/((total_traffic/1e9)/(peak_bw*0.6)*1000 + overhead_ms):.0f} tok/s

This is much closer to the observed ~30 tok/s for vLLM!

The key factors are:
1. KV cache reads (~{n_layers * kv_cache_read / 1e6:.0f} MB for context=2048)
2. Activation read/write traffic (~{n_layers * layer_activation_traffic / 1e6:.0f} MB)
3. Kernel launch overhead (~{overhead_ms:.1f} ms)

llama.cpp is faster because:
1. Fused kernels reduce activation I/O (no intermediate writes)
2. Fewer kernel launches (less overhead)
3. Better cache utilization (C++ vs Python/PyTorch)
4. Pre-quantized activations reused across projections
""")
