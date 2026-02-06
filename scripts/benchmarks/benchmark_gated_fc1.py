#!/usr/bin/env python3
"""
Benchmark the gated FC1 (SwiGLU) kernel vs standard two-kernel approach.

Measures:
1. Gated FC1 kernel (single fused kernel)
2. Standard: FC1 + separate SwiGLU activation
"""

import os
import sys
import torch
import time

# Enable gated FC1 path
os.environ["FLASHINFER_GATED_FC1_LAUNCH"] = "1"

def precise_benchmark(fn, warmup=20, iters=100):
    """Benchmark using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)])
    return times[len(times)//2]  # median in μs


def main():
    print("Importing flashinfer...")
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
    from flashinfer import mxfp4_quantize, mxfp8_quantize
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Model dimensions (gpt-oss-120b like)
    hidden_size = 2048
    inter_size = 4096  # After gating, output is [M, inter_size]
    num_experts = 8
    topk = 2
    
    # Test various batch sizes (M dimension)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"\nGated FC1 Benchmark")
    print(f"=" * 60)
    print(f"Config: hidden_size={hidden_size}, inter_size={inter_size}, experts={num_experts}, topk={topk}")
    print(f"Tile: 64x64x128 (gated mode)")
    print(f"=" * 60)
    
    # Create weights once
    torch.manual_seed(42)
    
    # FC1 weights: [num_experts, 2*inter_size, hidden_size] (gate + linear)
    fc1_weight_bf16 = torch.randn(num_experts, 2 * inter_size, hidden_size, device=device, dtype=dtype)
    # FC2 weights: [num_experts, hidden_size, inter_size]
    fc2_weight_bf16 = torch.randn(num_experts, hidden_size, inter_size, device=device, dtype=dtype)
    
    # Quantize weights
    fc1_weight_fp4, fc1_scale = mxfp4_quantize(fc1_weight_bf16)
    fc2_weight_fp4, fc2_scale = mxfp4_quantize(fc2_weight_bf16)
    fc1_weight_fp4 = fc1_weight_fp4.view(torch.long)
    fc2_weight_fp4 = fc2_weight_fp4.view(torch.long)
    
    # Prepare scale tensors
    k_blocks = (hidden_size + 31) // 32
    fc1_block_scale = fc1_scale.view(num_experts, 2 * inter_size, k_blocks).contiguous()
    fc1_block_scale = fc1_block_scale.view(torch.int32)
    
    inter_blocks = (inter_size + 31) // 32
    fc2_block_scale = fc2_scale.view(num_experts, hidden_size, inter_blocks).contiguous()
    fc2_block_scale = fc2_block_scale.view(torch.int32)
    
    fc1_global_scale = torch.ones(num_experts, device=device, dtype=torch.float32)
    fc2_global_scale = torch.ones(num_experts, device=device, dtype=torch.float32)
    quant_scales = [fc1_block_scale, fc1_global_scale, fc2_block_scale, fc2_global_scale]
    
    print(f"\n{'Batch':<8} {'Time (μs)':<12} {'Tokens/s':<12} {'TFLOPS':<10}")
    print("-" * 50)
    
    for num_tokens in batch_sizes:
        # Create per-batch inputs
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        
        # Expert assignment
        topk_indices = torch.zeros(num_tokens, topk, device=device, dtype=torch.int32)
        for t in range(num_tokens):
            topk_indices[t, 0] = t % num_experts
            topk_indices[t, 1] = (t + 1) % num_experts
        topk_weights = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
        
        # Quantize activations
        hidden_states_fp8, hidden_states_scale = mxfp8_quantize(hidden_states, True, 32)
        
        def run_gated_fc1():
            return cutlass_fused_moe(
                input=hidden_states_fp8,
                token_selected_experts=topk_indices,
                token_final_scales=topk_weights,
                fc1_expert_weights=fc1_weight_fp4,
                fc2_expert_weights=fc2_weight_fp4,
                output_dtype=dtype,
                quant_scales=quant_scales,
                input_sf=hidden_states_scale,
                activation_type=ActivationType.Swiglu,
                use_mxfp8_act_scaling=True,
            )
        
        # Warmup and benchmark
        try:
            time_us = precise_benchmark(run_gated_fc1, warmup=10, iters=50)
            
            # Calculate throughput
            # FC1 gated: 2 * M * (2*inter_size) * hidden_size flops (2 GEMMs: linear + gate)
            flops = 2 * num_tokens * (2 * inter_size) * hidden_size * 2  # *2 for MAC
            tflops = flops / (time_us * 1e-6) / 1e12
            tokens_per_sec = num_tokens / (time_us * 1e-6)
            
            print(f"{num_tokens:<8} {time_us:<12.1f} {tokens_per_sec:<12.0f} {tflops:<10.2f}")
        except Exception as e:
            print(f"{num_tokens:<8} ERROR: {e}")
    
    print("\n*** Benchmark complete! ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
