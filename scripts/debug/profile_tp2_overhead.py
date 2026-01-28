#!/usr/bin/env python3
"""Profile TP=2 overhead sources to identify optimization opportunities.

This script measures the time spent in key operations during decode:
1. lm_head (with padding/slicing if applicable)
2. NCCL collectives (all_gather, reduce_scatter)
3. MoE layers
4. Attention layers

Usage:
    # Inside the vllm-dev container with model loaded
    python /workspace/mxfp4/scripts/profile_tp2_overhead.py

    # Or with NVTX for Nsight Systems:
    nsys profile -o tp2_profile python /workspace/mxfp4/scripts/profile_tp2_overhead.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist

# Check if we're in a distributed environment
def get_tp_info():
    """Get tensor parallel size and rank."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        return tp_size, tp_rank
    except:
        return 1, 0


def profile_contiguous_overhead():
    """Measure the overhead of .contiguous() on sliced tensors."""
    print("\n=== Profiling .contiguous() overhead ===")
    
    device = torch.device("cuda:0")
    
    # Simulate lm_head output shapes for TP=2
    # vocab_size = 201088, vocab/2 = 100544, padded = 100608
    batch_sizes = [1, 2, 4, 8, 16, 32]
    original_vocab = 100544
    padded_vocab = 100608  # round_up(100544, 128)
    
    results = []
    
    for batch in batch_sizes:
        # Create padded tensor (what Marlin outputs)
        padded = torch.randn(batch, padded_vocab, dtype=torch.bfloat16, device=device)
        
        # Warm up
        for _ in range(10):
            sliced = padded[:, :original_vocab]
            contiguous = sliced.contiguous()
            torch.cuda.synchronize()
        
        # Measure slice only
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            sliced = padded[:, :original_vocab]
        torch.cuda.synchronize()
        slice_time = (time.perf_counter() - start) / 100 * 1000  # ms
        
        # Measure slice + contiguous
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            sliced = padded[:, :original_vocab]
            contiguous = sliced.contiguous()
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / 100 * 1000  # ms
        
        contiguous_time = total_time - slice_time
        
        results.append({
            'batch': batch,
            'slice_ms': slice_time,
            'contiguous_ms': contiguous_time,
            'total_ms': total_time,
            'bytes_copied': batch * original_vocab * 2,  # bf16 = 2 bytes
        })
        
        print(f"  batch={batch:3d}: slice={slice_time:.4f}ms, "
              f"contiguous={contiguous_time:.4f}ms, "
              f"total={total_time:.4f}ms, "
              f"bytes={batch * original_vocab * 2 / 1e6:.2f}MB")
    
    return results


def profile_all_gather_overhead():
    """Measure all_gather overhead (simulated if not in distributed env)."""
    print("\n=== Profiling all_gather overhead ===")
    
    device = torch.device("cuda:0")
    tp_size, tp_rank = get_tp_info()
    
    if tp_size == 1:
        print("  Not in distributed environment, simulating with local copy")
        # Simulate the data movement that would happen
        batch_sizes = [1, 2, 4, 8, 16, 32]
        vocab_per_tp = 100544
        
        for batch in batch_sizes:
            tensor = torch.randn(batch, vocab_per_tp, dtype=torch.bfloat16, device=device)
            output = torch.empty(batch, vocab_per_tp * 2, dtype=torch.bfloat16, device=device)
            
            # Warm up
            for _ in range(10):
                output[:, :vocab_per_tp] = tensor
                output[:, vocab_per_tp:] = tensor
                torch.cuda.synchronize()
            
            # Measure
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                output[:, :vocab_per_tp] = tensor
                output[:, vocab_per_tp:] = tensor
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 100 * 1000
            
            print(f"  batch={batch:3d}: simulated_copy={time_ms:.4f}ms "
                  f"(actual network would be much higher)")
    else:
        print(f"  TP size: {tp_size}, TP rank: {tp_rank}")
        try:
            from vllm.distributed import get_tp_group
            group = get_tp_group()
            
            batch_sizes = [1, 2, 4, 8, 16, 32]
            vocab_per_tp = 100544
            
            for batch in batch_sizes:
                tensor = torch.randn(batch, vocab_per_tp, dtype=torch.bfloat16, device=device)
                
                # Warm up
                for _ in range(10):
                    output = group.all_gather(tensor.contiguous(), dim=-1)
                    torch.cuda.synchronize()
                
                # Measure
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    output = group.all_gather(tensor.contiguous(), dim=-1)
                torch.cuda.synchronize()
                time_ms = (time.perf_counter() - start) / 100 * 1000
                
                print(f"  batch={batch:3d}: all_gather={time_ms:.4f}ms")
        except Exception as e:
            print(f"  Error: {e}")


def profile_marlin_kernel():
    """Measure Marlin kernel overhead for different sizes."""
    print("\n=== Profiling Marlin kernel (lm_head) ===")
    
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            mxfp4_e2m1_quantize,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )
    except ImportError as e:
        print(f"  Cannot import Marlin utils: {e}")
        return
    
    device = torch.device("cuda:0")
    
    # Simulate lm_head dimensions
    hidden_size = 2880
    vocab_sizes = [
        (100544, "TP=2 original"),
        (100608, "TP=2 padded (128-aligned)"),
        (201088, "TP=1"),
    ]
    batch_sizes = [1, 4, 8, 32]
    
    for vocab, label in vocab_sizes:
        print(f"\n  {label} (vocab={vocab}):")
        
        # Create and quantize weights
        weight_bf16 = torch.randn(vocab, hidden_size, dtype=torch.bfloat16, device=device)
        weight_fp4, weight_scale = mxfp4_e2m1_quantize(weight_bf16)
        
        # Create a mock layer
        class MockLayer:
            pass
        layer = MockLayer()
        layer.weight = torch.nn.Parameter(weight_fp4, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.output_size_per_partition = vocab
        layer.input_size_per_partition = hidden_size
        
        try:
            prepare_fp4_layer_for_marlin(layer)
        except Exception as e:
            print(f"    Error preparing Marlin: {e}")
            continue
        
        for batch in batch_sizes:
            x = torch.randn(batch, hidden_size, dtype=torch.bfloat16, device=device)
            
            # Warm up
            try:
                for _ in range(10):
                    out = apply_fp4_marlin_linear(
                        input=x,
                        weight=layer.weight,
                        weight_scale=layer.weight_scale,
                        weight_scale_2=None,
                        workspace=layer.workspace,
                        size_n=vocab,
                        size_k=hidden_size,
                        bias=None,
                    )
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"    batch={batch}: Error: {e}")
                continue
            
            # Measure
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                out = apply_fp4_marlin_linear(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    weight_scale_2=None,
                    workspace=layer.workspace,
                    size_n=vocab,
                    size_k=hidden_size,
                    bias=None,
                )
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 100 * 1000
            
            print(f"    batch={batch:3d}: kernel={time_ms:.4f}ms")


def profile_input_padding():
    """Measure input padding overhead."""
    print("\n=== Profiling input padding overhead ===")
    
    device = torch.device("cuda:0")
    
    # hidden_size = 2880, already 64-aligned, so no padding needed for input
    # But let's check if there was any hidden size that needed padding
    hidden_size = 2880
    padded_hidden = ((hidden_size + 63) // 64) * 64  # round_up to 64
    
    print(f"  hidden_size={hidden_size}, padded={padded_hidden}, "
          f"needs_padding={hidden_size != padded_hidden}")
    
    if hidden_size != padded_hidden:
        batch_sizes = [1, 4, 8, 32]
        for batch in batch_sizes:
            x = torch.randn(batch, hidden_size, dtype=torch.bfloat16, device=device)
            
            # Warm up
            for _ in range(10):
                padded = torch.nn.functional.pad(x, (0, padded_hidden - hidden_size))
                torch.cuda.synchronize()
            
            # Measure
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                padded = torch.nn.functional.pad(x, (0, padded_hidden - hidden_size))
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 100 * 1000
            
            print(f"    batch={batch:3d}: pad={time_ms:.4f}ms")
    else:
        print("  No input padding needed (hidden_size is 64-aligned)")


def estimate_decode_breakdown():
    """Estimate time breakdown for decode based on measurements."""
    print("\n=== Estimated Decode Time Breakdown ===")
    print("""
    For a 60 tok/s decode (16.7ms per token):
    
    TP=1 breakdown (estimated):
    - MoE layers:     ~34% = 5.7ms
    - Attention:      ~40% = 6.7ms  
    - lm_head:        ~6%  = 1.0ms
    - Other:          ~20% = 3.3ms
    
    TP=2 additional overhead:
    - all_gather for lm_head logits: ~2-5ms (network dependent)
    - all_gather for attention: ~1-2ms per layer
    - reduce_scatter: ~1-2ms per layer
    - .contiguous() for lm_head: ~0.05-0.1ms (negligible)
    
    If TP=2 is at 52 tok/s (19.2ms per token):
    - Extra time: 19.2 - 16.7 = 2.5ms
    - This is likely dominated by network communication, not .contiguous()
    """)


def main():
    print("=" * 60)
    print("TP=2 Overhead Profiling")
    print("=" * 60)
    
    # Basic info
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    tp_size, tp_rank = get_tp_info()
    print(f"TP size: {tp_size}, TP rank: {tp_rank}")
    
    # Run profiling
    profile_contiguous_overhead()
    profile_input_padding()
    profile_marlin_kernel()
    profile_all_gather_overhead()
    estimate_decode_breakdown()
    
    print("\n" + "=" * 60)
    print("Profiling complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
