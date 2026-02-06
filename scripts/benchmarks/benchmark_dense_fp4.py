#!/usr/bin/env python3
"""
Benchmark: Marlin vs CUTLASS/cuDNN for MXFP4 Dense GEMM

This script compares the current Marlin path (used for QKV/O/lm_head)
against FlashInfer's mm_fp4 backends without requiring vLLM rebuilds.

Usage:
    python benchmark_dense_fp4.py
    python benchmark_dense_fp4.py --shapes decode  # Just decode shapes (M=1)
    python benchmark_dense_fp4.py --shapes all     # All shapes
"""

import argparse
import time
import torch
import sys

# Timing utilities
def benchmark_fn(fn, warmup=10, iters=100):
    """Benchmark a function, return median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    
    times.sort()
    return times[len(times) // 2]  # Median


def check_imports():
    """Check which backends are available."""
    backends = {}
    
    # Check FlashInfer
    try:
        from flashinfer import mxfp4_quantize, nvfp4_quantize
        from flashinfer.gemm import mm_fp4
        backends["flashinfer"] = True
    except ImportError as e:
        print(f"FlashInfer not available: {e}")
        backends["flashinfer"] = False
    
    # Check Marlin (via vLLM)
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            rand_marlin_weight_mxfp4_like,
        )
        backends["marlin"] = True
    except ImportError as e:
        print(f"Marlin not available: {e}")
        backends["marlin"] = False
    
    return backends


def benchmark_flashinfer_mxfp4(M, N, K, backend="cudnn"):
    """Benchmark FlashInfer mm_fp4 with MXFP4."""
    from flashinfer import mxfp4_quantize
    from flashinfer.gemm import mm_fp4
    
    device = "cuda"
    
    # Create and quantize test data
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    
    A_q, A_scale = mxfp4_quantize(A)
    W_q, W_scale = mxfp4_quantize(W)
    
    # IMPORTANT: Don't use .contiguous() on transposed tensors!
    W_q_T = W_q.T
    W_scale_T = W_scale.T
    
    # Run once to compile/warm up
    out = mm_fp4(
        A_q, W_q_T,
        A_scale, W_scale_T,
        alpha=None,
        out_dtype=torch.bfloat16,
        block_size=32,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=False,
    )
    
    # Benchmark
    time_us = benchmark_fn(lambda: mm_fp4(
        A_q, W_q_T,
        A_scale, W_scale_T,
        alpha=None,
        out_dtype=torch.bfloat16,
        block_size=32,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=False,
    ))
    
    return time_us, out.shape


def benchmark_flashinfer_nvfp4(M, N, K, backend="cutlass"):
    """Benchmark FlashInfer mm_fp4 with NVFP4."""
    from flashinfer import nvfp4_quantize, SfLayout
    from flashinfer.gemm import mm_fp4
    
    device = "cuda"
    
    # Create test data
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    
    # Calculate global scale
    global_sf_a = (448 * 6) / A.float().abs().nan_to_num().max()
    global_sf_w = (448 * 6) / W.float().abs().nan_to_num().max()
    alpha = torch.tensor([1.0 / (global_sf_a * global_sf_w)], dtype=torch.float32, device=device)
    
    # Quantize
    A_q, A_scale = nvfp4_quantize(A, global_sf_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    W_q, W_scale = nvfp4_quantize(W, global_sf_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    
    # IMPORTANT: Don't use .contiguous() on transposed tensors!
    W_q_T = W_q.T
    W_scale_T = W_scale.T
    
    # Run once
    out = mm_fp4(
        A_q, W_q_T,
        A_scale, W_scale_T,
        alpha=alpha,
        out_dtype=torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
    )
    
    # Benchmark
    time_us = benchmark_fn(lambda: mm_fp4(
        A_q, W_q_T,
        A_scale, W_scale_T,
        alpha=alpha,
        out_dtype=torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
    ))
    
    return time_us, out.shape


def benchmark_marlin_mxfp4(M, N, K):
    """Benchmark Marlin with MXFP4."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        rand_marlin_weight_mxfp4_like,
    )
    
    device = "cuda"
    group_size = 32  # MXFP4 group size
    
    # Create activation
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Create random Marlin-compatible weight
    W_placeholder = torch.empty(N, K, dtype=torch.bfloat16, device=device)
    weight_ref, marlin_qweight, marlin_scales = rand_marlin_weight_mxfp4_like(
        W_placeholder, group_size, input_dtype=None
    )
    
    # Workspace
    max_workspace_size = 256 * 1024 * 1024
    workspace = torch.zeros(max_workspace_size // 4, dtype=torch.int32, device=device)
    
    # Run once
    out = apply_fp4_marlin_linear(
        input=A,
        weight=marlin_qweight,
        weight_scale=marlin_scales,
        weight_scale_2=None,
        workspace=workspace,
        size_n=N,
        size_k=K,
        bias=None,
    )
    
    # Benchmark
    time_us = benchmark_fn(lambda: apply_fp4_marlin_linear(
        input=A,
        weight=marlin_qweight,
        weight_scale=marlin_scales,
        weight_scale_2=None,
        workspace=workspace,
        size_n=N,
        size_k=K,
        bias=None,
    ))
    
    return time_us, out.shape


def main():
    parser = argparse.ArgumentParser(description="Benchmark Marlin vs CUTLASS for FP4 GEMM")
    parser.add_argument("--shapes", choices=["decode", "prefill", "all"], default="all",
                        help="Which shapes to benchmark")
    args = parser.parse_args()
    
    print("=" * 90)
    print("Dense FP4 GEMM Benchmark: Marlin vs CUTLASS/cuDNN")
    print("=" * 90)
    
    # Check available backends
    print("\nChecking available backends...")
    backends = check_imports()
    print(f"  FlashInfer:    {'✓' if backends.get('flashinfer') else '✗'}")
    print(f"  Marlin:        {'✓' if backends.get('marlin') else '✗'}")
    
    if not any(backends.values()):
        print("\nNo backends available!")
        sys.exit(1)
    
    # Define test shapes based on gpt-oss-120b architecture
    decode_shapes = [
        (1, 6144, 6144, "QKV proj (M=1)"),
        (1, 256000, 6144, "lm_head (M=1)"),
    ]
    
    prefill_shapes = [
        (32, 6144, 6144, "QKV proj (M=32)"),
        (128, 6144, 6144, "QKV proj (M=128)"),
        (512, 6144, 6144, "QKV proj (M=512)"),
        (128, 256000, 6144, "lm_head (M=128)"),
    ]
    
    if args.shapes == "decode":
        shapes = decode_shapes
    elif args.shapes == "prefill":
        shapes = prefill_shapes
    else:
        shapes = decode_shapes + prefill_shapes
    
    # Run benchmarks
    print("\n" + "=" * 90)
    print(f"{'Description':<20} {'Backend':<20} {'Time (μs)':<12} {'vs Marlin':<12}")
    print("=" * 90)
    
    for M, N, K, desc in shapes:
        results = {}
        marlin_time = None
        
        # Marlin MXFP4
        if backends.get("marlin"):
            try:
                time_us, shape = benchmark_marlin_mxfp4(M, N, K)
                results["Marlin MXFP4"] = time_us
                marlin_time = time_us
            except Exception as e:
                results["Marlin MXFP4"] = f"Error: {str(e)[:30]}"
        
        # FlashInfer cuDNN MXFP4
        if backends.get("flashinfer"):
            try:
                time_us, shape = benchmark_flashinfer_mxfp4(M, N, K, backend="cudnn")
                results["cuDNN MXFP4"] = time_us
            except Exception as e:
                results["cuDNN MXFP4"] = f"Error: {str(e)[:30]}"
        
        # FlashInfer CUTLASS NVFP4
        if backends.get("flashinfer"):
            try:
                time_us, shape = benchmark_flashinfer_nvfp4(M, N, K, backend="cutlass")
                results["CUTLASS NVFP4"] = time_us
            except Exception as e:
                results["CUTLASS NVFP4"] = f"Error: {str(e)[:30]}"
        
        # FlashInfer CUTLASS MXFP4 (our goal)
        if backends.get("flashinfer"):
            try:
                time_us, shape = benchmark_flashinfer_mxfp4(M, N, K, backend="cutlass")
                results["CUTLASS MXFP4"] = time_us
            except ValueError as e:
                if "Only cudnn and auto" in str(e):
                    results["CUTLASS MXFP4"] = "Not implemented"
                else:
                    results["CUTLASS MXFP4"] = f"Error: {str(e)[:30]}"
            except Exception as e:
                results["CUTLASS MXFP4"] = f"Error: {str(e)[:30]}"
        
        # Print results
        first = True
        for backend, result in results.items():
            desc_str = desc if first else ""
            first = False
            
            if isinstance(result, float):
                time_str = f"{result:.1f}"
                if marlin_time and backend != "Marlin MXFP4":
                    speedup = marlin_time / result
                    speedup_str = f"{speedup:.2f}x"
                elif backend == "Marlin MXFP4":
                    speedup_str = "baseline"
                else:
                    speedup_str = "-"
            else:
                time_str = "N/A"
                speedup_str = result[:12] if len(result) > 12 else result
            
            print(f"{desc_str:<20} {backend:<20} {time_str:<12} {speedup_str:<12}")
        
        print("-" * 90)
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("""
Backends:
  - Marlin MXFP4:    Current vLLM path (dequant to BF16, then BF16 compute)
  - cuDNN MXFP4:     FlashInfer cuDNN (native FP4 tensor cores)
  - CUTLASS NVFP4:   FlashInfer CUTLASS with NVFP4 (group size 16)
  - CUTLASS MXFP4:   Goal: FlashInfer CUTLASS with MXFP4 (group size 32)

If cuDNN MXFP4 beats Marlin, CUTLASS MXFP4 should be similar or faster,
giving us a unified CUTLASS backend for all layers (MoE + dense).
""")


if __name__ == "__main__":
    main()
