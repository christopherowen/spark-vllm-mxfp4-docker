#!/usr/bin/env python3
"""Find the crossover point where cuDNN beats Marlin."""

import torch

def precise_benchmark(fn, warmup=50, iters=200):
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
    
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times)//2]


def main():
    from flashinfer import mxfp4_quantize
    from flashinfer.gemm import mm_fp4
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear, rand_marlin_weight_mxfp4_like
    )

    device = "cuda"
    N, K = 6144, 6144

    print(f"Finding crossover point for N={N}, K={K}")
    print()
    print(f"{'M':<6} {'Marlin (μs)':<15} {'cuDNN (μs)':<15} {'Ratio':<10} {'Winner':<10}")
    print("-" * 60)

    for M in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        # Prepare Marlin data
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        W_placeholder = torch.empty(N, K, dtype=torch.bfloat16, device=device)
        _, marlin_qweight, marlin_scales = rand_marlin_weight_mxfp4_like(W_placeholder, 32, None)
        workspace = torch.zeros(256*1024*1024//4, dtype=torch.int32, device=device)

        # Prepare cuDNN data  
        A_q, A_scale = mxfp4_quantize(A)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        W_q, W_scale = mxfp4_quantize(W)

        # Benchmark
        marlin_time = precise_benchmark(
            lambda: apply_fp4_marlin_linear(A, marlin_qweight, marlin_scales, None, workspace, N, K, None)
        )
        cudnn_time = precise_benchmark(
            lambda: mm_fp4(A_q, W_q.T, A_scale, W_scale.T, None, torch.bfloat16, None, 32, False, "cudnn", False)
        )
        
        ratio = cudnn_time / marlin_time
        winner = "Marlin" if ratio > 1 else "cuDNN"
        print(f"{M:<6} {marlin_time:<15.1f} {cudnn_time:<15.1f} {ratio:<10.2f} {winner:<10}")


if __name__ == "__main__":
    main()
