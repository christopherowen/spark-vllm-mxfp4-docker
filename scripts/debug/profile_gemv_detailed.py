"""Detailed profiling of GEMV kernel to understand where time goes."""
import torch
import time

# gpt-oss-120b dimensions
K = 2880
N = 5120  # QKV projection

def profile_with_events():
    """Use CUDA events for precise timing of kernel phases."""
    from flashinfer.gemv import gemv_mxfp4_dp4a
    
    M = 1
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
    
    # Warmup
    for _ in range(10):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    torch.cuda.synchronize()
    
    # Profile with events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / 100
    print(f"GEMV M=1, N={N}, K={K}: {elapsed_ms:.4f} ms")
    
    # Compute theoretical times
    # Memory traffic for GEMV:
    input_bytes = M * K * 2  # BF16 input
    weight_bytes = N * K // 2  # FP4 packed weights
    scale_bytes = N * K // 32  # E8M0 scales
    output_bytes = M * N * 2  # BF16 output
    total_bytes = input_bytes + weight_bytes + scale_bytes + output_bytes
    
    # Note: input is read N times (once per output row) in current impl
    actual_input_reads = N * M * K * 2  # Redundant reads!
    actual_total = actual_input_reads + weight_bytes + scale_bytes + output_bytes
    
    peak_bw = 273  # GB/s for GB10
    
    theoretical_time_ideal = total_bytes / (peak_bw * 1e9) * 1000  # ms
    theoretical_time_actual = actual_total / (peak_bw * 1e9) * 1000  # ms
    
    achieved_bw = total_bytes / (elapsed_ms / 1000) / 1e9
    
    print(f"\nMemory analysis:")
    print(f"  Input (M×K×2):     {input_bytes:>12,} bytes")
    print(f"  Weights (N×K/2):   {weight_bytes:>12,} bytes")
    print(f"  Scales (N×K/32):   {scale_bytes:>12,} bytes")
    print(f"  Output (M×N×2):    {output_bytes:>12,} bytes")
    print(f"  Total (ideal):     {total_bytes:>12,} bytes")
    print(f"  Actual input reads:{actual_input_reads:>12,} bytes (N× redundant!)")
    print(f"  Total (actual):    {actual_total:>12,} bytes")
    print()
    print(f"Bandwidth analysis:")
    print(f"  Peak bandwidth:    {peak_bw} GB/s")
    print(f"  Theoretical (ideal): {theoretical_time_ideal:.4f} ms")
    print(f"  Theoretical (actual): {theoretical_time_actual:.4f} ms")
    print(f"  Achieved:          {elapsed_ms:.4f} ms")
    print(f"  Achieved bandwidth: {achieved_bw:.1f} GB/s (based on ideal traffic)")
    print(f"  Efficiency:        {theoretical_time_ideal / elapsed_ms * 100:.1f}%")


def profile_breakdown():
    """Break down time into quantization vs dot product."""
    print("\n" + "="*60)
    print("Time Breakdown Analysis")
    print("="*60)
    
    M = 1
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
    
    # 1. Profile just the quantization part (if we had a separate kernel)
    # Simulate: find max + quantize K elements
    
    # Time to find max of K BF16 values
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(1000):
        max_val = input_bf16.abs().max()
    end.record()
    torch.cuda.synchronize()
    quant_time = start.elapsed_time(end) / 1000
    print(f"\nQuantization analysis (PyTorch ops for reference):")
    print(f"  abs().max() time: {quant_time:.4f} ms")
    
    # 2. Profile memory read of weights only
    start.record()
    for _ in range(1000):
        _ = weight_fp4.sum()  # Force read all weights
    end.record()
    torch.cuda.synchronize()
    weight_read = start.elapsed_time(end) / 1000
    print(f"  Weight read time: {weight_read:.4f} ms")
    
    # 3. Profile a simple reduction (simulate dot product reduction)
    temp = torch.randn(N, K, dtype=torch.float32, device="cuda")
    start.record()
    for _ in range(1000):
        _ = temp.sum(dim=1)  # Sum over K for each N
    end.record()
    torch.cuda.synchronize()
    reduction_time = start.elapsed_time(end) / 1000
    print(f"  Reduction (N×K→N): {reduction_time:.4f} ms")


def profile_marlin_comparison():
    """Direct comparison with Marlin internals."""
    print("\n" + "="*60)
    print("Marlin Comparison")
    print("="*60)
    
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
    except ImportError as e:
        print(f"Marlin import failed: {e}")
        return
    
    M = 1
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            )
            self.weight_scale = torch.nn.Parameter(
                torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            )
            self.params_dtype = torch.bfloat16
            self.input_size_per_partition = K
            self.output_size_per_partition = N

    layer = MockLayer().cuda()
    prepare_fp4_layer_for_marlin(layer, input_dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        out = apply_fp4_marlin_linear(
            input=input_bf16,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_scale_2=None,
            workspace=layer.workspace,
            size_n=N,
            size_k=K,
            bias=None,
        )
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        out = apply_fp4_marlin_linear(
            input=input_bf16,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_scale_2=None,
            workspace=layer.workspace,
            size_n=N,
            size_k=K,
            bias=None,
        )
    end.record()
    torch.cuda.synchronize()
    
    marlin_time = start.elapsed_time(end) / 100
    
    # Marlin memory traffic:
    # - Reads FP4 weights: N * K / 2 bytes
    # - Reads scales: N * K / 32 bytes  
    # - Reads BF16 input: M * K * 2 bytes
    # - Writes BF16 output: M * N * 2 bytes
    marlin_bytes = N * K // 2 + N * K // 32 + M * K * 2 + M * N * 2
    marlin_bw = marlin_bytes / (marlin_time / 1000) / 1e9
    
    print(f"\nMarlin M=1, N={N}, K={K}:")
    print(f"  Time: {marlin_time:.4f} ms")
    print(f"  Memory traffic: {marlin_bytes:,} bytes")
    print(f"  Achieved bandwidth: {marlin_bw:.1f} GB/s")
    
    # Marlin also uses tensor cores, so compute is essentially free
    # The question is: why is Marlin 1.8x faster?
    # Answer: No activation quantization, uses tensor cores
    
    print("\nWhy Marlin is faster:")
    print("  1. No activation quantization (BF16 used directly)")
    print("  2. Tensor cores for compute (vs DP4A on CUDA cores)")
    print("  3. Optimized memory access patterns (years of tuning)")


def analyze_kernel_launch():
    """Analyze kernel launch overhead."""
    print("\n" + "="*60)
    print("Kernel Launch Overhead Analysis")
    print("="*60)
    
    from flashinfer.gemv import gemv_mxfp4_dp4a
    
    M = 1
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
    
    # Warmup
    for _ in range(10):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    torch.cuda.synchronize()
    
    # Time single call vs batched
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Single call
    torch.cuda.synchronize()
    start.record()
    out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    end.record()
    torch.cuda.synchronize()
    single_time = start.elapsed_time(end)
    
    # 10 calls
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    end.record()
    torch.cuda.synchronize()
    ten_time = start.elapsed_time(end)
    
    print(f"\nKernel launch overhead:")
    print(f"  1 call:  {single_time:.4f} ms")
    print(f"  10 calls: {ten_time:.4f} ms")
    print(f"  Per-call (from 10): {ten_time/10:.4f} ms")
    print(f"  Launch overhead estimate: {single_time - ten_time/10:.4f} ms")


if __name__ == "__main__":
    print("="*60)
    print("Detailed GEMV Profiling")
    print("="*60)
    
    profile_with_events()
    profile_breakdown()
    profile_marlin_comparison()
    analyze_kernel_launch()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
The GEMV kernel is slower than Marlin because:

1. REDUNDANT QUANTIZATION: Input x is quantized N times (once per output row)
   - For N=5120, that's 5120 redundant max-reductions and quantizations
   - Each takes ~0.003ms, totaling ~15ms of pure waste (but cached helps)

2. CUDA CORES vs TENSOR CORES:
   - GEMV uses DP4A on CUDA cores (~1 TFLOP/s effective)
   - Marlin uses BF16 tensor cores (~10+ TFLOP/s)

3. MEMORY ACCESS PATTERNS:
   - GEMV: One block per output row, strided access
   - Marlin: Optimized for coalesced access and cache reuse

To fix this properly:
   - Pre-quantize input ONCE (separate kernel)
   - Use SM121 FP4 tensor cores directly (not DP4A)
   - Or just use Marlin (it works and it's fast)
""")
