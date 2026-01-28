"""Compare Marlin vs CUTLASS FP4 GEMM for dense layers.

Key questions:
1. How does Marlin handle M=1?
2. Could CUTLASS FP4 GEMM replace Marlin?
3. What are the tradeoffs?
"""

print("""
================================================================================
MARLIN vs CUTLASS FP4 GEMM COMPARISON FOR DENSE LAYERS
================================================================================

MARLIN (vLLM's current approach):
---------------------------------
Architecture:
- Uses Tensor Cores for BF16 computation
- Dequantizes FP4 weights → BF16 on-the-fly
- Uses ldmatrix for efficient shared memory → register loads

Small M (M ≤ 8) handling:
- Uses m_block_size_8 = true when M ≤ 8
- Loads 2x 8x8 fragments instead of 4x 8x8 fragments
- Still uses tensor cores (no fallback to CUDA cores)
- Uses ldsm<2> instead of ldsm<4> for matrix fragments

Key advantages:
1. Tensor Core utilization even for M=1
2. BF16 activations used directly (no quantization needed)
3. Near-zero weight dequant overhead (bitwise ops)
4. Highly optimized for years

CUTLASS FP4 GEMM (FlashInfer's SM120 kernel):
---------------------------------------------
Architecture:
- Uses SM120's native FP8×FP4 tensor cores (block-scaled MMA)
- Requires activations to be FP8 (needs quantization)
- Uses TMA for efficient global → shared memory loads

Small M handling:
- Minimum tile M = 64 (tcgen05 hardware constraint)
- For M < 64: Uses swap_ab trick (transpose problem)
- Physical tile becomes (N, M) instead of (M, N)
- Still uses tensor cores (native FP4 support)

Key advantages:
1. Native FP4 tensor core support (no dequant needed for weights)
2. Block-scaled operations (E8M0 scales built into hardware)
3. Higher theoretical throughput for FP4

COMPARISON FOR M=1 QKV (K=2880, N=5120):
----------------------------------------
Operation                    Marlin              CUTLASS FP4
-----------------------------------------------------------------
Activation prep              None (BF16)         Quantize BF16→FP8
Weight handling              Dequant FP4→BF16    Native FP4
Tensor Core type             BF16 MMA            FP8×FP4 MMA
Minimum M tile               8 (m_block_size_8)  64 (swap to N)
Memory read (weights)        7.37 MB             7.37 MB
Memory read (scales)         0.46 MB             0.46 MB
Memory read (input)          5.6 KB (BF16)       2.8 KB (FP8)
Compute                      BF16 tensor cores   FP8×FP4 tensor cores

THEORETICAL ANALYSIS:
---------------------
Memory bandwidth is the bottleneck for M=1 (memory-bound).
Both read ~7.83 MB of weights per QKV projection.

At 273 GB/s:
- Weight read time: 7.83 MB / 273 GB/s = 28.7 µs per projection
- 36 layers × 4 projections = ~4.1 ms (weights only)

The kernel choice (Marlin vs CUTLASS) matters less than:
1. Number of kernel launches
2. Activation I/O overhead
3. Fusion opportunities

KEY INSIGHT:
------------
For M=1 dense layers, the difference between Marlin and CUTLASS FP4 is SMALL
because both are memory-bound (reading 7.83 MB of weights).

The ~2x performance gap vs llama.cpp comes from:
1. Framework overhead (Python vs C++)
2. Kernel count (900 vs 400)
3. Fusion (GEMV+activation, top-k+softmax)
4. Activation reuse (quantize once, use 4x)

RECOMMENDATION:
---------------
1. Keep Marlin for now - it's highly optimized and works
2. Focus on reducing kernel count via CUDA graphs
3. Consider CUTLASS FP4 GEMM for prefill (M >> 1) where native FP4 helps
4. Long-term: Fuse operations like llama.cpp does

TO TEST CUTLASS FP4 GEMM vs MARLIN:
-----------------------------------
The FlashInfer CUTLASS FP4 GEMM is available at:
  flashinfer.gemm.get_gemm_sm120_module_cutlass_fp4()

But it requires FP8 activations, so the workflow would be:
1. Quantize BF16 activations → FP8 (extra kernel)
2. Run CUTLASS FP4 GEMM
3. Output is BF16

This adds a quantization step that Marlin doesn't need, likely making it slower
for small M unless the quantization is fused or amortized.
""")

# Let's check if we can test both
print("\n" + "=" * 70)
print("ACTUAL BENCHMARK COMPARISON")
print("=" * 70)

import torch
import time

def test_marlin_availability():
    """Check if Marlin is available and working."""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
        return True
    except ImportError:
        return False

def test_cutlass_fp4_availability():
    """Check if CUTLASS FP4 GEMM is available."""
    try:
        from flashinfer.gemm.gemm_base import get_gemm_sm120_module_cutlass_fp4
        module = get_gemm_sm120_module_cutlass_fp4()
        return True
    except Exception as e:
        print(f"CUTLASS FP4 not available: {e}")
        return False

print(f"\nMarlin available: {test_marlin_availability()}")
print(f"CUTLASS FP4 available: {test_cutlass_fp4_availability()}")

# Benchmark if both available
if test_marlin_availability() and test_cutlass_fp4_availability():
    print("\nBoth available - running benchmarks...")
    
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )
    from flashinfer.gemm.gemm_base import get_gemm_sm120_module_cutlass_fp4
    from flashinfer import mxfp8_quantize
    
    # Test dimensions (QKV projection)
    M = 1
    K = 2880
    N = 5120
    
    # Create test inputs
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(1, 255, (N, K // 32), dtype=torch.uint8, device="cuda")
    
    warmup = 20
    iters = 100
    
    # Benchmark Marlin
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight_fp4.clone(), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(weight_scale.clone(), requires_grad=False)
            self.params_dtype = torch.bfloat16
            self.input_size_per_partition = K
            self.output_size_per_partition = N
    
    layer = MockLayer().cuda()
    prepare_fp4_layer_for_marlin(layer, input_dtype=torch.bfloat16)
    
    # Warmup Marlin
    for _ in range(warmup):
        out_marlin = apply_fp4_marlin_linear(
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
    
    start = time.perf_counter()
    for _ in range(iters):
        out_marlin = apply_fp4_marlin_linear(
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
    marlin_time = (time.perf_counter() - start) / iters * 1000
    
    print(f"\nMarlin (M={M}, K={K}, N={N}): {marlin_time:.4f} ms")
    
    # Benchmark CUTLASS FP4 (includes quantization)
    cutlass_module = get_gemm_sm120_module_cutlass_fp4()
    runner = cutlass_module.cutlass_fp4_gemm_runner()
    
    # CUTLASS needs FP8 activations - this is the overhead
    # For fair comparison, we include quantization time
    
    # Prepare tensors for CUTLASS
    # Note: CUTLASS FP4 GEMM expects different tensor layouts
    try:
        # Quantize input to FP8
        input_fp8, input_scale = mxfp8_quantize(input_bf16)
        
        output_bf16 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        alpha = torch.ones(1, dtype=torch.float32, device="cuda")
        workspace = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # 32MB workspace
        
        # Warmup CUTLASS
        for _ in range(warmup):
            input_fp8, input_scale = mxfp8_quantize(input_bf16)
            runner.forward([
                input_fp8, weight_fp4, input_scale, weight_scale,
                alpha, None, output_bf16, None, None, workspace
            ])
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            input_fp8, input_scale = mxfp8_quantize(input_bf16)
            runner.forward([
                input_fp8, weight_fp4, input_scale, weight_scale,
                alpha, None, output_bf16, None, None, workspace
            ])
        torch.cuda.synchronize()
        cutlass_time = (time.perf_counter() - start) / iters * 1000
        
        print(f"CUTLASS FP4 (incl. quant): {cutlass_time:.4f} ms")
        print(f"Ratio (CUTLASS/Marlin): {cutlass_time/marlin_time:.2f}x")
        
    except Exception as e:
        print(f"CUTLASS FP4 benchmark failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\nCannot run benchmarks - missing dependencies")
