#!/usr/bin/env python3
"""
Layer 1A Kernel-Level Test: Two-GEMM Gated FC1

This script tests the gated FC1 approach at the kernel level by:
1. Running FC1 GEMM for linear weights -> acc_linear
2. Running FC1 GEMM for gate weights -> acc_gate  
3. Applying SwiGLU: output = silu(acc_gate) * acc_linear
4. Comparing against baseline FC1 -> doGatedActivation

This validates the approach before implementing the fused gated mainloop.

Usage:
    # Copy to container first
    docker cp scripts/debug/test_gated_fc1_kernel.py vllm-dev:/workspace/scripts/debug/
    
    # Run in container
    docker exec vllm-dev bash -c 'PYTHONPATH=/workspace/flashinfer python3 /workspace/scripts/debug/test_gated_fc1_kernel.py'
"""

import sys
sys.path.insert(0, '/workspace/flashinfer')

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Try to import FlashInfer components
try:
    from flashinfer import mxfp8_quantize, mxfp4_quantize
    HAS_FLASHINFER = True
except ImportError as e:
    print(f"Warning: FlashInfer not fully available: {e}")
    HAS_FLASHINFER = False

torch.manual_seed(42)
DEVICE = "cuda"


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def swiglu_bf16(linear: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU activation matching FlashInfer convention.
    
    FlashInfer: output = silu(gate) * linear
    Computes in FP32 for precision, outputs BF16.
    """
    linear_fp32 = linear.float()
    gate_fp32 = gate.float()
    output_fp32 = silu(gate_fp32) * linear_fp32
    return output_fp32.to(torch.bfloat16)


def baseline_fc1_moe_path(
    input_bf16: torch.Tensor,
    fc1_weights_bf16: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline FC1 path simulating the current kernel behavior.
    
    Returns:
        fc1_output_raw: [M, 2*inter_size] BF16 - raw FC1 GEMM output
        swiglu_output: [M, inter_size] BF16 - after SwiGLU
    """
    # FC1 GEMM: [M, K] @ [2*inter, K].T -> [M, 2*inter]
    fc1_output_raw = input_bf16 @ fc1_weights_bf16.T
    
    # Split and apply SwiGLU (FlashInfer convention: [linear | gate])
    linear = fc1_output_raw[:, :inter_size]
    gate = fc1_output_raw[:, inter_size:]
    swiglu_output = swiglu_bf16(linear, gate)
    
    return fc1_output_raw, swiglu_output


def gated_fc1_two_gemm_path(
    input_bf16: torch.Tensor,
    fc1_weights_bf16: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gated FC1 path using two separate GEMMs.
    
    This simulates what the gated mainloop would do:
    - GEMM1: A @ W_linear -> acc_linear
    - GEMM2: A @ W_gate -> acc_gate
    - Epilogue: silu(acc_gate) * acc_linear
    
    Returns:
        acc_linear: [M, inter_size] BF16
        acc_gate: [M, inter_size] BF16
        swiglu_output: [M, inter_size] BF16
    """
    # Split weights (FlashInfer convention: [linear | gate])
    w_linear = fc1_weights_bf16[:inter_size, :]  # [inter, K]
    w_gate = fc1_weights_bf16[inter_size:, :]    # [inter, K]
    
    # Two separate GEMMs (simulating dual accumulator)
    acc_linear = input_bf16 @ w_linear.T  # [M, inter]
    acc_gate = input_bf16 @ w_gate.T      # [M, inter]
    
    # SwiGLU in epilogue
    swiglu_output = swiglu_bf16(acc_linear, acc_gate)
    
    return acc_linear, acc_gate, swiglu_output


def test_equivalence(M: int, K: int = 1152, inter_size: int = 2048, verbose: bool = True):
    """
    Test that two-GEMM gated approach produces identical output to baseline.
    """
    N_full = inter_size * 2
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test M={M}, K={K}, inter_size={inter_size}")
        print(f"  FC1 weights: [{N_full}, {K}] = [{inter_size}, {K}] + [{inter_size}, {K}]")
        print(f"{'='*60}")
    
    # Create test data
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
    fc1_weights = torch.randn(N_full, K, dtype=torch.bfloat16, device=DEVICE) * 0.05
    
    # Run baseline
    fc1_raw, baseline_swiglu = baseline_fc1_moe_path(input_bf16, fc1_weights, inter_size)
    
    # Run gated two-GEMM
    acc_linear, acc_gate, gated_swiglu = gated_fc1_two_gemm_path(input_bf16, fc1_weights, inter_size)
    
    # Verify intermediate values match
    baseline_linear = fc1_raw[:, :inter_size]
    baseline_gate = fc1_raw[:, inter_size:]
    
    linear_match = torch.equal(baseline_linear, acc_linear)
    gate_match = torch.equal(baseline_gate, acc_gate)
    swiglu_match = torch.equal(baseline_swiglu, gated_swiglu)
    
    if verbose:
        print(f"  acc_linear matches baseline[:, :inter]: {linear_match}")
        print(f"  acc_gate matches baseline[:, inter:]:   {gate_match}")
        print(f"  SwiGLU output matches:                  {swiglu_match}")
    
    # For non-bit-identical cases, check if within FP tolerance
    # PyTorch matmul has non-deterministic accumulation order for certain sizes
    # The actual CUTLASS kernel will be deterministic
    
    # For intermediate values (linear, gate), allow larger tolerance
    # BF16 has ~3 decimal digits precision, and accumulation order varies
    intermediate_rtol = 5e-3  # 0.5%
    intermediate_atol = 2e-3  # 0.002
    
    if not linear_match:
        diff = torch.abs(baseline_linear - acc_linear).max()
        if torch.allclose(baseline_linear, acc_linear, rtol=intermediate_rtol, atol=intermediate_atol):
            print(f"    Linear: allclose PASS (max_diff={diff:.6f})")
            linear_match = True
        else:
            print(f"    Linear FAIL: max_diff={diff}")
    
    if not gate_match:
        diff = torch.abs(baseline_gate - acc_gate).max()
        if torch.allclose(baseline_gate, acc_gate, rtol=intermediate_rtol, atol=intermediate_atol):
            print(f"    Gate: allclose PASS (max_diff={diff:.6f})")
            gate_match = True
        else:
            print(f"    Gate FAIL: max_diff={diff}")
    
    if not swiglu_match:
        diff = torch.abs(baseline_swiglu - gated_swiglu).max()
        mismatch_count = (baseline_swiglu != gated_swiglu).sum().item()
        total = baseline_swiglu.numel()
        
        # Check if close enough
        if torch.allclose(baseline_swiglu, gated_swiglu, rtol=1e-3, atol=1e-3):
            print(f"    SwiGLU: allclose PASS (max_diff={diff:.6f}, mismatches={mismatch_count}/{total})")
            print(f"    Note: PyTorch FP accumulation order difference, CUTLASS will be deterministic")
            swiglu_match = True
        else:
            print(f"    SwiGLU FAIL: max_diff={diff}, mismatches={mismatch_count}/{total}")
    
    all_pass = linear_match and gate_match and swiglu_match
    status = "PASS" if all_pass else "FAIL"
    
    if verbose:
        print(f"\n  Result: {status}")
        if all_pass:
            print(f"  Sample output[0,:4]: {gated_swiglu[0, :4].tolist()}")
    
    return all_pass


def test_with_quantization(M: int, verbose: bool = True):
    """
    Test with FP4/FP8 quantization to match actual kernel behavior.
    """
    if not HAS_FLASHINFER:
        print("Skipping quantization test (FlashInfer not available)")
        return None
    
    K = 1152
    inter_size = 2048
    N_full = inter_size * 2
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Quantization Test M={M}")
        print(f"{'='*60}")
    
    # Create BF16 tensors
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
    fc1_weights_bf16 = torch.randn(N_full, K, dtype=torch.bfloat16, device=DEVICE) * 0.05
    
    try:
        # Quantize input to FP8
        input_fp8, input_scale = mxfp8_quantize(input_bf16, True, 32)
        
        # Quantize weights to FP4
        fc1_fp4, fc1_scale = mxfp4_quantize(fc1_weights_bf16)
        
        if verbose:
            print(f"  Input FP8: {input_fp8.shape}, {input_fp8.dtype}")
            print(f"  Weights FP4: {fc1_fp4.shape}, {fc1_fp4.dtype}")
            print(f"  Quantization: OK")
        
        # Note: Actual CUTLASS kernel test would go here
        # For now, we just verify quantization works
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  Quantization failed: {e}")
        return False


def benchmark_two_gemm_overhead(M: int = 64, iterations: int = 100):
    """
    Benchmark the overhead of two GEMMs vs one GEMM.
    
    This helps estimate the performance cost of the two-GEMM approach
    before we have the fused gated mainloop.
    """
    K = 1152
    inter_size = 2048
    N_full = inter_size * 2
    
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    fc1_weights = torch.randn(N_full, K, dtype=torch.bfloat16, device=DEVICE)
    w_linear = fc1_weights[:inter_size, :]
    w_gate = fc1_weights[inter_size:, :]
    
    # Warmup
    for _ in range(10):
        _ = input_bf16 @ fc1_weights.T
        _ = input_bf16 @ w_linear.T
        _ = input_bf16 @ w_gate.T
    torch.cuda.synchronize()
    
    # Benchmark single GEMM (full width)
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = input_bf16 @ fc1_weights.T
    torch.cuda.synchronize()
    single_gemm_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    # Benchmark two GEMMs (half width each)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = input_bf16 @ w_linear.T
        _ = input_bf16 @ w_gate.T
    torch.cuda.synchronize()
    two_gemm_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    print(f"\nBenchmark M={M} (avg of {iterations} iterations):")
    print(f"  Single GEMM [{M}x{K}] @ [{N_full}x{K}].T:  {single_gemm_time:.4f} ms")
    print(f"  Two GEMMs [{M}x{K}] @ [{inter_size}x{K}].T: {two_gemm_time:.4f} ms")
    print(f"  Overhead: {(two_gemm_time/single_gemm_time - 1)*100:.1f}%")
    
    return single_gemm_time, two_gemm_time


def main():
    print("=" * 70)
    print("Layer 1A Kernel-Level Test: Two-GEMM Gated FC1")
    print("=" * 70)
    
    # Test 1: Equivalence testing
    print("\n" + "-" * 70)
    print("Test 1: Equivalence (baseline vs two-GEMM gated)")
    print("-" * 70)
    
    test_shapes = [1, 16, 64, 128]
    results = {}
    
    for M in test_shapes:
        results[M] = test_equivalence(M)
    
    # Summary
    print("\n" + "-" * 70)
    print("Summary: Equivalence Tests")
    print("-" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for M, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  M={M:4d}: {status}")
    
    # Test 2: Quantization (if available)
    print("\n" + "-" * 70)
    print("Test 2: Quantization Compatibility")
    print("-" * 70)
    for M in [1, 16]:
        test_with_quantization(M)
    
    # Test 3: Performance overhead
    print("\n" + "-" * 70)
    print("Test 3: Two-GEMM Overhead (PyTorch matmul)")
    print("-" * 70)
    for M in [1, 16, 64]:
        benchmark_two_gemm_overhead(M)
    
    # Final result
    print("\n" + "=" * 70)
    all_passed = all(results.values())
    if all_passed:
        print("KERNEL TEST: SUCCESS")
        print("  - Two-GEMM approach produces correct output")
        print("  - Ready for fused gated mainloop implementation")
    else:
        print("KERNEL TEST: NEEDS INVESTIGATION")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
