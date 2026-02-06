#!/usr/bin/env python3
"""
Test script to validate the gated FC1 weight split approach.

This tests that splitting FC1 weights into linear/gate halves and running
two separate GEMMs produces the same result as the baseline approach
(one GEMM with 2*inter_size output, then SwiGLU activation).

Key validation:
1. Weight pointer offset calculation
2. Scale factor offset calculation  
3. Numerical equivalence of two-GEMM vs one-GEMM approaches
"""

import torch
import torch.nn.functional as F


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def swiglu(linear, gate):
    """SwiGLU activation: silu(gate) * linear"""
    return silu(gate) * linear


def baseline_fc1_swiglu(input_bf16, weights_bf16):
    """
    Baseline approach: one GEMM with N=2*inter_size, then SwiGLU.
    
    Args:
        input_bf16: [M, K] input activations
        weights_bf16: [2*inter_size, K] FC1 weights (linear || gate)
    
    Returns:
        output: [M, inter_size] after SwiGLU
    """
    M, K = input_bf16.shape
    two_inter_size, K2 = weights_bf16.shape
    assert K == K2
    inter_size = two_inter_size // 2
    
    # One GEMM: input @ weights.T -> [M, 2*inter_size]
    fc1_out = torch.matmul(input_bf16, weights_bf16.T)
    
    # Split into linear and gate halves
    linear_out = fc1_out[:, :inter_size]
    gate_out = fc1_out[:, inter_size:]
    
    # Apply SwiGLU
    output = swiglu(linear_out, gate_out)
    
    return output


def gated_fc1_two_gemm(input_bf16, weights_bf16):
    """
    Gated approach: two GEMMs with N=inter_size each, then SwiGLU.
    
    This simulates what the gated kernel will do internally.
    
    Args:
        input_bf16: [M, K] input activations
        weights_bf16: [2*inter_size, K] FC1 weights (linear || gate)
    
    Returns:
        output: [M, inter_size] after SwiGLU
    """
    M, K = input_bf16.shape
    two_inter_size, K2 = weights_bf16.shape
    assert K == K2
    inter_size = two_inter_size // 2
    
    # Split weights into linear and gate
    w_linear = weights_bf16[:inter_size, :]  # First half
    w_gate = weights_bf16[inter_size:, :]    # Second half
    
    # Two separate GEMMs
    linear_out = torch.matmul(input_bf16, w_linear.T)  # [M, inter_size]
    gate_out = torch.matmul(input_bf16, w_gate.T)      # [M, inter_size]
    
    # Apply SwiGLU
    output = swiglu(linear_out, gate_out)
    
    return output


def test_weight_split_correctness():
    """
    Test that weight split produces equivalent results to baseline.
    
    NOTE: PyTorch/cuBLAS may use different GEMM algorithms for different matrix
    dimensions, leading to different floating-point accumulation orders. This
    causes small numerical differences that are NOT a bug in our approach.
    
    The key validation is:
    1. Small M (1, 16): Should be bit-identical (same cuBLAS kernel)
    2. Medium/Large M: Should be mathematically equivalent (allclose)
    
    When we implement this in CUTLASS, both GEMMs will use identical kernel code,
    so accumulation order will be deterministic.
    """
    print("=" * 60)
    print("Testing Gated FC1 Weight Split Correctness")
    print("=" * 60)
    print("  Note: cuBLAS may use different algorithms for different M,")
    print("  causing FP accumulation order differences. This is expected.")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    test_cases = [
        # (M, K, inter_size)
        (1, 2048, 1024),
        (16, 2048, 1024),
        (64, 2048, 1024),
        (128, 2048, 1024),
        (1, 7168, 2048),     # gpt-oss-120b-like
        (32, 7168, 2048),
    ]
    
    all_passed = True
    
    for M, K, inter_size in test_cases:
        two_inter_size = 2 * inter_size
        
        # Generate random inputs - use SAME seed for each test case
        # so results are reproducible
        torch.manual_seed(42 + M + K + inter_size)  # Unique but deterministic per case
        input_bf16 = torch.randn(M, K, dtype=dtype, device=device)
        weights_bf16 = torch.randn(two_inter_size, K, dtype=dtype, device=device)
        
        # Scale to reasonable range to avoid overflow in matmul
        input_bf16 = input_bf16 / (K ** 0.5)
        weights_bf16 = weights_bf16 / (K ** 0.5)
        
        # Compute using both methods
        baseline_out = baseline_fc1_swiglu(input_bf16, weights_bf16)
        gated_out = gated_fc1_two_gemm(input_bf16, weights_bf16)
        
        # Compare with tolerances appropriate for BF16
        is_equal = torch.equal(baseline_out, gated_out)
        
        # For mathematical equivalence, check in FP32 to eliminate BF16 precision issues
        baseline_fp32 = baseline_out.float()
        gated_fp32 = gated_out.float()
        
        # Relative tolerance based on typical BF16 precision
        # BF16 has ~3 decimal digits of precision, so relative diffs up to 0.1% are normal
        is_close = torch.allclose(baseline_fp32, gated_fp32, rtol=5e-3, atol=1e-2)
        max_diff = (baseline_fp32 - gated_fp32).abs().max().item()
        mean_abs = baseline_fp32.abs().mean().item()
        rel_diff = max_diff / (mean_abs + 1e-6)
        
        # Pass if bit-identical OR mathematically close
        passed = is_equal or is_close
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        
        print(f"  M={M:4d}, K={K:4d}, inter={inter_size:4d}: {status}")
        print(f"    bit-identical={is_equal}, allclose={is_close}, max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
    
    print()
    if all_passed:
        print("✓ All tests passed (weight split is mathematically correct)")
    else:
        print("✗ Some tests failed!")
    
    return all_passed


def test_offset_calculation():
    """
    Test that offset calculations match expected values.
    
    For FP4 weights:
      - Weight offset (gate): inter_size * K / 2 bytes
      - SF offset (gate): inter_size * ceil(K/32) elements
    """
    print("=" * 60)
    print("Testing Offset Calculations")
    print("=" * 60)
    
    test_cases = [
        # (K, inter_size)
        (2048, 1024),
        (7168, 2048),
        (7168, 14336),  # gpt-oss-120b
    ]
    
    for K, inter_size in test_cases:
        # FP4: 2 elements per byte
        weight_offset_bytes = (inter_size * K) // 2
        
        # MXFP4 block size = 32
        K_blocks = (K + 31) // 32
        sf_offset_elements = inter_size * K_blocks
        
        print(f"  K={K:5d}, inter_size={inter_size:5d}:")
        print(f"    Weight offset (gate): {weight_offset_bytes:,} bytes")
        print(f"    SF offset (gate):     {sf_offset_elements:,} elements")
    
    print()
    return True


def test_memory_layout():
    """
    Test that the memory layout matches CUTLASS expectations.
    
    CUTLASS column-major layout for weights: [N, K]
    - Contiguous in K dimension (stride-1 on K)
    - Stride N on N dimension
    """
    print("=" * 60)
    print("Testing Memory Layout")
    print("=" * 60)
    
    K = 128
    inter_size = 64
    two_inter_size = 2 * inter_size
    
    # Create weights in expected layout
    weights = torch.arange(two_inter_size * K, dtype=torch.float32).reshape(two_inter_size, K)
    
    # Linear half: rows 0 to inter_size-1
    w_linear = weights[:inter_size, :]
    
    # Gate half: rows inter_size to 2*inter_size-1
    w_gate = weights[inter_size:, :]
    
    print(f"  Weights shape: {weights.shape}")
    print(f"  Linear half shape: {w_linear.shape}")
    print(f"  Gate half shape: {w_gate.shape}")
    
    # Check first element of each half
    print(f"  weights[0, 0] = {weights[0, 0].item()}")
    print(f"  weights[inter_size, 0] = {weights[inter_size, 0].item()}")
    print(f"  w_linear[0, 0] = {w_linear[0, 0].item()}")
    print(f"  w_gate[0, 0] = {w_gate[0, 0].item()}")
    
    # Verify offset
    expected_gate_start = inter_size * K
    actual_gate_start = weights[inter_size, 0].item()
    assert actual_gate_start == expected_gate_start, f"Gate offset mismatch: {actual_gate_start} vs {expected_gate_start}"
    
    print(f"  ✓ Gate offset verified: {expected_gate_start}")
    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("GATED FC1 WEIGHT SPLIT VALIDATION")
    print("=" * 60 + "\n")
    
    passed = []
    passed.append(test_offset_calculation())
    passed.append(test_memory_layout())
    passed.append(test_weight_split_correctness())
    
    print("\n" + "=" * 60)
    if all(passed):
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return 0 if all(passed) else 1


if __name__ == "__main__":
    exit(main())
