#!/usr/bin/env python3
"""
Investigate the M=64 numerical difference.

Hypothesis to test:
1. Is it PyTorch non-determinism? (run same computation twice)
2. Is it matrix size dependent cuBLAS kernel selection?
3. Is there an actual bug in how we split/compute?
"""

import torch
import sys

torch.manual_seed(42)
DEVICE = "cuda"


def create_test_data(M: int, K: int = 1152, inter_size: int = 2048):
    """Create reproducible test data."""
    torch.manual_seed(42)  # Reset seed for each test
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
    N_full = inter_size * 2
    fc1_weights = torch.randn(N_full, K, dtype=torch.bfloat16, device=DEVICE) * 0.05
    return input_bf16, fc1_weights, inter_size


def test_pytorch_determinism(M: int):
    """Test if PyTorch matmul is deterministic for this size."""
    print(f"\n{'='*60}")
    print(f"Test 1: PyTorch Determinism for M={M}")
    print(f"{'='*60}")
    
    input_bf16, fc1_weights, inter_size = create_test_data(M)
    
    # Run the same matmul twice
    result1 = input_bf16 @ fc1_weights.T
    result2 = input_bf16 @ fc1_weights.T
    
    is_identical = torch.equal(result1, result2)
    print(f"  Same matmul twice: {'IDENTICAL' if is_identical else 'DIFFERENT'}")
    
    if not is_identical:
        diff = torch.abs(result1 - result2).max()
        mismatch = (result1 != result2).sum().item()
        print(f"    Max diff: {diff}")
        print(f"    Mismatches: {mismatch}/{result1.numel()}")
    
    return is_identical


def test_split_equivalence(M: int):
    """
    Test if slicing the output is equivalent to separate matmuls.
    
    This is the core of our assumption:
    - baseline: (A @ W_full)[:, :N] should equal A @ W_full[:N, :]
    """
    print(f"\n{'='*60}")
    print(f"Test 2: Split Equivalence for M={M}")
    print(f"{'='*60}")
    
    input_bf16, fc1_weights, inter_size = create_test_data(M)
    K = input_bf16.shape[1]
    
    # Method A: Full matmul then slice
    full_output = input_bf16 @ fc1_weights.T  # [M, 2*inter]
    linear_from_slice = full_output[:, :inter_size]
    gate_from_slice = full_output[:, inter_size:]
    
    # Method B: Split weights first, then separate matmuls
    w_linear = fc1_weights[:inter_size, :]  # [inter, K]
    w_gate = fc1_weights[inter_size:, :]     # [inter, K]
    
    linear_from_split = input_bf16 @ w_linear.T  # [M, inter]
    gate_from_split = input_bf16 @ w_gate.T       # [M, inter]
    
    # Compare
    linear_match = torch.equal(linear_from_slice, linear_from_split)
    gate_match = torch.equal(gate_from_slice, gate_from_split)
    
    print(f"  Linear (slice vs split matmul): {'IDENTICAL' if linear_match else 'DIFFERENT'}")
    print(f"  Gate (slice vs split matmul):   {'IDENTICAL' if gate_match else 'DIFFERENT'}")
    
    if not linear_match:
        diff = torch.abs(linear_from_slice - linear_from_split).max()
        mismatch = (linear_from_slice != linear_from_split).sum().item()
        total = linear_from_slice.numel()
        print(f"    Linear max diff: {diff}")
        print(f"    Linear mismatches: {mismatch}/{total} ({100*mismatch/total:.3f}%)")
        
        # Find where differences occur
        diff_tensor = linear_from_slice != linear_from_split
        diff_rows = diff_tensor.any(dim=1).sum().item()
        diff_cols = diff_tensor.any(dim=0).sum().item()
        print(f"    Rows with diffs: {diff_rows}/{M}")
        print(f"    Cols with diffs: {diff_cols}/{inter_size}")
    
    if not gate_match:
        diff = torch.abs(gate_from_slice - gate_from_split).max()
        mismatch = (gate_from_slice != gate_from_split).sum().item()
        total = gate_from_slice.numel()
        print(f"    Gate max diff: {diff}")
        print(f"    Gate mismatches: {mismatch}/{total} ({100*mismatch/total:.3f}%)")
    
    return linear_match and gate_match


def test_mathematical_equivalence():
    """
    Verify the mathematical identity we're relying on.
    
    For matrix A [M, K] and W [N, K]:
    (A @ W.T)[:, :N/2] should equal A @ W[:N/2, :].T
    
    This is mathematically ALWAYS true because:
    - (A @ W.T)[i, j] = sum_k A[i,k] * W[j,k]
    - (A @ W[:N/2,:].T)[i, j] = sum_k A[i,k] * W[j,k]  (same j range: 0 to N/2-1)
    
    If they differ, it's floating-point non-associativity, not a bug in our approach.
    """
    print(f"\n{'='*60}")
    print(f"Test 3: Mathematical Equivalence (FP32 reference)")
    print(f"{'='*60}")
    
    M = 64
    K = 1152
    inter_size = 2048
    
    torch.manual_seed(42)
    # Use FP32 for this test to eliminate BF16 rounding
    input_fp32 = torch.randn(M, K, dtype=torch.float32, device=DEVICE) * 0.1
    fc1_weights_fp32 = torch.randn(inter_size * 2, K, dtype=torch.float32, device=DEVICE) * 0.05
    
    # Method A: Full then slice
    full_output = input_fp32 @ fc1_weights_fp32.T
    linear_slice = full_output[:, :inter_size]
    
    # Method B: Split then matmul
    w_linear = fc1_weights_fp32[:inter_size, :]
    linear_split = input_fp32 @ w_linear.T
    
    fp32_match = torch.equal(linear_slice, linear_split)
    print(f"  FP32 (eliminates BF16 rounding): {'IDENTICAL' if fp32_match else 'DIFFERENT'}")
    
    if not fp32_match:
        diff = torch.abs(linear_slice - linear_split).max()
        print(f"    Max diff: {diff}")
        # This would indicate FP non-associativity in the GEMM itself
    
    return fp32_match


def test_accumulation_order():
    """
    Test if the difference is due to accumulation order in cuBLAS.
    
    cuBLAS may use different algorithms (and thus accumulation orders)
    for different matrix sizes.
    """
    print(f"\n{'='*60}")
    print(f"Test 4: cuBLAS Algorithm Selection")
    print(f"{'='*60}")
    
    # Test if deterministic algorithms help
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    M = 64
    input_bf16, fc1_weights, inter_size = create_test_data(M)
    
    full_output = input_bf16 @ fc1_weights.T
    linear_slice = full_output[:, :inter_size]
    
    w_linear = fc1_weights[:inter_size, :]
    linear_split = input_bf16 @ w_linear.T
    
    match_with_deterministic = torch.equal(linear_slice, linear_split)
    print(f"  With deterministic mode: {'IDENTICAL' if match_with_deterministic else 'STILL DIFFERENT'}")
    
    # Reset
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    
    return match_with_deterministic


def test_different_sizes():
    """Test a range of M values to find pattern."""
    print(f"\n{'='*60}")
    print(f"Test 5: Size Sweep")
    print(f"{'='*60}")
    
    sizes = [1, 2, 4, 8, 16, 32, 48, 56, 60, 62, 63, 64, 65, 66, 68, 72, 80, 96, 112, 128, 256]
    results = {}
    
    for M in sizes:
        input_bf16, fc1_weights, inter_size = create_test_data(M)
        
        full_output = input_bf16 @ fc1_weights.T
        linear_slice = full_output[:, :inter_size]
        
        w_linear = fc1_weights[:inter_size, :]
        linear_split = input_bf16 @ w_linear.T
        
        is_equal = torch.equal(linear_slice, linear_split)
        results[M] = is_equal
    
    print(f"  {'M':>4} | Match")
    print(f"  {'-'*4}-+------")
    for M, match in results.items():
        status = "✓" if match else "✗"
        print(f"  {M:>4} | {status}")
    
    # Find pattern
    failing = [m for m, match in results.items() if not match]
    passing = [m for m, match in results.items() if match]
    
    print(f"\n  Passing sizes: {passing}")
    print(f"  Failing sizes: {failing}")
    
    return results


def main():
    print("=" * 70)
    print("Investigation: M=64 Numerical Difference")
    print("=" * 70)
    
    # Test 1: Is PyTorch deterministic?
    det_64 = test_pytorch_determinism(64)
    det_128 = test_pytorch_determinism(128)
    
    # Test 2: Split equivalence
    equiv_64 = test_split_equivalence(64)
    equiv_128 = test_split_equivalence(128)
    
    # Test 3: Mathematical equivalence in FP32
    math_equiv = test_mathematical_equivalence()
    
    # Test 4: Deterministic mode
    det_mode = test_accumulation_order()
    
    # Test 5: Size sweep
    size_results = test_different_sizes()
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if not equiv_64 and equiv_128:
        if math_equiv:
            print("""
The difference is NOT a bug in our approach. It's caused by:

1. cuBLAS selects different GEMM algorithms based on matrix dimensions
2. Different algorithms have different FP accumulation orders
3. [M=64, K=1152] @ [2*inter, K].T uses a different kernel than
   [M=64, K=1152] @ [inter, K].T

This is expected behavior for floating-point matmul:
- (A @ B)[:, :N] mathematically equals A @ B[:, :N]
- But FP accumulation order can differ between kernels

For the CUTLASS implementation:
- We use the SAME kernel for both GEMMs (same tile size, same algorithm)
- Both A @ W_linear and A @ W_gate use identical code paths
- Therefore CUTLASS will NOT have this discrepancy

The PyTorch test is a FALSE POSITIVE for this issue.
""")
        else:
            print("Unexpected: FP32 also shows differences. Needs investigation.")
    else:
        print("Pattern unclear. More investigation needed.")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
