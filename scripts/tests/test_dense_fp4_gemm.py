#!/usr/bin/env python3
"""
Test FlashInfer dense FP4 GEMM on SM121.

This script tests:
1. NVFP4 dense GEMM with CUTLASS backend (should work)
2. MXFP4 dense GEMM with cuDNN backend (should work)
3. MXFP4 dense GEMM with CUTLASS backend (gap to fill)

Run inside vllm-dev container:
    docker exec -it vllm-dev bash -c '
        export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
        python /workspace/mxfp4/scripts/tests/test_dense_fp4_gemm.py
    '
"""

import torch
import torch.nn.functional as F
import sys

def test_backend(m, n, k, backend, fp4_type, verbose=True):
    """Test a specific backend + fp4_type combination."""
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize, mxfp4_quantize
    from flashinfer.utils import get_compute_capability
    
    use_nvfp4 = fp4_type == "nvfp4"
    block_size = 16 if use_nvfp4 else 32
    
    # Create test data
    input_bf16 = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    weight_bf16 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    
    # Reference result
    reference = torch.mm(input_bf16, weight_bf16.T)
    
    # Quantize
    if use_nvfp4:
        global_sf_input = (448 * 6) / input_bf16.float().abs().nan_to_num().max()
        global_sf_weight = (448 * 6) / weight_bf16.float().abs().nan_to_num().max()
        
        input_fp4, input_sf = nvfp4_quantize(
            input_bf16, global_sf_input, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        weight_fp4, weight_sf = nvfp4_quantize(
            weight_bf16, global_sf_weight, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        alpha = 1.0 / (global_sf_input * global_sf_weight)
    else:
        # MXFP4
        input_fp4, input_sf = mxfp4_quantize(input_bf16)
        weight_fp4, weight_sf = mxfp4_quantize(weight_bf16)
        alpha = None  # MXFP4 scales are per-block, no global alpha
    
    # Output buffer
    output = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    
    try:
        mm_fp4(
            input_fp4,
            weight_fp4.T,
            input_sf,
            weight_sf.T,
            alpha,
            torch.bfloat16,
            output,
            block_size=block_size,
            use_8x4_sf_layout=False,
            backend=backend,
            use_nvfp4=use_nvfp4,
            skip_check=False,
        )
        
        # Check accuracy
        cos_sim = F.cosine_similarity(reference.reshape(-1), output.reshape(-1), dim=0)
        
        if verbose:
            print(f"  ✓ {backend:8s} + {fp4_type:6s}: cos_sim={cos_sim:.4f}")
        return True, cos_sim.item()
        
    except Exception as e:
        if verbose:
            print(f"  ✗ {backend:8s} + {fp4_type:6s}: {type(e).__name__}: {e}")
        return False, str(e)


def main():
    print("=" * 70)
    print("FlashInfer Dense FP4 GEMM Test on SM121")
    print("=" * 70)
    
    # Check compute capability
    from flashinfer.utils import get_compute_capability
    cc = get_compute_capability(torch.device("cuda"))
    print(f"\nCompute Capability: SM{cc[0]}{cc[1]}")
    
    # Test matrix sizes (typical for QKV projection)
    test_cases = [
        (1, 4096, 4096, "Decode M=1"),
        (32, 4096, 4096, "Small batch"),
        (128, 4096, 4096, "Medium batch"),
        (512, 4096, 4096, "Large batch"),
    ]
    
    backends = ["cutlass", "cudnn", "auto"]
    fp4_types = ["nvfp4", "mxfp4"]
    
    results = {}
    
    for m, n, k, desc in test_cases:
        print(f"\n{desc} [{m}×{k}] × [{k}×{n}]:")
        for backend in backends:
            for fp4_type in fp4_types:
                key = (backend, fp4_type)
                success, result = test_backend(m, n, k, backend, fp4_type)
                if key not in results:
                    results[key] = []
                results[key].append((success, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Backend + FP4 Type Support")
    print("=" * 70)
    
    for backend in backends:
        for fp4_type in fp4_types:
            key = (backend, fp4_type)
            successes = sum(1 for s, _ in results[key] if s)
            total = len(results[key])
            status = "✓ WORKS" if successes == total else f"✗ FAILS ({successes}/{total})"
            print(f"  {backend:8s} + {fp4_type:6s}: {status}")
    
    print("\n" + "=" * 70)
    print("Key Finding:")
    print("=" * 70)
    
    cutlass_mxfp4_works = all(s for s, _ in results.get(("cutlass", "mxfp4"), [(False, "")]))
    if cutlass_mxfp4_works:
        print("  CUTLASS + MXFP4 already works! No kernel changes needed.")
    else:
        print("  CUTLASS + MXFP4 does NOT work yet.")
        print("  This is the gap to fill for unified dense/MoE kernels.")


if __name__ == "__main__":
    main()
