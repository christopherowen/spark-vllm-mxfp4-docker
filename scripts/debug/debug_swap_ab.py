#!/usr/bin/env python3
"""Minimal repro for swap_ab illegal memory access in SM120 MoE.

Run with:
    cuda-gdb --args python3 debug_swap_ab.py
    (cuda-gdb) run
    (cuda-gdb) bt
"""
import os
import sys

# Set up paths for local FlashInfer
if os.path.exists("/workspace/flashinfer"):
    FLASHINFER_PATH = "/workspace/flashinfer"
    VLLM_PATH = "/workspace/vllm"
else:
    # Host fallback
    home = os.path.expanduser("~")
    FLASHINFER_PATH = f"{home}/projects/flashinfer"
    VLLM_PATH = f"{home}/projects/vllm"

os.environ["PYTHONPATH"] = f"{FLASHINFER_PATH}:{VLLM_PATH}"
sys.path.insert(0, FLASHINFER_PATH)
sys.path.insert(0, VLLM_PATH)

# Minimal build profile to speed up compilation
os.environ["FLASHINFER_FUSED_MOE_BUILD_PROFILE"] = "mxfp4_minimal"

import torch


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def test_swap_ab_tile(logical_m: int = 32, logical_n: int = 128):
    """Test a swap_ab tile configuration.
    
    swap_ab is triggered when logical_m < 64.
    This test uses minimal dimensions to reproduce the issue quickly.
    """
    print(f"\n{'='*70}")
    print(f"Testing swap_ab tile: logical_m={logical_m}, logical_n={logical_n}")
    print(f"swap_ab = {logical_m < 64}")
    print(f"{'='*70}\n")
    
    device = "cuda"
    
    # Minimal dimensions that satisfy alignment constraints
    # K must be >= 128 (CUTLASS alignment)
    # N (hidden_size) must be divisible by 128 for swap_ab
    num_tokens = logical_m  # Token count = logical M
    num_experts = 1
    topk = 1
    hidden_size = 256  # Small but aligned
    inter_size = 256   # Small but aligned
    
    print(f"num_tokens={num_tokens}, hidden_size={hidden_size}, inter_size={inter_size}")
    
    # Create input tensors
    # For MXFP4: activations are FP8, weights are FP4
    x_fp8 = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    
    token_selected_experts = torch.zeros(
        (num_tokens, topk), device=device, dtype=torch.int32
    )
    token_final_scales = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)
    
    # Weight tensors (FP4 packed as int64)
    # Shape: (num_experts, out_features, in_features // 16)
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )
    
    # Scale factors for MXFP4
    FP8_PER_INT32 = 4
    SFVEC = 32  # Scale factor vector size for MXFP4
    
    MinNDimAlignment = 128
    MinKDimAlignment = 128
    
    hs_aligned_k = align_to(hidden_size, MinKDimAlignment)
    hs_aligned_n = align_to(hidden_size, MinNDimAlignment)
    inter_aligned_n = align_to(inter_size, MinNDimAlignment)
    inter_aligned_k = align_to(inter_size, MinKDimAlignment)
    
    fc1_weight_block = torch.zeros(
        (num_experts, inter_aligned_n * 2, hs_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc2_weight_block = torch.zeros(
        (num_experts, hs_aligned_n, inter_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc1_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    fc2_act_global = torch.ones((), device=device, dtype=torch.float32)
    fc2_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    
    quant_scales = [fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]
    
    # Get the CUTLASS module
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    print(f"Backend: {backend}")
    
    tile_mn = (logical_m, logical_n)
    print(f"Requesting tile_mn = {tile_mn}")
    
    print("Getting module (may JIT compile)...")
    mod = get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=tile_mn)
    print("Module ready.")
    
    # Prepare output tensor
    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    
    print("Calling kernel...")
    try:
        y = mod.cutlass_fused_moe(
            out, x_fp8, token_selected_experts, token_final_scales,
            fc1_expert_weights, None, fc2_expert_weights, None,
            torch.bfloat16, quant_scales, None, None, None, None,
            1, 0, 1, 0, 1, 0,  # fc1/fc2 params
            use_packed_weights=False,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            tune_max_num_tokens=256,
            enable_pdl=False,
            activation_type=3,  # SwiGLU
        )
        torch.cuda.synchronize()
        # `cutlass_fused_moe` returns a list in non-min-latency mode:
        # [output, num_active_experts_per_node, experts_to_token_score, active_expert_global_ids]
        out_y = y[0] if isinstance(y, (list, tuple)) else y
        print(f"SUCCESS! Output shape: {out_y.shape}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("swap_ab Debug Script")
    print("=" * 70)
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    
    # Test cases that trigger swap_ab (logical_m < 64)
    test_cases = [
        # (logical_m, logical_n) - swap_ab tiles (logical_m < 64)
        (32, 128),  # Common decode batch size
        (16, 128),
        (32, 64),
        (16, 64),
    ]
    
    # Also test a non-swap case for comparison
    test_cases_non_swap = [
        (64, 128),  # No swap (logical_m >= 64)
        (128, 128), # No swap
    ]
    
    print("\n--- Testing swap_ab tiles (logical_m < 64) ---")
    for m, n in test_cases:
        try:
            success = test_swap_ab_tile(m, n)
            if not success:
                print(f"FAILED: ({m}, {n})")
                return 1
        except Exception as e:
            print(f"EXCEPTION for ({m}, {n}): {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n--- Testing non-swap tiles (logical_m >= 64) for comparison ---")
    for m, n in test_cases_non_swap:
        try:
            success = test_swap_ab_tile(m, n)
            if not success:
                print(f"FAILED: ({m}, {n})")
        except Exception as e:
            print(f"EXCEPTION for ({m}, {n}): {e}")
    
    print("\n" + "=" * 70)
    print("All tests completed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
