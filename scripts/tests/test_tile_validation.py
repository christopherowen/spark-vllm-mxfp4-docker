#!/usr/bin/env python3
"""Standalone tile validation script for SM120/SM121 MoE GEMM.

This script tests each tile configuration in isolation (outside vLLM) to identify
which tiles work and which crash. This helps isolate MoE kernel issues from
vLLM's compilation/warmup phase.

Run inside the dev container:
    CUDA_LAUNCH_BLOCKING=1 python3 /workspace/scripts/tests/test_tile_validation.py

Or with compute-sanitizer for detailed crash info:
    CUDA_LAUNCH_BLOCKING=1 compute-sanitizer python3 /workspace/scripts/tests/test_tile_validation.py
"""
import os
import sys
import time
import argparse
from typing import Optional

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


# All tiles from SM120_SUPPORTED_TILE_MN in core.py
ALL_TILES = [
    # Native tiles (M >= 64, no swap_ab)
    (64, 16), (64, 32), (64, 64), (64, 128),
    (128, 16), (128, 32), (128, 64), (128, 128),
    (256, 16),
    # Swapped tiles (M < 64, uses swap_ab)
    (16, 64), (32, 64),
    (16, 128), (32, 128),
    (16, 256), (32, 256),
    (16, 512),
]

# Priority tiles for quick testing
PRIORITY_TILES = [
    (128, 128),  # Known good baseline
    (64, 128),   # Currently failing
    (128, 64),   # Alternative
    (32, 128),   # Swap tile
]


def test_tile(
    logical_m: int,
    logical_n: int,
    num_tokens: int = 64,
    hidden_size: int = 256,
    inter_size: int = 256,
    verbose: bool = False,
) -> tuple[bool, Optional[str]]:
    """Test a specific tile configuration.
    
    Args:
        logical_m: Logical M dimension of the tile
        logical_n: Logical N dimension of the tile
        num_tokens: Number of tokens to test with (affects batch size)
        hidden_size: Model hidden size (must be aligned)
        inter_size: Intermediate size (must be aligned)
        verbose: Print detailed output
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    device = "cuda"
    swap_ab = logical_m < 64
    
    if verbose:
        print(f"  num_tokens={num_tokens}, hidden_size={hidden_size}, inter_size={inter_size}")
        print(f"  swap_ab={swap_ab}")
    
    try:
        # Create input tensors
        num_experts = 1
        topk = 1
        
        x_fp8 = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
            torch.float8_e4m3fn
        )
        
        token_selected_experts = torch.zeros(
            (num_tokens, topk), device=device, dtype=torch.int32
        )
        token_final_scales = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)
        
        # Weight tensors (FP4 packed as int64)
        fc1_expert_weights = torch.zeros(
            (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
        )
        fc2_expert_weights = torch.zeros(
            (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
        )
        
        # Scale factors for MXFP4
        FP8_PER_INT32 = 4
        SFVEC = 32
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
        
        tile_mn = (logical_m, logical_n)
        
        if verbose:
            print(f"  Getting module for tile {tile_mn}...")
        
        mod = get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=tile_mn)
        
        # Prepare output tensor
        out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
        
        if verbose:
            print("  Calling kernel...")
        
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
        
        out_y = y[0] if isinstance(y, (list, tuple)) else y
        
        if verbose:
            print(f"  Output shape: {out_y.shape}")
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def run_validation(
    tiles: list[tuple[int, int]],
    test_batch_sizes: list[int],
    verbose: bool = False,
) -> dict[tuple[int, int], dict]:
    """Run validation for all specified tiles.
    
    Args:
        tiles: List of (M, N) tile configurations to test
        test_batch_sizes: List of batch sizes to test for each tile
        verbose: Print detailed output
        
    Returns:
        Dictionary mapping tile -> {status, errors, timings}
    """
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Testing {len(tiles)} tile configurations")
    print(f"Batch sizes to test: {test_batch_sizes}")
    print(f"{'='*70}\n")
    
    for tile_m, tile_n in tiles:
        swap_ab = tile_m < 64
        tile_key = (tile_m, tile_n)
        
        print(f"Tile ({tile_m:3d}, {tile_n:3d}) [{'swap' if swap_ab else 'native':6s}]: ", end="", flush=True)
        
        tile_results = {"passes": 0, "fails": 0, "errors": [], "compile_time": 0}
        
        # Test with different batch sizes
        for batch_size in test_batch_sizes:
            start = time.time()
            success, error = test_tile(
                tile_m, tile_n,
                num_tokens=batch_size,
                verbose=verbose,
            )
            elapsed = time.time() - start
            
            if success:
                tile_results["passes"] += 1
                if tile_results["compile_time"] == 0:
                    tile_results["compile_time"] = elapsed
            else:
                tile_results["fails"] += 1
                tile_results["errors"].append((batch_size, error))
        
        # Determine overall status
        if tile_results["fails"] == 0:
            status = "PASS"
            status_char = "✓"
        elif tile_results["passes"] > 0:
            status = "PARTIAL"
            status_char = "~"
        else:
            status = "FAIL"
            status_char = "✗"
        
        tile_results["status"] = status
        results[tile_key] = tile_results
        
        print(f"{status_char} {status:7s} (passes={tile_results['passes']}, fails={tile_results['fails']})")
        
        if tile_results["errors"] and verbose:
            for batch_size, error in tile_results["errors"][:2]:
                print(f"    batch={batch_size}: {error[:100]}...")
    
    return results


def print_summary(results: dict[tuple[int, int], dict]):
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = [k for k, v in results.items() if v["status"] == "PASS"]
    failed = [k for k, v in results.items() if v["status"] == "FAIL"]
    partial = [k for k, v in results.items() if v["status"] == "PARTIAL"]
    
    print(f"\nPassed ({len(passed)}):")
    for m, n in sorted(passed):
        swap = "swap" if m < 64 else "native"
        print(f"  ({m:3d}, {n:3d}) [{swap}]")
    
    if partial:
        print(f"\nPartial ({len(partial)}):")
        for m, n in sorted(partial):
            swap = "swap" if m < 64 else "native"
            print(f"  ({m:3d}, {n:3d}) [{swap}]")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for m, n in sorted(failed):
            swap = "swap" if m < 64 else "native"
            errors = results[(m, n)]["errors"]
            first_error = errors[0][1][:60] if errors else "unknown"
            print(f"  ({m:3d}, {n:3d}) [{swap}]: {first_error}...")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(passed)} passed, {len(partial)} partial, {len(failed)} failed")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate SM120/SM121 MoE tiles")
    parser.add_argument("--tiles", nargs="+", help="Specific tiles to test (e.g., 64x128 128x128)")
    parser.add_argument("--priority", action="store_true", help="Test only priority tiles")
    parser.add_argument("--all", action="store_true", help="Test all tiles")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 64, 128],
                        help="Batch sizes to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SM120/SM121 MoE Tile Validation")
    print("=" * 70)
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    else:
        print("ERROR: CUDA not available")
        return 1
    
    # Determine which tiles to test
    if args.tiles:
        tiles = []
        for t in args.tiles:
            m, n = map(int, t.lower().split("x"))
            tiles.append((m, n))
    elif args.priority:
        tiles = PRIORITY_TILES
    elif args.all:
        tiles = ALL_TILES
    else:
        # Default: priority tiles
        tiles = PRIORITY_TILES
    
    # Run validation
    results = run_validation(tiles, args.batch_sizes, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on failures
    failed = sum(1 for v in results.values() if v["status"] == "FAIL")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
