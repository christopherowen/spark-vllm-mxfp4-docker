#!/usr/bin/env python3
"""
Benchmark MoE tile configurations to evaluate swap_ab for decode.

Tests native (64, 128) vs swapped (32, 128) tiles for small batch sizes.
"""

import os
import torch
import time
import sys
import argparse
import subprocess


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def precise_benchmark(fn, warmup=20, iters=100):
    """Benchmark using CUDA events."""
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
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)])
    return times[len(times)//2]  # median in μs


def create_moe_tensors(num_tokens, num_experts=64, hidden_size=6144, inter_size=24576, topk=2, device="cuda"):
    """Create properly formatted tensors for MoE CUTLASS kernel."""

    # Deterministic inputs for fair tile-to-tile comparisons (each tile is run in
    # a separate process; using a fixed seed makes the workload identical).
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    # FP8 activations (required for MXFP4 path)
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    
    # Expert routing
    token_selected_experts = torch.empty((num_tokens, topk), device=device, dtype=torch.int32)
    for t in range(num_tokens):
        token_selected_experts[t, 0] = t % num_experts
        if topk > 1:
            token_selected_experts[t, 1] = (t + 1) % num_experts
        for k in range(2, topk):
            token_selected_experts[t, k] = (t + k) % num_experts

    # Keep scales simple/constant so we measure GEMM, not softmax noise.
    token_final_scales = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    
    # Packed FP4 weights
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )
    
    # Quant scales
    FP8_PER_INT32 = 4
    SFVEC = 32
    MinNDimAlignmentMXFPX = 128
    MinKDimAlignmentMXFPX = 128
    
    hs_aligned_k = align_to(hidden_size, MinKDimAlignmentMXFPX)
    hs_aligned_n = align_to(hidden_size, MinNDimAlignmentMXFPX)
    inter_aligned_n = align_to(inter_size, MinNDimAlignmentMXFPX)
    inter_aligned_k = align_to(inter_size, MinKDimAlignmentMXFPX)
    
    fc1_weight_block = torch.zeros(
        (num_experts, inter_aligned_n * 2, hs_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device, dtype=torch.int32,
    )
    fc2_weight_block = torch.zeros(
        (num_experts, hs_aligned_n, inter_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device, dtype=torch.int32,
    )
    fc1_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    fc2_act_global = torch.ones((), device=device, dtype=torch.float32)
    fc2_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    
    quant_scales = [fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]
    
    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    
    return {
        'out': out,
        'x': x,
        'token_selected_experts': token_selected_experts,
        'token_final_scales': token_final_scales,
        'fc1_expert_weights': fc1_expert_weights,
        'fc2_expert_weights': fc2_expert_weights,
        'quant_scales': quant_scales,
    }


def benchmark_tile(tile_mn, num_tokens, tensors, *, backend: str, warmup: int, iters: int):
    """Benchmark a specific tile configuration."""
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    try:
        mod = get_cutlass_fused_moe_module(
            backend=backend,
            use_fast_build=True,
            tile_mn=tile_mn,
        )
    except Exception as e:
        return None, f"BUILD_FAIL: {str(e)[:120]}"
    
    def run_moe():
        return mod.cutlass_fused_moe(
            tensors['out'],
            tensors['x'],
            tensors['token_selected_experts'],
            tensors['token_final_scales'],
            tensors['fc1_expert_weights'],
            None,
            tensors['fc2_expert_weights'],
            None,
            torch.bfloat16,
            tensors['quant_scales'],
            None,  # input_sf
            None,  # swiglu_alpha
            None,  # swiglu_beta
            None,  # swiglu_limit
            1, 0, 1, 0, 1, 0,  # tp/ep/cluster size and rank
            use_packed_weights=False,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            tune_max_num_tokens=256,
            enable_pdl=False,
            activation_type=3,  # Swiglu
        )
    
    # Run benchmark
    try:
        time_us = precise_benchmark(run_moe, warmup=warmup, iters=iters)
        return time_us, "OK"
    except Exception as e:
        # Keep more of the exception around; many failures have critical info
        # (e.g., which CUTLASS internal check failed) beyond 120 chars.
        return None, f"RUN_FAIL: {str(e)[:600]}"

def moe_flops_per_token(hidden_size: int, inter_size: int, topk: int) -> int:
    # Rough FLOPs for MoE MLP per token:
    # - FC1: (hidden_size x 2*inter_size) multiply-add => 2 * hidden_size * (2*inter_size)
    # - FC2: (inter_size x hidden_size) multiply-add   => 2 * inter_size * hidden_size
    # - Multiply by topk experts per token.
    return topk * (2 * hidden_size * (2 * inter_size) + 2 * inter_size * hidden_size)


def format_tput(num_tokens: int, time_us: float, hidden_size: int, inter_size: int, topk: int) -> tuple[float, float]:
    # returns (tokens_per_s, tflops)
    sec = time_us * 1e-6
    tps = num_tokens / sec
    flops = moe_flops_per_token(hidden_size, inter_size, topk) * num_tokens
    tflops = flops / sec / 1e12
    return tps, tflops


def main():
    # Keep JIT compilation surface small for iteration on SM120/121.
    os.environ.setdefault("FLASHINFER_FUSED_MOE_BUILD_PROFILE", "mxfp4_minimal")

    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=6144)
    ap.add_argument("--inter-size", type=int, default=24576)
    # gpt-oss-120b MoE typically uses 8 experts (top-k routing).
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--backend", type=str, default="121", help="FlashInfer backend string, e.g. 121")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--tiles",
        type=str,
        default="default",
        choices=("default", "all"),
        help="Which tile set to benchmark: 'default' (curated) or 'all' (SM120_SUPPORTED_TILE_MN).",
    )
    ap.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated num_tokens to benchmark (e.g. '1,2,4,8,16,32,64,128').",
    )
    ap.add_argument(
        "--single",
        nargs=3,
        type=int,
        metavar=("TILE_M", "TILE_N", "NUM_TOKENS"),
        help="Run a single (tile_mn, num_tokens) case in an isolated process.",
    )
    args = ap.parse_args()

    if args.single is not None:
        tile_m, tile_n, num_tokens = args.single
        tile_mn = (tile_m, tile_n)
        tensors = create_moe_tensors(
            num_tokens,
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            inter_size=args.inter_size,
            topk=args.topk,
            device="cuda",
        )
        time_us, msg = benchmark_tile(
            tile_mn,
            num_tokens,
            tensors,
            backend=args.backend,
            warmup=args.warmup,
            iters=args.iters,
        )
        if time_us is None:
            print(f"FAIL tile_mn={tile_mn} num_tokens={num_tokens} {msg}")
            raise SystemExit(2)
        tps, tflops = format_tput(num_tokens, time_us, args.hidden_size, args.inter_size, args.topk)
        print(f"OK tile_mn={tile_mn} num_tokens={num_tokens} time_us={time_us:.1f} "
              f"tps={tps:.6f} tflops={tflops:.6f}")
        return

    print("=" * 80)
    print("MoE Tile Configuration Benchmark: Native vs Swapped")
    print("=" * 80)
    print()
    print("Testing if swap_ab tiles are faster for decode (small num_tokens)")
    print()
    
    device = "cuda"
    
    # Tiles to compare
    if args.tiles == "all":
        from flashinfer.fused_moe.core import SM120_SUPPORTED_TILE_MN
        tiles = [(t, "SUPPORTED") for t in SM120_SUPPORTED_TILE_MN]
    else:
        tiles = [
            ((64, 128), "NATIVE - current default"),
            ((32, 128), "SWAPPED - decode"),
            ((16, 128), "SWAPPED - smaller M"),
            ((32, 64), "SWAPPED - smaller N"),
            ((16, 64), "SWAPPED - smaller M,N"),
            ((16, 256), "SWAPPED - larger N"),
        ]
    
    # Test different batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    
    print(f"{'num_tokens':<12}", end="")
    for tile, desc in tiles:
        print(f"{str(tile):<20}", end="")
    print()
    print("-" * 80)
    
    for num_tokens in batch_sizes:
        print(f"{num_tokens:<12}", end="")
        
        results = []
        for tile, desc in tiles:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--hidden-size",
                    str(args.hidden_size),
                    "--inter-size",
                    str(args.inter_size),
                    "--num-experts",
                    str(args.num_experts),
                    "--topk",
                    str(args.topk),
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--single",
                    str(tile[0]),
                    str(tile[1]),
                    str(num_tokens),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = (proc.stdout or "").strip()
            if proc.returncode == 0 and out.startswith("OK "):
                parts = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in out.split() if "=" in kv}
                time_us = float(parts["time_us"])
                results.append((tile, time_us))
                tps, _tflops = format_tput(
                    num_tokens, time_us, args.hidden_size, args.inter_size, args.topk
                )
                print(f"{time_us:>8.1f}μs {tps:>6.0f}t/s ", end="")
            else:
                label = "SKIP" if "SKIP" in out else "FAIL"
                print(f"{label:<4} {'':<15}", end="")
        print()
        
        # Show speedup
        if len(results) >= 2:
            baseline = results[0][1]  # Native tile
            print(f"{'speedup:':<12}", end="")
            for tile, time_us in results:
                speedup = baseline / time_us
                print(f"{speedup:>10.2f}x       ", end="")
            print()
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If swapped tiles are faster for small num_tokens, we should update
select_tile_mn_for_sm120() in flashinfer/fused_moe/core.py to use them.
""")


if __name__ == "__main__":
    main()
