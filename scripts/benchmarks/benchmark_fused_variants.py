#!/usr/bin/env python3
"""
Microbenchmark: {gated, non-gated} x {2x3, 3x3} kernel variants.

Terminology (from CUTLASS template params):
  2x3 = MainloopSm120...BlockScaled<2,3,...> → 128x128 tile (prefill)
  3x3 = MainloopSm120...BlockScaled<3,3,...> → 64x128 tile (decode)

Gated   = MainloopSm120...BlockScaledGated  (SwiGLU fused in GEMM1 mainloop)
Non-gated = MainloopSm120...BlockScaled     (separate doActivationKernel after GEMM1)

Tests all 4 variants with real M bucket sizes observed in production
(expert-routed token counts after topk expansion).

Usage (inside container):
    python scripts/benchmarks/benchmark_fused_variants.py
    python scripts/benchmarks/benchmark_fused_variants.py --m-values 48,88
    python scripts/benchmarks/benchmark_fused_variants.py --iters 200
"""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("FLASHINFER_FUSED_MOE_BUILD_PROFILE", "mxfp4_minimal")


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Each variant: (label, tile_mn, fuse_activation)
VARIANTS = [
    ("non-gated 2x3 (128x128)", (128, 128), False),
    ("non-gated 3x3  (64x128)", (64, 128),  False),
    ("gated 2x3     (128x128)", (128, 128), True),
    ("gated 3x3      (64x128)", (64, 128),  True),
]

# Real M buckets from production (expert-routed token counts):
#   48  - small decode batch (e.g. 3 tokens, topk=2, 8 experts → ~6/expert, 48 total)
#   88  - medium decode
#   2000 - short prefill
#   8144 - long prefill (near max_num_batched_tokens=8192 after topk)
DEFAULT_M_VALUES = [48, 88, 2000, 8144]


# ---------------------------------------------------------------------------
# Tensor setup (mirrors benchmark_moe_tiles.py)
# ---------------------------------------------------------------------------
def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def create_moe_tensors(
    num_tokens: int,
    num_experts: int = 8,
    hidden_size: int = 6144,
    inter_size: int = 24576,
    topk: int = 2,
    device: str = "cuda",
):
    """Create properly formatted tensors for MoE CUTLASS kernel."""
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # FP8 activations
    x = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.float16
    ).to(torch.float8_e4m3fn)

    # Expert routing (spread tokens across experts)
    token_selected_experts = torch.empty(
        (num_tokens, topk), device=device, dtype=torch.int32
    )
    for t in range(num_tokens):
        token_selected_experts[t, 0] = t % num_experts
        if topk > 1:
            token_selected_experts[t, 1] = (t + 1) % num_experts

    token_final_scales = torch.full(
        (num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32
    )

    # Packed FP4 weights
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16),
        device=device,
        dtype=torch.int64,
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16),
        device=device,
        dtype=torch.int64,
    )

    # Quant scales
    FP8_PER_INT32 = 4
    SFVEC = 32
    MinAlignMN = 128
    MinAlignK = 128

    hs_aligned_k = align_to(hidden_size, MinAlignK)
    hs_aligned_n = align_to(hidden_size, MinAlignMN)
    inter_aligned_n = align_to(inter_size, MinAlignMN)
    inter_aligned_k = align_to(inter_size, MinAlignK)

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

    quant_scales = [
        fc1_weight_block,
        fc1_global,
        fc2_act_global,
        fc2_weight_block,
        fc2_global,
    ]

    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    return {
        "out": out,
        "x": x,
        "token_selected_experts": token_selected_experts,
        "token_final_scales": token_final_scales,
        "fc1_expert_weights": fc1_expert_weights,
        "fc2_expert_weights": fc2_expert_weights,
        "quant_scales": quant_scales,
    }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def precise_benchmark(fn, warmup=20, iters=100):
    """Benchmark using CUDA events.  Returns median time in microseconds."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    # elapsed_time returns ms; convert to us
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    return median, p10, p90


def benchmark_single(
    tile_m: int,
    tile_n: int,
    fuse: int,
    num_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    inter_size: int,
    num_experts: int,
    topk: int,
):
    """Run one (variant, M) benchmark in the current process.

    Prints a machine-parseable result line:
        OK tile=MxN fuse=F M=<tokens> median_us=<> p10_us=<> p90_us=<>
    or
        FAIL tile=MxN fuse=F M=<tokens> <error>
    """
    import torch
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    tile_mn = (tile_m, tile_n)
    fuse_activation = bool(fuse)

    try:
        mod = get_cutlass_fused_moe_module(
            backend="121",
            tile_mn=tile_mn,
            fuse_activation=fuse_activation,
        )
    except Exception as e:
        print(
            f"FAIL tile={tile_m}x{tile_n} fuse={fuse} M={num_tokens} "
            f"BUILD: {str(e)[:200]}"
        )
        return

    tensors = create_moe_tensors(
        num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        inter_size=inter_size,
        topk=topk,
    )

    def run():
        return mod.cutlass_fused_moe(
            tensors["out"],
            tensors["x"],
            tensors["token_selected_experts"],
            tensors["token_final_scales"],
            tensors["fc1_expert_weights"],
            None,
            tensors["fc2_expert_weights"],
            None,
            torch.bfloat16,
            tensors["quant_scales"],
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
            fuse_activation=fuse_activation,
        )

    try:
        median, p10, p90 = precise_benchmark(run, warmup=warmup, iters=iters)
        print(
            f"OK tile={tile_m}x{tile_n} fuse={fuse} M={num_tokens} "
            f"median_us={median:.1f} p10_us={p10:.1f} p90_us={p90:.1f}"
        )
    except Exception as e:
        print(
            f"FAIL tile={tile_m}x{tile_n} fuse={fuse} M={num_tokens} "
            f"RUN: {str(e)[:400]}"
        )


# ---------------------------------------------------------------------------
# Orchestrator (runs each case in a subprocess for isolation)
# ---------------------------------------------------------------------------
def parse_result(line: str):
    """Parse an OK result line into a dict."""
    if not line.startswith("OK "):
        return None
    parts = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            parts[k] = v
    return parts


def run_orchestrator(args):
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]

    print("=" * 96)
    print("MoE Kernel Variant Microbenchmark: {gated,non-gated} x {2x3,3x3}")
    print("=" * 96)
    print(f"  Model dims:  hidden={args.hidden_size}  inter={args.inter_size}")
    print(f"  Experts={args.num_experts}  topk={args.topk}")
    print(f"  M values:    {m_values}")
    print(f"  Warmup={args.warmup}  Iters={args.iters}")
    print()

    # Compile all 4 modules first (sequential, shows JIT progress)
    print("--- JIT compilation (if needed) ---")
    for label, tile_mn, fuse in VARIANTS:
        tag = f"tile={tile_mn[0]}x{tile_mn[1]} fuse={int(fuse)}"
        sys.stdout.write(f"  {tag:30s} ... ")
        sys.stdout.flush()
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable, os.path.abspath(__file__),
                "--single", str(tile_mn[0]), str(tile_mn[1]), str(int(fuse)), "48",
                "--warmup", "2", "--iters", "2",
                "--hidden-size", str(args.hidden_size),
                "--inter-size", str(args.inter_size),
                "--num-experts", str(args.num_experts),
                "--topk", str(args.topk),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        dt = time.monotonic() - t0
        ok = proc.returncode == 0 and "OK " in (proc.stdout or "")
        status = "OK" if ok else "FAIL"
        print(f"{status} ({dt:.1f}s)")
        if not ok:
            # Show last few lines of output for diagnosis
            for line in (proc.stdout or "").strip().split("\n")[-5:]:
                print(f"    {line}")

    # Run benchmarks
    print()
    print("--- Benchmarks ---")
    print()

    # Collect results: results[m_val][variant_idx] = (median, p10, p90) or None
    results = {m: [None] * len(VARIANTS) for m in m_values}

    for mi, m_val in enumerate(m_values):
        for vi, (label, tile_mn, fuse) in enumerate(VARIANTS):
            sys.stdout.write(
                f"  M={m_val:<6d} {label:30s} ... "
            )
            sys.stdout.flush()
            proc = subprocess.run(
                [
                    sys.executable, os.path.abspath(__file__),
                    "--single",
                    str(tile_mn[0]), str(tile_mn[1]), str(int(fuse)), str(m_val),
                    "--warmup", str(args.warmup),
                    "--iters", str(args.iters),
                    "--hidden-size", str(args.hidden_size),
                    "--inter-size", str(args.inter_size),
                    "--num-experts", str(args.num_experts),
                    "--topk", str(args.topk),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = (proc.stdout or "").strip()
            parsed = None
            for line in out.split("\n"):
                parsed = parse_result(line.strip())
                if parsed:
                    break

            if parsed:
                med = float(parsed["median_us"])
                p10 = float(parsed["p10_us"])
                p90 = float(parsed["p90_us"])
                results[m_val][vi] = (med, p10, p90)
                print(f"{med:>10.1f} us  (p10={p10:.1f}  p90={p90:.1f})")
            else:
                tag = "FAIL" if proc.returncode != 0 else "???"
                print(f"{tag}")
                for line in out.split("\n")[-3:]:
                    print(f"    {line}")

    # Summary table
    print()
    print("=" * 96)
    print("SUMMARY (median us)")
    print("=" * 96)

    # Header
    hdr = f"{'M':>6s}"
    for label, _, _ in VARIANTS:
        hdr += f"  {label:>30s}"
    print(hdr)
    print("-" * len(hdr))

    for m_val in m_values:
        row = f"{m_val:>6d}"
        for vi in range(len(VARIANTS)):
            r = results[m_val][vi]
            if r:
                row += f"  {r[0]:>30.1f}"
            else:
                row += f"  {'FAIL':>30s}"
        print(row)

    # Ratio table: gated / non-gated for each pipeline shape
    print()
    print("GATED / NON-GATED RATIO (>1.0 = gated is slower)")
    print("-" * 60)
    print(f"{'M':>6s}  {'2x3 (128x128)':>16s}  {'3x3 (64x128)':>16s}")
    print("-" * 60)
    for m_val in m_values:
        row = f"{m_val:>6d}"
        # 2x3: non-gated=0, gated=2
        ng_2x3 = results[m_val][0]
        g_2x3 = results[m_val][2]
        if ng_2x3 and g_2x3:
            ratio = g_2x3[0] / ng_2x3[0]
            row += f"  {ratio:>16.2f}x"
        else:
            row += f"  {'N/A':>16s}"

        # 3x3: non-gated=1, gated=3
        ng_3x3 = results[m_val][1]
        g_3x3 = results[m_val][3]
        if ng_3x3 and g_3x3:
            ratio = g_3x3[0] / ng_3x3[0]
            row += f"  {ratio:>16.2f}x"
        else:
            row += f"  {'N/A':>16s}"
        print(row)

    print()
    print("If gated 2x3 ratio >> 1.0, the fused gated 128x128 path has a regression.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Microbenchmark: {gated,non-gated} x {2x3,3x3} MoE kernel variants"
    )
    ap.add_argument("--hidden-size", type=int, default=6144)
    ap.add_argument("--inter-size", type=int, default=24576)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--m-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_M_VALUES),
        help="Comma-separated M (num_tokens) values to benchmark.",
    )
    ap.add_argument(
        "--single",
        nargs=4,
        type=int,
        metavar=("TILE_M", "TILE_N", "FUSE", "NUM_TOKENS"),
        help="Run a single case in-process (used by orchestrator subprocess).",
    )
    args = ap.parse_args()

    if args.single is not None:
        tile_m, tile_n, fuse, num_tokens = args.single
        benchmark_single(
            tile_m, tile_n, fuse, num_tokens,
            warmup=args.warmup,
            iters=args.iters,
            hidden_size=args.hidden_size,
            inter_size=args.inter_size,
            num_experts=args.num_experts,
            topk=args.topk,
        )
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
