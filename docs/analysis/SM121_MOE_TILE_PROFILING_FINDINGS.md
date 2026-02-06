## Purpose

Document what we learned profiling SM120/SM121 CUTLASS MoE tiles (MXFP4) on GB10 (SM121), focusing strictly on the CUTLASS kernel behavior and actionable next steps for performance.

## Setup

- **GPU**: NVIDIA GB10 (SM121)
- **Torch-reported limits**:
  - **SM count**: 48
  - **Registers per SM**: 65,536
  - **Shared memory per SM**: 102,400 B
  - **Shared memory per block (default)**: 49,152 B
  - **Shared memory per block (opt-in)**: 101,376 B
  - **Max threads per SM**: 1,536
- **Benchmark harness**: `scripts/benchmarks/benchmark_moe_tiles.py` copied into container as `/tmp/benchmark_moe_tiles.py`
- **Workload parameters** (decode-like):
  - hidden_size=6144, inter_size=24576, num_experts=8, topk=2
  - Deterministic routing/scales to ensure comparable kernel work across runs.

## Benchmark results: which tiles actually work?

We swept a handful of tiles that are listed as “supported” in FlashInfer for SM120:

- **Works**:
  - `(64,128)` (baseline)
  - `(32,128)` (works, ~same perf)
  - `(32,64)` (works, ~same perf)

- **Fails at runtime** (despite being in `SM120_SUPPORTED_TILE_MN`):
  - `(16,128)`: **`CUDA error: unspecified launch failure`**
  - `(16,64)`: **`CUDA error: unspecified launch failure`**
  - `(16,256)`: **`CUDA error: unspecified launch failure`** (observed earlier)
  - `(64,64)`: **launcher init fails**:
    - Root cause (confirmed): the kernel requests **102,400 bytes** dynamic shared memory per block:
      - `cudaFuncSetAttribute(MaxDynamicSharedMemorySize=102400) failed: invalid argument`
      - GB10’s per-block opt-in limit is **101,376 B**, so this is **unlaunchable** as-is.
    - Symptom (before improved logging): `initialize failed: Error Internal`

**Actionable takeaway**: right now the “smallest” tiles that would plausibly help decode (M=16) are **not usable**; therefore “tile selection tuning” cannot deliver meaningful gains until we fix these runtime failures.

## Critical new link: `16x128` is actually an **N=16** kernel

We ran `compute-sanitizer` on the failing logical tile `tile_mn=(16,128)` and it reported the instantiated CUTLASS kernel’s
mainloop tile shape as:

- **TileShape_MNK = (128,16,128)** (from the template parameters shown in the sanitizer output)

This is extremely important: it means the “small M” path is effectively relying on a kernel with **CTA_N = 16** (i.e. the
same problematic **N=16** regime we’ve been debugging for `128x16`).

**Actionable takeaway**: to unlock decode-relevant small-M tiles like `16x128`, we must first fix (or provide an alternate
implementation for) the **CTA_N=16** blockscaled-TMA mainloop path. Until N=16 is stable, the best tiles for decode are
unavailable by construction.

## Nsight Compute: why tile tweaks don’t move the needle yet

We profiled the CUTLASS MoE kernel using `ncu --set basic` for M=32 at the same workload.

### `64x128` vs `32x128`

Both are **1 block/SM limited** by **registers + shared memory**, with essentially identical occupancy.

- **Registers/thread**: 168 (both)
- **Dynamic smem/block**:
  - `64x128`: 92.16 KB
  - `32x128`: 101.38 KB
- **Theoretical occupancy**: 25%
- **Achieved occupancy**: ~22.5%
- **Block limit**: 1 block/SM (regs and smem)

Throughput metrics:
- `64x128`: Compute 23.61%, Memory 21.15%
- `32x128`: Compute 11.74%, Memory 16.11%

Nsight’s SoL rule flags this regime as **latency-limited** (both compute and memory utilization far below peak).

### `64x128` vs `32x64`

`32x64` also shows **the same fundamental constraints**:

- **Registers/thread**: 168
- **Dynamic smem/block**: 101.38 KB (higher than baseline)
- **Theoretical occupancy**: 25%
- **Achieved occupancy**: ~21–22%
- **Block limit**: 1 block/SM (regs and smem)

So even when the tile changes, we’re not changing the dominant resource bottleneck enough to get >1 CTA/SM, which means we can’t hide latency and shouldn’t expect large gains.

## Stall signals (explicit warp-issue stall metrics)

Because `WarpStateStats` didn’t include the per-reason breakdown, we collected explicit `smsp__warp_issue_stalled_*_per_warp_active.pct` metrics.

For M=32:

- Dominant named stalls are **`stalled_wait`** and **`stalled_long_scoreboard`**.
- This is consistent with **latency/dependency** being the limiter, and with only **1 block/SM** there’s limited ability to hide those waits.

## A concrete, actionable performance direction

If we want a real 5–10% win *from tile work*, we need at least one of:

1. **Make the small tiles (M=16) actually run**: `(16,128)`, `(16,64)`, `(16,256)` currently crash with launch failure.
2. **Reduce per-CTA resource usage** so we can schedule **2 CTAs/SM**:
   - With 384 threads/block and 168 regs/thread, the reg footprint is \(384 * 168 = 64,512\) regs/block, essentially saturating the 65,536 regs/SM → hard 1 block/SM.
   - Even if registers drop, smem is ~92–101 KB/block and SM has 102.4 KB → also hard 1 block/SM.

Therefore, **the most actionable kernel-side change** is to target **both**:
- **regs/thread** substantially below ~85 (to allow 2 blocks by regs), and
- **dynamic smem** below ~51 KB (to allow 2 blocks by smem, given 102.4 KB/SM).

This is not a “tile selection” change; it’s a **kernel configuration** change (pipeline stages, smem layouts, epilogue smem, etc.), or adding a truly small-CTA variant for decode.

## Next steps (ordered)

1. **Diagnose why `(16,128)` and `(16,64)` crash**:
   - Run a single-case repro under `CUDA_LAUNCH_BLOCKING=1` + `cuda-gdb` or `compute-sanitizer` to recover the first failing PC / cause.
   - Verify whether it’s the same class of issue as the known small-N (e.g., TMA descriptor invalid / tensormap plane invalid) or something different (e.g., shared memory overrun).
2. **Understand the `(64,64)` init failure at launcher line 763**:
   - That’s a host-side failure in the SM120 mixed-input launcher. It should tell us exactly which CUTLASS invariant fails for M64N64.
3. Once a small tile actually runs, re-run `ncu --set basic` to see if it reduces **regs**/**smem**, and only then consider updating tile selection logic.

.

