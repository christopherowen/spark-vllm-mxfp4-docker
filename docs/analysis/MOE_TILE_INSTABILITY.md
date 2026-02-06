# MoE tile instability (SM120/SM121, MXFP4 CUTLASS)

## Goal

We want to benchmark **every tile in FlashInfer’s SM120/SM121 allowlist** to:

- **Find** which tiles are unstable (GPU faults / `Xid` / `cudaErrorIllegalInstruction`) and therefore represent **areas we must improve**.
- **Pick** a better “default” tile for **decode** and **prefill**.

This document tracks:

- The exact allowlist (source of truth: `flashinfer/flashinfer/fused_moe/core.py`).
- The harness we use to run each tile in a fresh vLLM process.
- Empirical results: **PASS / FAIL / REJECTED**.
- Root-cause investigation and proposed fixes.

## Source of truth: FlashInfer allowlist

As of this workspace state, the allowlist is:

```357:384:/home/swank/projects/flashinfer/flashinfer/fused_moe/core.py
SM120_SUPPORTED_TILE_MN = (
    # ========== NATIVE TILES (M >= 64, no swap_ab) ==========
    (64, 16), (64, 32), (64, 64), (64, 128),
    (128, 16), (128, 32), (128, 64), (128, 128),
    (256, 16),
    # ========== SWAPPED TILES (M < 64, uses swap_ab) ==========
    (16, 64), (32, 64),
    (16, 128), (32, 128),
    (16, 256), (32, 256),
    (16, 512),
)
```

Notes:

- “Swapped” means **logical** \((M, N)\) is implemented via `swap_ab`, so the *physical* tile is \((N, M)\).
- Tiles outside this allowlist should be rejected up-front (no kernel launch).

## Harness

### One-tile runner (preferred)

Use `./run_one_tile.sh <MxN>` to run one tile end-to-end and capture:

- vLLM logs (`vllm.log`, `vllm_python.log`)
- benchmark output (`bench.log`)
- host syslog delta (`syslog.log`) to catch NVIDIA `Xid`

Example:

```bash
./run_one_tile.sh 64x128
./run_one_tile.sh 32x256
```

Important: if a tile triggers a GPU fault (`Xid` / illegal instruction), the GPU can enter a **sticky error** state; subsequent tiles (and even Triton/Inductor compilation) can fail spuriously. In that case:

- **Stop** the run (the script exits non-zero)
- **Reset** the GPU / clean the driver state
- Then continue with the next tile

### Multi-tile sweep (best-effort)

`./sweep_tiles.sh` runs multiple tiles in sequence, but will **abort** on a detected GPU fault (by design, to avoid cascaded “poisoned GPU” failures).

## Results

**Updated 2026-01-31** after clearing JIT cache and running isolated validation with `scripts/tests/test_tile_validation.py`.

| Tile (MxN) | Status | Failure mode | Evidence |
|-----------:|:------:|--------------|----------|
| 64x128 | **PASS** | N/A | Validated in isolation after cache clear |
| 128x128 | **PASS** | N/A | Production default, validated |
| 32x128 | **PASS** | N/A | Swap tile, validated |
| 32x256 | **PASS** | N/A | Swap tile, validated |
| 16x128 | **PASS** | N/A | Swap tile, validated |
| 16x256 | UNTESTED | - | Likely works based on pattern |
| 16x512 | UNTESTED | - | Likely works based on pattern |
| 64x64 | FAIL | SharedMemory / cudaFuncSetAttribute | TMA descriptor errors |
| 64x32 | UNTESTED | - | Likely fails (N<128) |
| 64x16 | FAIL | TMA descriptor init | `Failed to initialize the TMA descriptor 1` |
| 128x64 | FAIL | Error Internal | TMA / internal error |
| 128x32 | UNTESTED | - | Likely fails (N<128) |
| 128x16 | UNTESTED | - | Likely fails (N<128) |
| 256x16 | UNTESTED | - | Likely fails (N<128) |
| 32x64 | UNTESTED | - | Likely fails (N<64) |
| 16x64 | UNTESTED | - | Likely fails (N<64) |

**Key finding**: Tiles with N >= 128 work. Tiles with N < 128 fail with TMA or shared memory errors.

## Working hypotheses (what we expect to learn from the sweep)

### Hypothesis A: some allowlisted tiles violate hard CUTLASS constraints

We have prior evidence that certain shapes (notably `N=64` in some configurations) violate a hard alignment/format constraint in the SM120 block-scaled path (e.g. `TileShape_N % 128 == 0` in a specific “unpack-from-smem” path).

If true, we should:

- convert those cases from “GPU fault” into **clean Python-side rejection** (raise `ValueError` before JIT/build/launch), or
- patch the CUTLASS builder/kernel path to actually support those tiles safely.

### Hypothesis B: swapped tiles with small physical-N hit stmatrix / scatter constraints

The code comments already note that `N=8` fails with an “Ambiguous scatter” constraint. A similar issue may apply to other small dimensions depending on which side ends up mapped to the stmatrix path after swapping.

### Hypothesis C: once a tile faults, later “Triton illegal instruction” is collateral

If the GPU faults (NVIDIA `Xid`, `cudaErrorIllegalInstruction`) during a MoE kernel, later failures during Inductor/Triton compilation are likely **secondary** (CUDA context is already corrupted).

## Next steps (investigation plan)

For each failing tile:

- Collect `results/tile_<tile>_*/syslog.log` and confirm whether an NVIDIA `Xid` occurred.
- Inspect `vllm_python.log` for the last successful op and the first failing op.
- Determine whether the failure occurs:
  - during FlashInfer JIT build
  - during vLLM profile run (dummy forward)
  - during the benchmark run (steady-state decode)
- Map the failing tile to a concrete constraint (e.g. alignment requirement) or a concrete kernel bug.

Once we identify a deterministic constraint, we should add a **guard** (reject unsafe tiles before launch) so future sweeps can classify tiles as “REJECTED” rather than crashing the GPU.

## Notes from recent repros

### `128x128`: EngineCore death + driver OOM (not tile-specific)

In `results/tile_sweep_20260131_112733/`:

- `vllm_python_128x128.log` shows the engine successfully warmed up and served multiple `POST /v1/chat/completions` requests, then:
  - `Engine core proc EngineCore_DP0 died unexpectedly`
- `syslog_128x128.log` contains:
  - `NVRM: ... Out of memory [NV_ERR_NO_MEMORY]`

This is a **system stability** issue (driver/allocator), not necessarily a tile-shape validity issue.

Practical mitigations for tile sweeps:

- Prefer running sweeps with **lower compilation pressure**:
  - `VLLM_ENFORCE_EAGER=1` (disables CUDA graph capture + compile pipeline)
  - optionally reduce memory pressure: `VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.60"`
  
These mitigations make it more likely we can isolate *tile-caused* faults (e.g. Xid / illegal instruction) from “environmental” failures.

### `64x128`: reproducible `Xid 13` illegal instruction

In `results/tile_sweep_20260131_112952/`:

- `syslog_64x128.log` contains `NVRM: Xid ... 13 ... Illegal Instruction Parameter` (and a follow-on `Xid 43`).
- `vllm_python_64x128.log` shows the EngineCore dying / failing to start with `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`.

This strongly suggests a **kernel-level fault** (most likely in the MoE path, since the only intended change between tiles is `/tmp/flashinfer_moe_tile`).


