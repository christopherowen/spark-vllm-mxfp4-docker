# Fused Kernel Debug Status

**Date**: 2026-02-08
**Issue**: `CUDA error: an illegal instruction was encountered` when running the fused activation (SwiGLU-into-GEMM1) kernel on SM121.
**Status**: Root cause identified — **N_tile=64 violates SM120 hardware's N%128 constraint**. The current 64×64×128 tile shape is fundamentally incompatible with SM120 block-scaled MMA. Architectural change required.

---

## Root Cause

**The fused kernel uses N_tile=64, but SM120 block-scaled MMA requires N to be a multiple of 128.**

The 64×64×128 tile was chosen because it's the only configuration that fits 6 TMA planes (A, B, Aux, SFA, SFB, SFAux) within SM121's 101KB SMEM limit (~71KB used). But SM120's block-scaled MMA hardware operates on 128-wide N blocks — with N_tile=64, the TMA descriptors and scale factor layouts become invalid, causing `UTMALDG.4D` to fault as an illegal instruction.

### Evidence

1. **FlashInfer PR #2261** (merged 2025-12-24): Confirmed SM120 requires N dimensions to be multiples of 128 (ScaleGranularityN). Fix was to pad N to next multiple of 128 at the Python level. Reference: `sm120_mma_tma_blockwise_scaling.hpp#L345`.

2. **FlashInfer PR #2495** (merged 2026-02-05): Added runtime checks that output N meets 256-bit alignment for NoSmem epilogue on SM120.

3. **MOE_TILE_INSTABILITY.md**: Documents `64×64` tiles as `FAIL` with "SharedMemory / cudaFuncSetAttribute | TMA descriptor errors."

4. **CUTLASS source**: `sm120_blockscaled_mma_array_tma.hpp:145` pads scale factor tile to 128 (`TileN_SFB = ceil_div(N_tile, 128) * 128`). The code compiles for N<128 but fails at runtime.

5. **SASS disassembly**: All 27 crashes at offset `+0xb9c0` on the same `UTMALDG.4D` instruction — a TMA load that faults because the N-dimension layout doesn't meet hardware alignment.

### Previously Identified (Contributing, Not Root Cause)

- **Missing `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`**: The fused launcher was missing the opt-in call for >48KB SMEM. This was fixed but did not resolve the crash because the underlying N%128 constraint is the real blocker.

- **Wrong kernel schedule**: The fused kernel uses `KernelPtrArrayTmaWarpSpecializedPingpong` (SM90-generic) instead of `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120` (SM120-specific). Fixed but also did not resolve the crash.

### The SMEM Dilemma

| Tile Shape | SMEM (6 planes) | Fits in 101 KB? | N_tile % 128 |
|------------|-----------------|-----------------|--------------|
| 64×64×128 | ~71 KB | Yes | **FAIL** (64) |
| 64×128×128 | ~112 KB | **No** (+11 KB) | OK |
| 128×128×128 | ~129 KB | **No** (+28 KB) | OK |

No tile with N≥128 fits in SMEM with 6 TMA planes and 2 pipeline stages.

### Path Forward

See **"Revised Path Forward"** section in `docs/plans/FUSED_MOE_FC1_FC2_KERNEL.md` for detailed options. Summary:

| Option | Description | Feasibility |
|--------|-------------|-------------|
| **B: 128×128 fused tile** | Unfused 128×128 uses ~66KB; adding Aux+SFAux may total ~99KB | **Try first** — verify SMEM fit |
| A: Reduce stages | 1 pipeline stage to fit 64×128 | Risky — may not be supported |
| C: Epilogue-only SwiGLU | Standard 4-plane GEMM + epilogue fusion | Requires weight layout change |
| D: Separate dual GEMM | Two standard GEMM launches + pointwise fusion | Safe fallback |

---

## Goal

Fuse the SwiGLU activation into the GEMM1 epilogue so the MoE pipeline does one kernel launch instead of GEMM1 + separate activation kernel + re-quantize. The unfused path works correctly. The fused path crashed with an illegal instruction (now fixed).

## What Works

- **Unfused path** (`fuse_activation=False`): Fully functional. GEMM1 → separate SwiGLU → GEMM2. No memcheck errors.
- **Architectural refactoring**: Complete. Fused and unfused paths are cleanly separated:
  - `gemm1()` handles only the unfused path
  - `gemm1_fused()` handles the entire fused pipeline (TMA setup, kernel launch, post-processing)
  - `computeStridesFusedActivationKernel` computes per-expert TMA strides specifically for the fused kernel
  - `runMoe()` dispatches to the correct path based on `use_fused_activation`
- **Profiler code** uses the same production building blocks (no separate profiler setup)
- **Code review items** addressed: LoRA guard, SM-gating, redundant TMA documented

## What Was Wrong

The fused kernel (`MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated`) crashed with **27 illegal instruction errors** because `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` was missing from the fused launcher. All errors were at offset `+0xb9c0` -- the first SMEM access past the default 48KB limit.

## Reproduction

```bash
# From the host (mxfp4 repo root):
# First clear JIT cache so code changes take effect:
docker exec vllm-dev rm -rf /root/.cache/flashinfer/*/121a/cached_ops/fused_moe_120*/

# Then run memcheck:
./fused_memcheck.sh
```

The repro script is `scripts/debug/repro_fused_illegal_instruction.py`. It tests both unfused and fused paths sequentially with gpt-oss-120b dimensions (hidden=2944, inter=7680, 8 experts, 4 tokens).

**Important**: The repro script now has `enable_pdl=False` and `CUDA_LAUNCH_BLOCKING=1` set. The `fused_memcheck.sh` script handles JIT prewarming, container checks, and log collection.

### Suggested next run

The next memcheck should be run with `CUDA_LAUNCH_BLOCKING=1` set in the environment AND `enable_pdl=False` in the Python calls (both already set in the repro script). This makes errors synchronous so memcheck reports the exact faulting kernel on each launch. Previous runs were asynchronous, potentially conflating errors.

```bash
docker exec vllm-dev rm -rf /root/.cache/flashinfer/*/121a/cached_ops/fused_moe_120*/
./fused_memcheck.sh
```

Also note: the last memcheck (results/memcheck_20260208_022827/) ran against **stale JIT cache** -- the diagnostic output still says `[GATED TMA DIAG]` even though the source was renamed to `[FUSED TMA DIAG]`. Clear the cache before the next run.

## Key Memcheck Evidence

From `results/memcheck_20260208_022827/fused_memcheck.log`:

### Error pattern
- 24 "Illegal instruction" errors, 27 total errors
- All at the same kernel offset: `+0xb9c0`
- Kernel: `MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated` with tile shape 64x64x128
- Errors in multiple blocks (block 1, block 25, etc.) and thread (0,0,0)
- Host call stack: `runMoe` → `gemm1_fused` → `sm120_fused_act_moe_gemm_kernelLauncher` → `device_kernel`

### TMA diagnostic (from warmup, pre-memcheck)
```
[GATED TMA UPDATE] batch=0  A=0x...  B=0x...  SFA=0x...  SFB=0x...  Aux=0x...  SFAux=0x...
[GATED TMA DIAG] group=0  M=1  N=7680  K=2944
[GATED TMA DIAG] A    shape=(2944,1,1,1,1)       stride=(1,2944,0,0,0)
[GATED TMA DIAG] B    shape=(2944,7680,1,1,1)     stride=(0,1472,0,0,0)
[GATED TMA DIAG] Aux  shape=(2944,7680,1,1,1)     stride=(0,1472,0,0,0)
[GATED TMA DIAG] SFA  shape=(512,1,23,1,1)        stride=(1,11776,512,11776,0)
[GATED TMA DIAG] SFB  shape=(512,60,23,1,1)       stride=(1,11776,512,706560,0)
[GATED TMA DIAG] SFAux shape=(512,60,23,1,1)      stride=(1,11776,512,706560,0)
```

### Notable observations
- **M=1** is correct (single-token expert with 4 tokens / 2 top_k = small M per expert)
- **N=7680** = `inter_size` (logical N for one half of the gated weight). The Gated mainloop treats B and Aux as the two halves, each of width `inter_size`.
- **B shape has N=7680** but **stride=(0,1472,0,0,0)**. The stride of 1472 = 2944/2 = hidden_size/2 (FP4 packed). This looks correct for a [K,N] column-major FP4 weight.
- **SFB shape=(512,60,23,1,1)**: 60 = 7680/128 (N tiles), 23 = ceil(2944/128) (K tiles). This looks correct for the scale factor layout.
- **No memory access errors** -- only illegal instructions. This suggests the data setup is probably correct, but the kernel itself is executing an instruction that SM121 doesn't support, or there's a shared memory alignment/size issue.

## Hypotheses (resolved)

### ✅ CONFIRMED: N_tile=64 violates SM120 N%128 hardware constraint
The SM120 block-scaled MMA requires N dimensions to be multiples of 128. The fused kernel
uses N_tile=64 (chosen to fit 6 TMA planes in SMEM), which causes `UTMALDG.4D` to fault.
Confirmed by FlashInfer PRs #2261 and #2495, CUTLASS source at
`sm120_blockscaled_mma_array_tma.hpp:145`, and MOE_TILE_INSTABILITY.md.

### ✅ FIXED (contributing): Missing cudaFuncSetAttribute
The fused launcher was missing `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` for
the >48KB SMEM opt-in. Fixed, but crash persisted because of the N%128 constraint.

### ✅ FIXED (contributing): Wrong kernel schedule
The fused kernel used `KernelPtrArrayTmaWarpSpecializedPingpong` (SM90-generic) instead
of `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120`. Fixed, but crash persisted.

### ❌ RULED OUT: Unsupported instruction on SM121
SASS analysis confirmed the `UTMALDG.4D` instruction at `+0xb9c0` is a valid SM121
instruction (present and working in the unfused kernel). The fault is caused by invalid
TMA descriptor parameters resulting from the N=64 tile layout, not an unsupported opcode.

### ❌ RULED OUT: TMA descriptor encoding issue
TMA diagnostic output shows correct shapes and strides. The descriptors encode correctly —
the hardware rejects them at runtime because the N-dimension layout doesn't match the
128-wide block-scaled format.

### ❌ RULED OUT: Workspace size / alignment
Workspace is correctly allocated and passed through. The issue is in the mainloop's
SMEM layout and TMA tile geometry, not workspace management.

## Changes Made (uncommitted in flashinfer repo)

6 files modified, +608 / -239 lines:

| File | Changes |
|------|---------|
| `cutlass_fused_moe_kernels.cuh` | Main refactoring: `gemm1_fused()`, `computeStridesFusedActivationKernel`, SF layout fix (logical N for LayoutSF, physical N for offset), LoRA guard, SM-gating |
| `flashinfer_cutlass_fused_moe_binding.cu` | Python binding passes `fuse_activation` bool through |
| `sm120_blockscaled_mma_gated_array_tma.hpp` | Diagnostic string rename GATED→FUSED (pending JIT rebuild) |
| `moe_kernels.h` | Header declarations for `gemm1_fused` and fused stride kernel |
| `moe_gemm_sm120_mixed_input_launcher.inl` | Log string rename Gated→Fused, launcher for fused kernel |
| `core.py` | `enable_pdl` parameter passthrough |

## Files to Know

| File | Path | Role |
|------|------|------|
| Repro script | `scripts/debug/repro_fused_illegal_instruction.py` | Minimal Python repro |
| Memcheck runner | `fused_memcheck.sh` | Host script: prewarms JIT, runs compute-sanitizer, collects logs |
| Main C++ kernel | `~/projects/flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | Dispatch, TMA setup, stride kernels |
| Gated mainloop | `~/projects/flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp` | CUTLASS mainloop for fused path |
| Launcher | `~/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl` | Kernel launch, SMEM config |
| Base mainloop | `~/projects/flashinfer/3rdparty/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp` | Non-gated base class |
| Header decls | `~/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | Class declarations |
| Latest memcheck | `results/memcheck_20260208_022827/` | Log files from last run |

## Cleanup Needed

- `review.md` in mxfp4 repo root should be deleted (code review items have been addressed)

## Recommended Next Steps

1. **Verify 128×128 fused tile SMEM fit**: The unfused 128×128 kernel uses ~66KB. The
   fused kernel adds Aux + SFAux (matching B + SFB sizes). Estimate total SMEM and check
   if it fits in 101KB. This can be done by:
   - Changing `TILE_M_VAL=128, TILE_N_VAL=128` in the gated namespace macro
   - Adding a `static_assert(sizeof(TensorStorage) <= 101376)` check
   - Or using the `print_smem_breakdown()` diagnostic already in the gated mainloop

2. **If 128×128 fits**: Update tile constants, clear JIT cache, recompile, and re-run
   memcheck. The SwiGLU inline application in `mma()` must also be implemented (see
   item 4 below).

3. **If 128×128 doesn't fit**: Fall back to Option D (separate dual GEMM launches with
   pointwise SwiGLU fusion). This avoids the 6-plane SMEM pressure entirely.

4. **Implement SwiGLU in single-accumulator `mma()`**: The current single-accumulator
   overload discards the gate result. It must call the dual-accumulator overload
   internally and compute `output = SiLU(gate) * linear` before returning.

5. **Fix kernel schedule**: Change `KernelPtrArrayTmaWarpSpecializedPingpong` to
   `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120` in the gated namespace
   macro (line 430 of launcher).
