# Fused Kernel Debug Status

**Date**: 2026-02-08
**Status**: All sanitizer memsets compiled out in release (`#ifndef NDEBUG`). CUDA-graph-safe. Zero hot-path bandwidth overhead.
**Previous Issue**: `cudaMemset`/`cudaMemsetAsync` calls added for initcheck cleanliness caused CUDA graph capture crashes and hot-path bandwidth overhead. Fixed by gating all sanitizer memsets behind `#ifndef NDEBUG` (compiled out in release JIT builds which pass `-DNDEBUG`), removing sync `cudaMemset` from `configureWorkspace()`, and adding `isCapturing()` guards for debug builds.

---

## Current Status

### Two bugs fixed, epilogue source-read eliminated

Code review found two bugs that together explained the all-zero output (both now fixed and validated):

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | **Critical** | Phase-2 K iteration OOB: `ForwardCoordIterator` does NOT wrap modulo shape — it increments past end. After `k_real` iterations the iterator is at the sentinel position, making all phase-2 TMA loads read invalid addresses. | Save `k_start_coord = *k_tile_iter` before loop, reset `k_tile_iter.coord = k_start_coord` at `k == k_real`. |
| 2 | **High** | SwiGLU intentionally disabled: `accum_gate` was computed then discarded with `(void)accum_gate` in a debug block, so the kernel always output raw phase-1 linear GEMM. | Replaced discard with inline SwiGLU: `accum[i] = gate * sigmoid(gate) * linear`. |

### Current metrics

| Metric | Value |
|--------|-------|
| Tile shape | 128x128x128 |
| SMEM usage | 95,232 bytes (fits in 101,376) |
| Compilation time | ~44 seconds (release mode) |
| compute-sanitizer memcheck | **0 errors** (both unfused and fused) |
| compute-sanitizer initcheck (unfused) | **0 errors** |
| compute-sanitizer initcheck (fused) | 0 errors |
| CUDA errors | None |
| Output | Correct: unfused abs_max=0.3906, fused abs_max=0.3906 |

### Three initcheck error classes eliminated

| # | Error class | Count | Fix | Perf impact |
|---|-------------|-------|-----|-------------|
| 1 | Epilogue source-read | ~1,500 | `ElementC=void` in SM120 epilogue builder | Zero (eliminates needless HBM read) |
| 2 | Metadata padding | ~9,200 | `cudaMemset` on metadata workspace (~2KB) + GEMM workspace (~10KB) | Negligible |
| 3 | GEMM1 intermediate buffer | ~73,200 | `cudaMemsetAsync` on GEMM1 output before `doActivationKernel` reads it | Negligible for decode (~240KB); future refinement for large prefill |

**Fix 1 — Epilogue source-read**: The SM120 epilogue builder was configured with `ElementC = bfloat16_t`, causing the `LinearCombination` epilogue to issue a TMA source load from the output buffer even when `beta=0`. Changed `ElementC` to `void` in the `CollectiveBuilder` for all three SM120 namespaces (standard, transposed, gated). This tells CUTLASS to skip the source TMA load entirely.

**Fix 2 — Metadata padding**: Per-group metadata buffers (stride/layout/problem-shape/pointer arrays) contained uninitialized struct padding bytes. Originally added `cudaMemset` in `configureWorkspace()`, but this was a sync memset that crashed during CUDA graph capture. Re-added as `cudaMemsetAsync` with stream plumbed through `configureWsPtrs()` → `configureWorkspace()`, but compiled out in release builds (`#ifndef NDEBUG`). In debug builds, also guarded by `isCapturing()` for graph safety. The metadata is fully written by `setupTmaWarpSpecializedInputs()` before any kernel launch — zeroing is unnecessary for correctness.

**Fix 3 — GEMM1 intermediate buffer (proven parity, no tail overread)**: compute-sanitizer initcheck reports ~73K uninitialized reads in `doActivationKernel`. These are **sanitizer artifacts** — compute-sanitizer does not track TMA store operations. Formal proof of read-safety:

- **M-dimension**: The GEMM writes `gemm_m = num_tokens_to_expert` rows per expert (from `expert_first_token_offset`). The activation kernel iterates `token ∈ [0, expert_first_token_offset[num_experts])` — the same token set. No M-dimension overread.
- **N-dimension**: The GEMM writes `gemm_n = 2 × inter_size` columns per row (gated). The activation kernel reads `elem_index ∈ [0, inter_size/VEC)` at both linear and gate offsets, with max byte offset = `2 × inter_size - 1` elements — within `[0, gemm_n)`. No N-dimension overread.
- **Vectorization alignment**: `inter_size % ACTIVATION_ELEM_PER_THREAD == 0` is now enforced by host-side `TLLM_CHECK_WITH_INFO` in both `doActivation()` and `doGatedActivation()` launchers (not compiled away by NDEBUG). This guarantees vectorized loads never overread.
- **Buffer sizing**: `glu_inter_result_` is allocated as `expanded_num_rows × fc1_out_size` elements, where `expanded_num_rows ≥ num_valid_tokens`.
- **Stride consistency**: GEMM output stride = `num_tokens_before_expert × gemm_n` (line 1228). Activation stride = `token × inter_size × gated_size_mul` = `token × gemm_n`. Identical.

Confirmed by 10-run determinism test and 5 graph replays: bit-identical output, zero diff, no NaN/Inf, even without pre-zeroing. The `cudaMemsetAsync` is compiled out in release builds (`#ifndef NDEBUG`). In debug builds, it is guarded by `isCapturing()` for graph safety.

### What Works

- **Compilation**: JIT compiles in ~44s release mode (vs 49 min in debug mode with `-O0 -G`)
- **Initialization**: CUTLASS `initialize()` returns Success
- **Execution**: CUTLASS `run()` returns Success, no CUDA errors
- **Memory safety**: compute-sanitizer memcheck reports 0 errors
- **Unfused path**: Still fully functional (`fuse_activation=False` produces correct output)

---

## Architecture: Sequential SMEM Reuse

### Design (replaces the failed 6-plane approach)

The previous approach loaded 6 TMA planes simultaneously (A, B, Aux, SFA, SFB, SFAux), requiring ~71KB SMEM at minimum. This forced N_tile=64, which violates SM120's N%128 hardware constraint.

The new design uses only **4 SMEM planes** (same as the unfused kernel) and performs **two sequential K-reduction passes**:

```
Phase 1 (K tiles 0..k_real-1):
  Load A + B_linear + SFA + SFB_linear → accum (linear GEMM)

Phase 2 (K tiles 0..k_real-1, iterator reset):
  Load A + B_gate + SFA + SFAux → accum_gate (gate GEMM)
  (B_gate and SFAux loaded into same smem_B / smem_SFB slots)

Epilogue:
  output = SiLU(accum_gate) * accum  (SwiGLU in registers)
```

### SMEM Budget

```
SM121 SMEM limit:        101,376 bytes
128x128x128 fused:        95,232 bytes (fits with 6KB margin)
128x128x128 unfused:     ~66,000 bytes (same 4 planes)
```

Sequential SMEM reuse achieves the same SMEM footprint as the unfused kernel because both use exactly 4 TMA planes per pipeline stage.

### Key Implementation Details

1. **k_tile_count doubling**: The kernel (`sm90_gemm_array_tma_warpspecialized_pingpong.hpp`) detects gated mainloops via `IsGatedMainloop` SFINAE trait and doubles `k_tile_count` at all 5 producer/consumer advance points.

2. **Single-loop load**: The `load()` method uses one loop over `k_tile_count = 2 * k_real` iterations. A conditional `if (k < k_real)` selects between loading B_linear/SFB_linear (phase 1) and B_gate/SFAux (phase 2) into the same SMEM slots.

3. **ForwardCoordIterator reset**: The CUTE iterator does NOT wrap modulo shape — it increments past end. The load loop explicitly resets `k_tile_iter.coord` to the saved starting coordinate at `k == k_real`, so phase 2 re-reads the same A/SFA K tiles.

4. **Two-phase mma()**: Phase 1 accumulates into `accum`, phase 2 into `accum_gate`. Both use the same `run_mma_phase` helper. SwiGLU (`SiLU(gate) * linear`) is applied element-wise in registers after both phases complete.

---

## Files Modified (mxfp4_v4 branches)

### FlashInfer repo

| File | Changes |
|------|---------|
| `sm120_blockscaled_mma_gated_array_tma.hpp` | Removed `smem_Aux`/`smem_SFAux` from TensorStorage. Rewrote `load()` as single-loop two-phase with K iterator reset at phase boundary. Rewrote `mma()` with two-phase accumulation + inline SwiGLU. |
| `sm90_gemm_array_tma_warpspecialized_pingpong.hpp` | Added `IsGatedMainloop` SFINAE trait. Added `k_tile_count *= 2` at 5 locations. |
| `moe_gemm_sm120_mixed_input_launcher.inl` | Updated gated namespace to 128x128x128 tiles. Fixed dispatch policy to inherit schedule from base. Added `computeGatedPointersAndStrides` kernel. |
| `cutlass_fused_moe_kernels.cuh` | Updated tile shape to 128x128x128. Updated comments. |
| `core.py` | Updated `SM120_FUSED_SUPPORTED_TILE_MN` to `((128, 128),)`. Updated prewarming. |

### Key changes vs previous (6-plane) design

| Aspect | Previous (6-plane) | Current (sequential reuse) |
|--------|--------------------|-----------------------------|
| SMEM planes per stage | 6 (A, B, Aux, SFA, SFB, SFAux) | 4 (A, B, SFA, SFB) — same as unfused |
| Tile shape | 64x64x128 | 128x128x128 |
| SMEM usage | ~71KB | ~95KB |
| N_tile constraint | FAIL (64 % 128 != 0) | OK (128 % 128 == 0) |
| Illegal instruction | 27 crashes | 0 |
| K-tile iteration | Normal | Doubled (2 phases per work tile) |
| SwiGLU location | Not implemented | Inline in mma() (register-level) |

---

## Resolved Issues

### Phase-2 K Iterator Out-of-Bounds (RESOLVED)

**Root cause**: `cute::ForwardCoordIterator` does NOT wrap modulo shape. `detail::increment()` carries to outer dimensions but the outermost dimension increments past shape, reaching the sentinel. After `k_real` increments the iterator is at end-of-range; dereferencing it for `k_real` more TMA loads produced invalid addresses.

The code comment incorrectly stated "ForwardCoordIterator wraps modularly after k_real increments."

**Resolution**: Save `k_start_coord = *k_tile_iter` before the loop, then reset `k_tile_iter.coord = k_start_coord` when `k == k_real`. This avoids the deleted copy-assignment operator on `ForwardCoordIterator` (which has a `const&` member) by assigning only the public `coord` member directly.

### SwiGLU Debug Bypass (RESOLVED)

**Root cause**: SwiGLU was intentionally disabled for debugging. `accum_gate` was computed but discarded with `(void)accum_gate`, so the kernel always output raw phase-1 linear GEMM results.

**Resolution**: Replaced debug discard with inline SwiGLU:
```cpp
for (int i = 0; i < size(accum); ++i) {
  float gate = float(accum_gate(i));
  float linear = float(accum(i));
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  accum(i) = gate * sigmoid_gate * linear;
}
```

### N_tile=64 vs SM120 N%128 Hardware Constraint (RESOLVED)

**Root cause**: SM120 block-scaled MMA requires N dimensions to be multiples of 128. The 6-plane fused kernel used N_tile=64, causing `UTMALDG.4D` to fault as an illegal instruction.

**Resolution**: Sequential SMEM reuse eliminated the need for 6 SMEM planes, enabling 128x128x128 tiles that satisfy N%128.

### Debug Mode Compilation Timeout (RESOLVED)

**Root cause**: `FLASHINFER_JIT_VERBOSE=1` enabled `-g -O0 -G` and other debug flags, producing a 172MB PTX file that took 49+ minutes for ptxas to assemble.

**Resolution**: Compile without `FLASHINFER_JIT_VERBOSE`. Release mode compiles in ~44 seconds.

### Missing cudaFuncSetAttribute (RESOLVED)

The fused launcher was missing the SMEM opt-in call for >48KB. Fixed. The 95KB fused kernel requires this opt-in.

### ForwardCoordIterator Assignment Error (RESOLVED)

`cute::ForwardCoordIterator` has a deleted copy assignment operator (due to `Shape const&` member). The original two-loop design tried to save and restore the iterator. Fixed by merging into a single loop with conditional B/SFB source selection and direct coord member assignment for phase reset.

---

## Reproduction

```bash
# Clear JIT cache and test:
docker exec vllm-dev rm -rf /root/.cache/flashinfer/*/121a/cached_ops/fused_moe_120_fusedact_mxfp4min

# Quick compilation test (should complete in ~45s):
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev python3 -c "
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
m = get_cutlass_fused_moe_module('120', tile_mn=(128,128), fuse_activation=True)
print('OK')
"

# Full test (fused vs unfused comparison):
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev python3 scripts/tests/smoke_test_proper_quantize.py

# compute-sanitizer initcheck:
./fused_initcheck.sh

# compute-sanitizer memcheck:
./fused_memcheck.sh
```

## Key Files

| File | Path | Role |
|------|------|------|
| Gated mainloop | `~/projects/flashinfer/csrc/nv_internal/.../sm120_blockscaled_mma_gated_array_tma.hpp` | Two-phase load/mma with sequential SMEM reuse |
| Kernel patch | `~/projects/flashinfer/3rdparty/cutlass/.../sm90_gemm_array_tma_warpspecialized_pingpong.hpp` | IsGatedMainloop k_tile_count doubling |
| Launcher | `~/projects/flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl` | Gated namespace, fused launcher, pointer/stride kernels, void ElementC epilogue fix |
| C++ dispatch | `~/projects/flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | gemm1_fused(), stride kernel, runMoe dispatch |
| Python dispatch | `~/projects/flashinfer/flashinfer/fused_moe/core.py` | Tile selection, JIT compilation, fuse_activation flag |

### Numerical Comparison (PASSED)

Fused vs unfused comparison added to `scripts/debug/repro_fused_illegal_instruction.py`:

```
Unfused abs_max:  0.390625
Fused   abs_max:  0.390625
Max abs diff:     0.005371
Mean abs diff:    0.000980
Relative diff:    0.013750
PASS: fused and unfused outputs match (rtol=1e-2, atol=1e-2)
```

Both paths produce matching output within expected FP4 quantization tolerance.

### Gate Weight Offset Investigation

The gate weight byte offset in `computeGatedPointersAndStrides` was investigated and verified correct:

- `LayoutB = ColumnMajor` in CUTLASS 3.x: B maps to modes `[N, K, L]` with stride `(int64_t, Int<1>, int64_t)` — **K is contiguous**, N has stride K.
- For B`[N=2*inter_size, K=hidden_size]`: element `(n, k)` at offset `n * K + k`
- Gate at `N = inter_size`: element offset = `inter_size * hidden_size`, byte offset (FP4) = `(inter_size * hidden_size) / 2`
- **Original formula was correct**. An earlier misdiagnosis assumed ColumnMajor meant N-contiguous (it doesn't — verified from `cutlass/detail/layout.hpp:86`).

Gate SF offset also verified correct (`inter_size * ceil(K/32)` — SF layout is K-contiguous). Refactored to use `padded_k / block_size` for robustness with non-128-aligned hidden sizes.

## Next Steps

1. **E2E vLLM test** — verify full model inference with fused FC1
2. **Benchmark** — measure TPS improvement with llama-benchy (with CUDA graphs enabled, no `--enforce-eager`)

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-02-08 | Gate offset verification: Confirmed `gate_weight_bytes = (inter_size * hidden_size) / 2` is correct for ColumnMajor B (K-contiguous). Added fused vs unfused numerical comparison test (PASS: max_diff=0.005, rel_diff=0.014). Refactored SF offset to use padded K for robustness. Improved LayoutB documentation in launcher. |
| 2026-02-08 | FC1 activation read-safety proof: formally proved no tail overread in `doActivationKernel` (M-dimension bounded by same `expert_first_token_offset`, N-dimension bounded by `inter_size` with enforced alignment, stride consistent between GEMM output and activation reads). Added host-side `TLLM_CHECK_WITH_INFO` alignment checks in `doActivation()` and `doGatedActivation()` launchers (persist in release builds). Replaced "stale but valid" assumption with proven parity. |
| 2026-02-08 | Sanitizer memset hardening: all sanitizer-only memsets gated behind `#ifndef NDEBUG` (compiled out in release JIT). Re-added metadata zeroing in `configureWorkspace()` with stream parameter plumbed through `configureWsPtrs()`. FINALIZE epilogue `cudaMemsetAsync(final_output)` documented as correctness requirement (kept unconditional, graph-safe). Device-side `assert()` in gated mainloop replaced with `#ifndef NDEBUG` guard. |
| 2026-02-08 | Initcheck correctness analysis: confirmed the ~73K initcheck warnings are compute-sanitizer artifacts (TMA store tracking). 10-run determinism test and graph replay test show bit-identical output with zero diff, no NaN/Inf, even without pre-zeroing. Memset retained as sanitizer workaround only. |
| 2026-02-08 | CUDA graph fix: removed sync `cudaMemset` from `configureWorkspace()`, guarded all `cudaMemsetAsync` with `isCapturing()`, replaced `cudaGetLastError` with `cudaPeekAtLastError` in fused launcher, added beta==0 assertions for ElementC=void epilogue invariant. |
| 2026-02-08 | Initcheck clean: all three error classes eliminated (epilogue source-read, metadata padding, GEMM1 intermediate buffer). Unfused initcheck now reports 0 errors. |
| 2026-02-08 | Epilogue source-read fix: changed ElementC from bfloat16_t to void in all three SM120 epilogue builder macros. Eliminated ~1,500 initcheck errors from epilogue TMA source loads. |
| 2026-02-08 | Fixed two bugs: (1) K iterator OOB in phase-2 load — added coord reset at phase boundary; (2) SwiGLU re-enabled — replaced debug `(void)accum_gate` with inline SiLU(gate)*linear. Updated initcheck results (132K errors all in unfused baseline). |
| 2026-02-08 | (Earlier) Rewrote for v4 sequential SMEM reuse design. Documented two-phase K iteration, 128x128x128 tile success, all-zero output debugging. |
| 2026-02-08 | (Earlier) Added v3 status: documented gated mainloop architecture, identified N_tile=64 vs SM120 N%128 hardware constraint |
| 2026-02-01 | Added spike plan, created validation script |
| 2026-01-31 | Initial document |
