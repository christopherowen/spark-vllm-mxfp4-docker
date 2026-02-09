# Fused MoE FC1+FC2 Optimization Review

## Status Summary (2026-02-09)

| Section | Type | Status |
| --- | --- | --- |
| Section 1: Performance Gap | Measurement | DONE |
| Section 2: Tile Shape Verification | Analysis | DONE |
| Section 3: doActivation(Identity) | Analysis + proposal | Analysis DONE, implementation PENDING (Phase 3) |
| Section 4: Supporting Kernels / Duplicated FC1 metadata | Analysis + fix | DONE — `setupGemm2TmaInputForFused()` created, fused path uses FC2-only setup |
| Section 5: Doubled A-Operand Traffic | Root cause + proposal | Analysis DONE, implementation PENDING (Phase 2) |
| Section 6: Pipeline Bubble | Hypothesis | Documented, validation PENDING (resolved by Phase 2 if implemented) |
| Section 7: Deeper Fusion | Analysis | Analysis DONE, implementation out of scope (research-grade) |
| **Phase 1**: Eliminate redundant FC1 metadata | Implementation | **DONE** |
| **Phase 2**: Per-K-tile interleaving | Implementation | PENDING |
| **Phase 3**: EVT epilogue fusion | Implementation | PENDING |

## Section 1: Current Performance Gap [DONE — measured]

Benchmark results from `results/compare_20260208_2029/` (3 runs each, mean +/- stddev):

| Test | Unfused t/s | Fused t/s | Delta | Notes |
| --- | --- | --- | --- | --- |
| pp2048/tg32 | 58.81 +/- 0.08 | 57.86 +/- 0.10 | -1.6% | Stable |
| pp2048/tg128 | 58.99 +/- 0.03 | 58.05 +/- 0.04 | -1.6% | Stable |
| pp2048/tg512 | 58.13 +/- 0.02 | 57.47 +/- 0.49 | -1.1% | Fused variance 25x higher |
| pp8192/tg32 | 56.62 +/- 0.01 | 52.68 +/- 4.27 | -7.0% | **Outlier**: fused stddev ~4.3 vs unfused ~0.01 |
| pp8192/tg128 | 56.83 +/- 0.01 | 55.88 +/- 0.02 | -1.7% | Stable |
| pp8192/tg512 | 56.21 +/- 0.04 | 55.62 +/- 0.40 | -1.0% | Fused variance 10x higher |
| pp2048 (prefill) | 4808-4926 +/- 24-63 | 4527-4555 +/- 5-57 | -6 to -8% | |
| pp8192 (prefill) | 6654-6693 +/- 11-17 | 5723-6355 +/- 15-923 | -5 to -14% | **Outlier**: pp8192/tg32 fused = 5723 +/- 923 |

**Outlier policy**: The pp8192/tg32 fused row shows extreme variance (923 t/s stddev on prefill, 4.27 on decode). This suggests one of the three runs had a severe performance anomaly (possibly thermal throttling, memory pressure, or a scheduling pathology). Until re-run with more trials and variance controls, treat that row as unreliable. The remaining rows show a consistent **1-2% decode regression** and **5-8% prefill regression**.

The fused kernel currently **regresses** both decode and prefill, and introduces **variance/instability** in some configurations that the unfused path does not exhibit. The following sections explain why and propose fixes.

---

## Section 2: Tile Shape Verification (64x128 for Decode) [DONE]

The fused path uses 64x128 tiles for decode, confirmed by `select_tile_mn_for_sm120_fused()` in `core.py`:

```python
# flashinfer/fused_moe/core.py:397-412
SM120_FUSED_SUPPORTED_TILE_MN = (
    (64, 128),   # ~58KB SMEM - decode (small M), fits with ~43KB margin
    (128, 128),  # ~95KB SMEM - prefill (large M), fits with ~6KB margin
)

def select_tile_mn_for_sm120_fused(num_tokens: int) -> tuple[int, int]:
    if num_tokens < 64:
        return (64, 128)
    return (128, 128)
```

Decode (num_tokens=1) selects `(64, 128)`. JIT sets `-DLOGICAL_TILE_M=64 -DLOGICAL_TILE_N=128`.

### N%128 Hardware Constraint

N_tile must be >= 128 for SM120 block-scaled MMA. N_tile=64 is not viable. Evidence:

1. **Gated mainloop header** (`sm120_blockscaled_mma_gated_array_tma.hpp` line 16): *"satisfying the N%128 hardware constraint"*

2. **Runtime validation** (`core.py` line 426): *"Tiles with N < 128 fail with TMA/shared memory errors"*

3. **FP4 alignment** (launcher line 400): `AlignmentB = 128` -- FP4 weights require 128-element alignment, which at 4 bits/element = 64 bytes. This is a CUTLASS/TMA descriptor constraint for block-scaled operands.

4. **6-plane design failure** (launcher lines 533-534): The earlier 6-plane approach (A, B, Aux, SFA, SFB, SFAux) exceeded SMEM limits for tiles **larger than** 64x64. At 64x64 it fit but violated the N%128 constraint. This forced the move to sequential SMEM reuse with 4 planes and N_tile=128.

---

## Section 3: Eliminating doActivation(Identity) [analysis DONE, implementation PENDING]

In the fused path, `doActivation` is called with `ActivationType::Identity` after the fused GEMM (`cutlass_fused_moe_kernels.cuh` line 3290-3298). Despite the "Identity" name, this kernel does significant work. Breaking down its operations:

| Step | Operation | What it does | Can it be eliminated? |
| --- | --- | --- | --- |
| 1 | Identity activation | No-op pass-through (`fn(x) = x`) | Already a no-op |
| 2 | FP8 quantization scale | `post_act_val = gate_act * quant_scale` (line 2333) | Must happen somewhere |
| 3 | Block-scaled FP8 quantization | `quantizePackedFPXValue()` converts BF16 to MXFP8 with block scaling (line 2338) | Must happen somewhere |
| 4 | Scale factor writeback | Writes block SFs to `fc2_fp4_act_flat` for FC2's TMA descriptor | Must happen somewhere |

**The identity activation is a no-op. Steps 2-4 are the real work: FP8 quantization with block scale factors for FC2 input.**

### Fusion opportunity (longer-term)

Steps 2-4 could in principle be fused into the GEMM epilogue via a custom CUTLASS Epilogue Visitor Tree (EVT) node. This would eliminate:

- One kernel launch (~2-4 us per layer x 24 layers)
- One HBM round-trip: the fused GEMM currently writes BF16 to `intermediate_result`, then `doActivation` reads BF16 back, applies quantization, and writes FP8 + scale factors

A custom EVT node would receive the SwiGLU output in registers, apply quantization scale, convert to FP8, compute block scale factors, and write FP8 + SFs directly. The SwiGLU BF16 intermediate never touches HBM.

### Complexity assessment

This is **nontrivial refactor territory**, not a straightforward EVT addition:

- The current SM120 epilogue outputs BF16 via standard `LinearCombination`. Changing `ElementD` to FP8 requires the epilogue to also emit block scale factor metadata (swizzled SF tensor), which the existing epilogue plumbing does not support.
- `quantizePackedFPXValue` uses warp-cooperative quantization (`cvt_warp_fp16_to_mxfp8`) with a specific thread-to-element mapping that may not align with the EVT's per-thread accumulator layout.
- The SF output tensor has a different layout and stride from the main output tensor, requiring a second TMA store or a separate global store path in the epilogue.
- The gated mainloop's SwiGLU is applied in `mma()` (in-register), so the epilogue receives post-SwiGLU FP32 accumulators. An EVT node would need to: (a) apply quant scale, (b) convert to FP8, (c) compute block max for SF, (d) write both FP8 data and SF metadata -- all within the epilogue's per-thread store loop.

**Priority**: Lower than Sections 4 (duplicated metadata), 5 (A-operand interleaving), and 6 (pipeline bubble). Pursue after the mainloop and metadata fixes have been validated.

---

## Section 4: Strategy for Supporting Kernels [DONE]

Complete kernel inventory for the fused path, with measured times from `profile.sqlite`:

| Kernel | Measured Time | Purpose | Fusion Potential |
| --- | --- | --- | --- |
| `computeStridesTmaWarpSpecializedKernel` | - | Sets up TMA descriptors (problem shapes, strides, pointers, scale factors) for **both** FC1 and FC2 in a single kernel (1 thread/expert). Launched by `setupTmaWarpSpecializedInputs` (line 4200). | **Duplicated FC1 work** -- see below. |
| `computeStridesFusedActivationKernel` | ~1.3 us/call | Per-expert TMA descriptor setup for FC1 fused GEMM (1 thread/expert). Writes fused-specific layout (logical N = `inter_size`, physical weight N = `2*inter_size`). Launched inside `gemm1_fused()` (line 3224). | Low ROI -- already minimal. |
| `computeGatedPointersAndStrides` | ~0.635 us/call | Computes gate weight pointers, scale factor pointers, aux output pointers/strides for the gated mainloop. Launched per fused GEMM (launcher line 1592). | Low ROI -- ~0.11% of fused kernel time. Not a regression source. |
| `doActivation(Identity)` | See Section 3 | FP8 quantization + block scale factors for FC2 input | **Primary target** -- fuse into GEMM epilogue (Section 3) |
| `expandInputRowsKernel` | - | Token expansion/permutation before FC1 | Required by both paths. Cannot be eliminated. |
| `finalizeMoeRoutingKernel` | - | Output unpermutation/reduction after FC2 | Required by both paths. Cannot be eliminated. |

### Duplicated FC1 metadata setup [DONE]

**Status**: Fixed. The fused path now calls `setupGemm2TmaInputForFused()` (line 4644) which sets up only GEMM2 TMA descriptors, skipping all FC1 metadata work. The `gemm1_tma_ws_unused` discard pattern is gone from the fused path.

**Original problem** (kept for reference):

The fused path prepares FC1 TMA metadata **twice**:

1. `setupTmaWarpSpecializedInputs()` (line 4200) calls `computeStridesTmaWarpSpecializedKernel`, which writes problem shapes, strides, pointers, and scale factor setup for **both** FC1 (`layout_info1`) and FC2 (`layout_info2`). The FC1 output is immediately discarded:

```cpp
// cutlass_fused_moe_kernels.cuh:4200
auto [gemm1_tma_ws_unused, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(...);
(void)gemm1_tma_ws_unused;
```

2. `gemm1_fused()` (line 3224) then launches `computeStridesFusedActivationKernel` to write FC1 metadata again with the fused-specific layout (logical N = `inter_size`, physical weight N = `2 * inter_size`, no `swap_ab`).

The `computeStridesTmaWarpSpecializedKernel` sets up FC1 with the **unfused** layout (`fc1_out_size = 2 * inter_size`, `swap_ab` from config) which differs from the fused layout. All of that FC1 work (lines 1398-1401, 1426-1443, 1450, 1453-1457 of the kernel) is wasted.

**Fix** (**implemented**): Created `setupGemm2TmaInputForFused()` — a FC2-only variant of `setupTmaWarpSpecializedInputs` that only populates GEMM2 TMA descriptors. The fused path calls this instead, skipping FC1 problem shape writes, FC1 stride/pointer computation, FC1 scale factor setup, and the `fc1_out_size` / `swap_ab` / fusion config that only applies to the unfused FC1 path.

This is a low-risk change -- the kernel is ~1 thread/expert with negligible per-call cost (~1.3 us), but it runs once per MoE layer per decode step (24 layers). The host-side `setupTmaWarpSpecializedInputs` also performs an MXFP4 scale factor `cudaMemsetAsync` (line 4527-4528) covering `max(fc1_sf_offset, fc2_sf_offset)` which in the fused path only needs `fc2_sf_offset`. More importantly, eliminating the redundant FC1 setup removes a source of confusion in profiling and debugging, and is much easier than mainloop surgery.

**Note on `quantize_with_block_size`**: This is a standalone kernel in `quantization.cuh` (lines 193-318), but it is **not directly called** in the SM120 MoE path. Instead, `doActivationKernel` calls the device function `quantizePackedFPXValue()` inline, which performs the same block-scaled quantization logic. The standalone kernel is used elsewhere (e.g., standalone quantization APIs).

---

## Section 5: Root Cause of Doubled A-Operand Traffic [analysis DONE, fix PENDING]

### Observation

The `load()` method in `sm120_blockscaled_mma_gated_array_tma.hpp` (lines 688-706) issues TMA loads for the A operand (activations) and SFA (activation scale factors) in **every** iteration of the K loop, for both Phase 1 (`k < k_real`) and Phase 2 (`k >= k_real`):

```cpp
// Line 688-692: A + SFA loaded every iteration (same for both phases)
copy(params.base.tma_load_a.with(...), tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
copy(params.base.tma_load_sfa.with(...), tAgSFA(_, _, _, *k_tile_iter), tAsSFA(_, _, _, write_stage));
```

Since Phase 2 resets the K iterator to re-read the same K positions (line 678-680):

```cpp
if (k == k_real) {
    k_tile_iter.coord = k_start_coord;  // Reset to beginning of K range
}
```

...the A/SFA TMA load addresses in Phase 2 are identical to those in Phase 1. This doubles the global-memory load-path traffic for A/SFA.

### Hypothesis (requires Nsight memory counters to confirm)

Whether this doubled load-path traffic translates to doubled **HBM** traffic or is partially absorbed by the L2 cache depends on:

- The time gap between Phase 1 and Phase 2 accesses to the same A tiles
- L2 cache capacity and eviction pressure from concurrent B/SFB loads
- Whether the A tile working set fits in L2

For decode (M=1), the A operand is small (~hidden_size = ~5KB per K-tile) and likely L2-resident across phases. For prefill (M=2048+), the A operand is ~11 MB total and unlikely to remain in L2 across the full Phase 1 pass. This aligns with the larger prefill regression (-5 to -8%) vs decode regression (-1 to -2%).

**Required validation**: Nsight Compute `l2_global_load` and `dram__bytes_read` counters, comparing fused vs unfused for the same problem size, would confirm whether L2 absorbs the redundant loads.

### Current flow (two sequential passes)

```
Phase 1: for k in 0..k_real:  TMA_load A[k], B_linear[k] → accum
          (all k_real A tiles loaded)
Phase 2: for k in 0..k_real:  TMA_load A[k], B_gate[k]   → accum_gate
          (same k_real A tiles re-loaded)
```

### Proposed fix (per-K-tile interleaving)

```
for k in 0..k_real:
  TMA_load A[k] → SMEM_A
  TMA_load B_linear[k] → SMEM_B;  MMA(A[k], B_linear[k]) → accum
  TMA_load B_gate[k] → SMEM_B;    MMA(A[k], B_gate[k])   → accum_gate
  // A[k] stays in SMEM (guaranteed) or L2 (probable) for both phases
```

By interleaving at the K-tile granularity, A[k] is loaded once and used twice in immediate succession. This guarantees SMEM residency for the second use (same pipeline stage), eliminating the redundant global load entirely -- no L2 dependency.

### Implementation changes required

- Remove the K-iterator reset logic (lines 672-680) and the `k < k_real` / `k >= k_real` phase branching
- Remove the `k_tile_count *= 2` doubling at all 5 locations in `sm90_gemm_array_tma_warpspecialized_pingpong.hpp`
- Restructure `load()`: for each K-tile, issue A+SFA load once, then issue B_linear+SFB load, then B_gate+SFAux load (two B-loads per K-tile, one A-load)
- Restructure `mma()`: for each K-tile, perform MMA into `accum` (linear), then MMA into `accum_gate` (gate), using the same A data
- This doubles the work per pipeline stage but halves the total number of stages (back to `k_real`)

---

## Section 6: K-Iterator Pipeline Bubble (Hypothesis) [PENDING — resolved by Phase 2]

### Observation

The `mma()` method runs two complete sequential passes over the pipeline:

```cpp
// Line 881: Phase 1 -- consume k_real pipeline stages
run_mma_phase(k_real, accum);

// Line 889: Phase 2 -- consume k_real pipeline stages
FrgTensorC accum_gate;
clear(accum_gate);
run_mma_phase(k_real, accum_gate);
```

There is no explicit pipeline drain or barrier between Phases 1 and 2. `run_mma_phase` internally uses `pipeline.consumer_wait()` / `pipeline.consumer_release()` for each stage, so Phase 2 begins consuming the next available stage after Phase 1 finishes.

### Potential stall mechanism

In the standard (unfused) pingpong schedule, the producer and consumer warps overlap: while the consumer processes stage N, the producer loads stage N+2. With two sequential `run_mma_phase` calls, the transition point may create a window where:

- The consumer finishes Phase 1's last stage and needs Phase 2's first stage
- The producer may have already loaded Phase 2's first stage (it runs ahead in the pipeline), or may be waiting on pipeline credits

Whether this creates a measurable bubble depends on the pipeline depth and how far ahead the producer runs. **This remains a hypothesis** -- Nsight Compute warp stall metrics (`smsp__warps_issue_stalled_*`) at the phase boundary would confirm or refute it.

### Resolution via interleaving

With the per-K-tile interleaving proposed in Section 5, the two-phase structure disappears. Both accumulators are updated within each pipeline stage, and the standard pingpong schedule runs without any phase boundary. This eliminates the potential bubble regardless of whether it is currently significant.

---

## Section 7: Deeper Fusion -- Beyond FC1+SwiGLU [out of scope]

### Current data path

```
FC1 GEMM (FP8xFP4) → BF16 SwiGLU output → doActivation(Identity) → FP8 + block SFs → FC2 GEMM (FP8xFP4) → BF16
```

The FC2 kernel is **FP8xFP4 by construction** -- `ElementInputA = cutlass::float_e4m3_t` (launcher line 139). This is a hardware constraint on SM120: the block-scaled tensor core MMA instruction operates on FP8 activations x FP4 weights. There is no BF16xFP4 MMA on SM120.

### What "deeper fusion" would mean

Eliminating the FC1→FC2 boundary entirely would require keeping FC1's SwiGLU output in on-chip memory and feeding it directly to FC2 without going through HBM. This raises several challenges:

1. **FP8 quantization is required** -- FC2's MMA hardware requires FP8 input. The SwiGLU output is in FP32 (accumulator) or BF16 (after epilogue conversion). A quantization step must happen *somewhere*. Deep fusion would move this quantization from a separate kernel to an in-kernel conversion between FC1 and FC2, but it cannot be skipped.

2. **Intermediate size** -- For gpt-oss-120b, `intermediate_size = 2880` (from `config.json`). The SwiGLU output per token is `2880` elements. At BF16, that's `2880 * 2 = 5,760 bytes` per token. At FP8 (post-quantize), `2880 bytes` per token.

3. **Register/SMEM budget** -- For decode (M=1): 5,760 bytes of BF16 intermediate is feasible in SMEM (~6% of 101KB budget) but would need to coexist with FC2's own A/B/SF tile data. For prefill (M=128 tiles): `128 * 5760 = 720KB` -- far exceeds SMEM.

4. **Different GEMM shapes** -- FC1 has K=`hidden_size`, N=`inter_size`. FC2 has K=`inter_size`, N=`hidden_size`. These are transposed problems requiring different tile mappings and potentially different TMA descriptors.

5. **Grouped GEMM complication** -- Both FC1 and FC2 are grouped GEMMs over experts. A fused kernel would need to handle per-expert problem shapes for both GEMMs within a single kernel launch.

### Assessment

Full FC1+FC2 fusion into a single kernel is a research-grade undertaking that requires:
- A new CUTLASS kernel architecture (not a mainloop modification)
- In-kernel FP8 quantization between the two GEMM phases
- Tiled accumulation with intermediate spilling to SMEM
- Handling different M/N/K shapes for the two GEMMs

This is a longer-term direction. The immediate optimizations (Phase 1, Sections 5, 6) address the current regression without requiring a new kernel architecture.

---

## Implementation Phases

### Phase 1: Eliminate redundant FC1 metadata setup (low risk, easy) [DONE]

**Completed.** The fused path now calls `setupGemm2TmaInputForFused()` (line 4644 in `cutlass_fused_moe_kernels.cuh`), a FC2-only variant that skips all FC1 metadata work. The `gemm1_tma_ws_unused` discard pattern is eliminated from the fused path.

**What was done**:

1. Created `setupGemm2TmaInputForFused()` — a FC2-only variant of `setupTmaWarpSpecializedInputs` that only populates `layout_info2` / `gemm2_tma_ws_input`. Skips FC1 problem shape writes, FC1 stride/pointer computation, FC1 scale factor setup, and the `fc1_out_size` / `swap_ab` / fusion config.

2. Updated `runMoe()` fused branch to call the FC2-only variant.

**Files changed**:
- `cutlass_fused_moe_kernels.cuh`: Added `setupGemm2TmaInputForFused()`, updated fused branch in `runMoe()`

### Phase 2: Per-K-tile interleaving (medium risk, mainloop surgery) [PENDING]

Implement the A-operand interleaving described in Section 5. This is the primary optimization for eliminating the prefill regression and reducing decode regression.

### Phase 3: EVT epilogue fusion (higher risk, longer-term) [PENDING]

Fuse `doActivation(Identity)` into the GEMM epilogue as described in Section 3. Pursue after Phase 1 and Phase 2 are validated.

---

## Key Files Referenced

- `flashinfer/fused_moe/core.py` -- Tile selection, dispatch
- `flashinfer/csrc/.../sm120_blockscaled_mma_gated_array_tma.hpp` -- Gated mainloop (load, mma, SwiGLU)
- `flashinfer/3rdparty/cutlass/.../sm90_gemm_array_tma_warpspecialized_pingpong.hpp` -- k_tile_count doubling (5 sites)
- `flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` -- Kernel dispatch (gemm1_fused, doActivation)
- `flashinfer/csrc/.../moe_gemm_sm120_mixed_input_launcher.inl` -- Tile instantiation, computeGatedPointersAndStrides
