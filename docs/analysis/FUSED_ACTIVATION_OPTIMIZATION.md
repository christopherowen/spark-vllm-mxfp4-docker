# TMA Descriptor Caching: Implementation Analysis

Analysis of whether caching `cuTensorMapEncodeTiled` results would improve performance
in the FlashInfer CUTLASS MoE GEMM path on SM121 (GB10).

## Verdict: Not Worth Implementing (for CUDA-graph-enabled workloads)

Profiling data from `compare_20260209_0200` (CUDA graphs active for decode) shows
`cuTensorMapEncodeTiled` costs ~185 ns per call, totaling **5.6 ms across 30,240 calls
in a 250-second profiled run (0.002% of runtime)**. Under graph replay, TMA descriptor
encoding only happens during graph capture, not on every forward pass. The savings from
caching are negligible in this regime.

In eager mode (no CUDA graphs), TMA encoding would happen on every forward pass and the
call count would scale with the number of MoE invocations. The overhead would still be
small in absolute terms (~185 ns/call) but should be re-evaluated if eager mode becomes
performance-critical.

---

## Profiling Evidence

Source: `results/compare_20260209_0200/fused/profile.sqlite` and `unfused/profile.sqlite`

| Metric | Fused | Unfused |
|--------|-------|---------|
| cuTensorMapEncodeTiled calls | 30,240 | 21,600 |
| Total time | 5.6 ms | 4.8 ms |
| Avg per call | 185 ns | 224 ns |
| % of profiled runtime | 0.002% | 0.002% |

For comparison, a single `cudaLaunchKernel` call averages 10-22 us — roughly 100x more
expensive than a `cuTensorMapEncodeTiled` call.

Note: `cudaGetDriverEntryPointByVersion_v12050` call counts match `cuTensorMapEncodeTiled`
exactly (30,240 / 21,600) in both profiles, indicating a driver-entry lookup per encode
call. This is additional per-call overhead but equally tiny.

**CUDA graph effect on call counts**: The stride computation kernel
`computeStridesGemm2Only` has 394,128 calls (one per MoE invocation), while
`cuTensorMapEncodeTiled` has only 30,240 calls. This ~13x ratio confirms that TMA
encoding does NOT happen on every forward pass — it happens during CUDA graph capture
phases only, and graph replay skips it entirely.

---

## Current Behavior

### Non-gated (unfused) path

Each GEMM (FC1, FC2) calls `to_underlying_arguments()` in the CUTLASS collective mainloop,
which calls `make_tma_copy()` 4 times (A, B, SFA, SFB) to create template TMA descriptors
via `cuTensorMapEncodeTiled`. These are filled with **dummy placeholder values** (null
pointers, tile-sized shapes). The actual per-expert pointers and dimensions are patched
on-device using PTX `tensormap.replace.*` instructions inside the kernel.

Per forward pass (outside CUDA graph): 4 descriptors x 2 GEMMs = 8 encode calls per
MoE layer.

### Gated (fused) path

The gated kernel's `to_underlying_arguments()` builds base params **twice**
(`sm120_blockscaled_mma_gated_array_tma.hpp`, lines 328 and 337):

1. First `Base::to_underlying_arguments()` call builds the primary mainloop params
   (A, B_linear, SFA, SFB) — 4 TMA descriptors.
2. Second `Base::to_underlying_arguments()` call builds aux params for the gate path.
   This creates 4 more TMA descriptors (A, B_gate, SFA_gate, SFB_gate), but only the
   B and SFB pieces are kept (lines 340-345). The A and SFA descriptors from this
   second call are discarded — redundant work.

Per forward pass (outside CUDA graph): up to 8 descriptors for gated GEMM1 + 4 for
GEMM2 = up to 12 encode calls per MoE layer. The discarded A/SFA descriptors in the
aux build are pure waste, though the absolute cost is negligible (~370 ns).

The template descriptors encode only compile-time-constant structural information (data
type, swizzle pattern, box shape) that never changes between calls.

---

## Implementation Approaches Considered

### Approach A: Cache Full Params Struct

Cache the CUTLASS `GemmKernel::Params` struct (which contains TMA descriptors) after the
first `gemm.initialize()` call. On subsequent calls, copy cached Params and update only
workspace-dependent fields.

- Use `static` variable inside the template launcher function (auto-keyed by template args)
- First call: full `gemm.initialize()`, cache result
- Subsequent calls: memcpy cached Params, patch workspace pointers

**Effort**: ~1 week. Main work is reverse-engineering which Params fields reference the
workspace buffer (deeply nested CUTLASS template types, not documented).

**Risk**: Medium. Missing a workspace-dependent field causes silent corruption. The Params
struct is not designed for partial updates.

**Savings with CUDA graphs**: <0.01% (encoding already skipped during graph replay).
**Savings in eager mode**: Eliminates ~12 encode calls per MoE layer (~2.2 us/layer,
~132 us for 60 layers). Negligible.

### Approach B: Cache TMA Descriptor Objects Only

Extract the `tma_load_*` objects from Params after the first call. On subsequent calls,
still run full `to_underlying_arguments()` but inject cached TMA objects, skipping the
`make_tma_copy()` calls.

- Requires modifying `to_underlying_arguments()` to accept pre-built TMA descriptors
- Or wrapping it with a conditional path

**Effort**: ~1.5 weeks. Requires modifying upstream CUTLASS headers
(`sm120_blockscaled_mma_array_tma.hpp`).

**Risk**: Medium. Touches CUTLASS internals shared with upstream. Needs to handle both
gated and non-gated variants.

**Savings**: Same as Approach A. Also eliminates the redundant A/SFA descriptors in the
gated aux build, but this is ~370 ns.

### Approach C: Upstream-Style Refactor

Split `to_underlying_arguments()` into `create_tma_templates()` (once) and
`update_dynamic_arguments()` (per-call). Clean separation of static vs dynamic state.

**Effort**: ~3-4 weeks. Touches CUTLASS core across multiple collective mainloop
specializations (SM90, SM100, SM120, gated, non-gated).

**Risk**: High. Large surface area, affects all architectures.

**Savings**: Same as A/B in practice.

---

## Precedent

There is **no existing TMA descriptor caching** anywhere in CUTLASS or FlashInfer:

- CUTLASS's `to_underlying_arguments()` creates fresh TMA descriptors every call
- CUTLASS's only "reuse" pattern is `gemm.run()` which re-launches with the same stored
  `params_` — but FlashInfer stack-allocates a fresh `Gemm gemm` each call
- FlashInfer's attention kernels (Hopper, Blackwell, cuDNN) all create TMA descriptors
  fresh per launch
- FlashInfer's `plan()`/`run()` two-phase pattern for attention caches tile schedules
  and index layouts, but NOT TMA descriptors
- The MoE code has a `// TODO Some of this setup could be cached` comment but no
  implementation

---

## Conclusion

With CUDA graphs enabled (the production configuration), TMA encoding is skipped during
graph replay and only occurs during capture. The 30,240 calls in a 250-second profiled
run total 5.6 ms — well below the threshold where caching would matter.

In eager mode, encoding happens per forward pass but each call costs ~185 ns. Even at
12 calls per MoE layer across 60 layers, the total is ~132 us per forward pass — far
less than a single decode step (~16.7 ms).

All three implementation approaches require 1-4 weeks of effort with medium-to-high risk
for negligible savings. The redundant A/SFA descriptors in the gated aux build are a
minor code cleanliness issue, not a performance issue.

**Revisit if**: TMA descriptor encoding cost increases on future architectures, eager mode
becomes the primary execution path, or the per-call cost rises above ~1 us.
