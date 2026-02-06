# Epilogue study (SM90/SM120/SM100): tile selection, C usage, and shared-memory “need”

## TL;DR

- CUTLASS “epilogue” is **not just a store**: it is a pipelined stage that may
  - optionally **load a source matrix `C`**,
  - apply fused math (\(D = \alpha \cdot Acc + \beta \cdot C\), bias/scales/scatter variants),
  - stage through **shared memory** and issue **TMA stores**.
- The epilogue’s subtile shape `EpilogueTile=(EPI_TILE_M,EPI_TILE_N)` is **per-iteration granularity**, not necessarily the CTA tile.
  - It is chosen to balance **iteration count**, **SMEM footprint**, **alignment/vectorization**, and **pipeline stage counts**.
- **SM90/SM120 TMA warp-specialized epilogues require**:
  - `EPI_TILE_M | CTA_M` and `EPI_TILE_N | CTA_N` (compile-time `static_assert`).
- **SM100 epilogues are remainder-capable** (explicit “residue” predication) and their builder falls back to `EPI_TILE_N=CTA_N` if a preferred value wouldn’t divide.
- In our SM12x MoE MXFP4 launcher, **C is not used at runtime**:
  - `ptr_C=nullptr`, `stride_C=nullptr`, `beta=0.0`.

---

## What “C” is (and what it is *not*)

In CUTLASS GEMM terms:

- `A @ B` produces an **accumulator** (internal).
- The epilogue writes the final output `D` and may optionally read a “source” matrix `C`.

Conceptually, the default linear-combination epilogue is:

\[
D = \alpha \cdot Acc + \beta \cdot C
\]

So:

- **`C` is the residual/source addend** used by the epilogue.
- `C` is **not** the activation matrix, not the weights, not the block-scale factors.

Whether `C` is used affects:

- **global memory reads** (C load) and associated latency,
- potential **shared-memory staging** for C (depending on epilogue schedule),
- sometimes **stage-count heuristics** (builders choose `StagesC/StagesD` differently when `C` is present vs void).

---

## Are we using `C` for SM121 MoE?

### 1) SM120 MXFP4 MoE launcher (used for SM12x path)

In `flashinfer/.../moe_gemm_sm120_mixed_input_launcher.inl`, the epilogue arguments are wired as:

- `ptr_C = nullptr`
- `stride_C = nullptr`
- `alpha = 1.0`
- `beta = 0.0`

So **we do not use C at runtime** for MoE.

Interpretation:

- The MoE GEMM is a “pure” `D = Acc` writeout.
- This is typically correct for MoE FC1/FC2 (no residual add at the GEMM).

### 2) Note: compile-time `ElementC` may still be non-void

Some instantiations still set `ElementC` to a real type (e.g. BF16) but pass `ptr_C=null` and `beta=0`.
That keeps the epilogue *capable* of C, but effectively disables the load path at runtime.

Tradeoff:

- Capability can increase template reuse / simplify code generation,
- but may influence epilogue builder heuristics and smem carveouts.

---

## What goes into “epilogue shared memory” (validated from source)

For TMA warp-specialized epilogues, the shared-memory budget is dominated by:

- **C staging buffers** (if source-supported and not reusing)
- **D staging buffers** (for TMA store)
- optional **C/D smem reuse** (allocate one max-sized buffer instead of separate C and D)
- **pipeline storage** (transaction barriers / stages)
- **tensormap descriptor storage**
- **fusion callback storage** (`FusionCallbacks::SharedStorage`)

### SM100: explicit per-stage-bit accounting (good mental model)

SM100 computes:

- `StageCBits`, `StageDBits` based on `SmemLayoutStage{C,D}`
- `MaxStageBits = max(StageCBits, StageDBits)`
- if `ReuseSmemC`, pad each stage to `MaxStageBits` so one buffer can serve C or D safely.

It then creates:

- `SmemLayoutC`: `StagesC` stages
- `SmemLayoutD`: `StagesD` stages, or `StagesC` stages when reusing

Storage selection:

- if C disabled: allocate only D (`CollectiveStorageWithoutC`)
- if C enabled and no reuse: allocate C + D (`CollectiveStorageWithC`)
- if reuse enabled: allocate a single max-aligned union (`CollectiveStorageReuseC`)

### SM90/SM120: same idea; different layout construction

SM90 and SM120 use a similar `CollectiveStorage{WithC,WithoutC,ReuseC}` scheme and choose stage counts via builder heuristics.
SM120 intentionally uses **smaller stage counts** to fit SMEM constraints.

---

## “Perfect” epilogue memory allocation as a function of need

Define epilogue “need” inputs:

- **needC**: does the epilogue need to read C? (e.g. \(\beta \ne 0\) and `ptr_C != nullptr`)
- **needD**: almost always true (we must store D)
- **reuseSmem**: can we reuse a shared buffer for C and D?
  - requires schedule- and layout-specific compatibility checks (`support_smem_reuse`)
- **StagesC / StagesD**: dispatch-policy-selected stage counts (depend on `EpiTiles`, data types, schedule, and arch)
- **EpilogueTile**: affects both per-stage sizes and `EpiTiles`

Then the “perfect” allocation is:

- **If needC is false**: allocate **no C buffers**, only D buffers + overheads.
- **If needC is true and reuseSmem is false**: allocate **C buffers (StagesC)** + **D buffers (StagesD)** + overheads.
- **If needC is true and reuseSmem is true**: allocate **one max-sized buffer** that can serve both C and D across the required stage schedule + overheads.

Practical consequences:

- Carrying a non-void `ElementC` when you don’t need it can inflate the carved-out SMEM budget in some configurations.
- Conversely, disabling C can allow more mainloop stages (more overlap) or enable larger epilogue tiles without exceeding SMEM.

---

## Why `EPI_TILE_N` isn’t always `CTA_N`

`EpilogueTile` is the **subtile granularity per epilogue iteration**.
Choosing it smaller than the CTA tile can:

- reduce smem staging footprint per stage,
- reduce register pressure / instruction size in heavy fusions,
- meet TMA alignment/vectorization constraints with fewer tradeoffs,
- improve overlap between epilogue work and mainloop/pipeline.

Builders often pick a “performant” `EPI_TILE_N` (32 or 64 are common) rather than always taking `CTA_N`.

SM100’s builder makes this explicit: it picks a preferred `N_tmp` and falls back to `CTA_N` if it wouldn’t divide.

---

## Why `(128,16)` fails on SM120 today

For SM90/SM120 TMA warp-specialized epilogues:

- `EPI_TILE_N` must divide `CTA_N`.

So if a builder selects `EPI_TILE_N=32` while `CTA_N=16`, the kernel fails to compile with:

- `static assertion failed with "EPI_TILE_N must divide CTA_N"`

This was the core issue for the SM12x MoE “small-N” tile investigation (`(tile_mn)=(128,16)`).

### Update (after forcing `EPI_TILE_N=16`)

After forcing `EPI_TILE_N=16` (so the divisibility `static_assert` no longer fires), `(128,16)` still fails at runtime.
`cuda-gdb` shows a **GPU trap** (`CUDA_EXCEPTION_27 Warp Illegal Instruction Parameter`) inside the **SM120 block-scaled TMA mainloop** kernel family.
So the epilogue divisibility constraint was necessary to fix for buildability, but it is **not sufficient** for correctness/execution stability.

---

## Action items for “small CTA_N=16” support

- Ensure that for `CTA_N=16` the chosen epilogue tile satisfies:
  - `EPI_TILE_N = 16` (or another divisor of 16).
- Consider whether it is beneficial to instantiate an epilogue with `ElementC=void` for MoE (since we pass `ptr_C=null` and `beta=0` anyway), to reduce SMEM carveout and avoid “C-present” heuristics.

