# SM120 MoE `swap_ab` crash: “Warp Illegal Instruction Parameter” / illegal instruction

## TL;DR

- **Non-swap tiles** (e.g. `(64, 128)`) work.
- **Swap tiles** (e.g. `(32, 128)`) used to trap in the **SM120 block-scaled TMA mainloop**.
- Root cause: **`swap_ab` mismatch between the compiled JIT module (`-DSWAP_AB=1`) and runtime inputs**
  used during workspace sizing / stride generation (`tma_inputs.swap_ab` was `false` while the compiled
  launcher was swap-mode).
- Fix: **force runtime swap mode to match the compiled module**, and make the transposed epilogue tile
  satisfy divisibility when `TILE_N_VAL=16`.
- Result: swap tiles like `(32,128)` and `(16,128)` now run successfully via
  `python3 /workspace/scripts/debug/debug_swap_ab.py`.

---

## Context (what was already confirmed)

- **Good (no swap)**:
  - When logical token dimension \(M \ge 64\), SM120 MoE works.
  - Example: tile `(64, 128)` succeeds.
- **Bad (swap)**:
  - When logical token dimension \(M < 64\), SM120 MoE crashes.
  - Example: tile `(32, 128)` fails with “Warp Illegal Instruction Parameter” / illegal instruction.
- Crash occurs during kernel execution in the SM120 **block-scaled mainloop** (TMA / tcgen path).
- The `swap_ab` implementation existed for a while (since commit `dc3bc8ac`) but appears to have never actually worked for SM120 swap tiles.

---

## Repro

### Why run in the container

Host Python in this repo does **not** have `torch`. Use the dev container.

### Minimal repro script

In `vllm-dev`, the repo is mounted under `/workspace/scripts/...`:

- `python3 /workspace/scripts/debug/debug_swap_ab.py`

### Make the failure synchronous (strongly recommended)

This avoids “kernel traps, error shows up later”.

```bash
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export TLLM_LOG_LEVEL=DEBUG
export TLLM_SM120_MOE_DUMP_TMA=1
export CUDA_LAUNCH_BLOCKING=1
python3 /workspace/scripts/debug/debug_swap_ab.py
```

---

## Tooling: compute-sanitizer confirmation

We ran:

```bash
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export TLLM_LOG_LEVEL=DEBUG
export TLLM_SM120_MOE_DUMP_TMA=1
export CUDA_LAUNCH_BLOCKING=1

compute-sanitizer --target-processes all --tool memcheck --show-backtrace yes \
  python3 /workspace/scripts/debug/debug_swap_ab.py
```

Key output:

- `========= Illegal instruction`
- faulting kernel is a CUTLASS `device_kernel<GemmUniversal<... MainloopSm120ArrayTmaWarpSpecializedBlockScaled ...>>`
- the printed kernel type includes tile shape with **`(128,32,128)`** (consistent with the SM120 transposed/swap kernel definition: virtual `M=128`, `N=logical_tile_m`, `K=128`).

This confirms:

- This is **not** just a host-side misconfiguration.
- The GPU is encountering a **real illegal instruction** inside the SM120 block-scaled TMA mainloop.

---

## Instrumentation we added (and what it showed)

### 1) Add a “dump TMA/stride/layout” switch in the SM120 launcher

File:

- `flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl`

Env var:

- `TLLM_SM120_MOE_DUMP_TMA=1`

Also requires:

- `TLLM_LOG_LEVEL=DEBUG` (otherwise `TLLM_LOG_DEBUG` doesn’t print).

What we dump (group 0):

- `problem_shape`
- `strideA`, `strideB`, `strideD`
- `layoutSFA`, `layoutSFB`
- type-size relationships:
  - `Stride*` sizes and `std::is_same_v` checks
  - `LayoutSFA/LayoutSFB` pointer-typed vs underlying MXFPX `LayoutSF` size

### 2) Fix a critical SM120 launcher bug: pointer-vs-value for grouped strides/layouts

We discovered that in this SM120 grouped GEMM instantiation, the kernel’s `StrideA/StrideB/StrideD` and `LayoutSFA/LayoutSFB` are **pointer types** (arrays) for grouped mode, not scalar values.

The original SM120 launcher incorrectly did:

- “dereference first element” style passing (`*reinterpret_cast<StrideA const*>(stride_A)`)

That can cause:

- segfaults / invalid reads in host code
- or passing the wrong thing to CUTLASS (value instead of array pointer), producing invalid descriptors

We updated the launcher to:

- pass **arrays** when `Stride*`/`LayoutSF*` are pointer-typed
- pass values only if the alias is not a pointer type (handle both via `if constexpr (std::is_pointer_v<...>)`).

This made behavior consistent and got us to a clean, reproducible device trap.

### 3) What the dump showed right before the illegal instruction

In swap mode (example `(32, 128)`), the first “good” dump before the trap printed:

- `group0 problem_shape=(32,512,256)`
- `group0 strideA=(256,_1,_0)`
- `group0 strideB=(256,_1,_0)`
- `group0 strideD=(_1,512,_0)`
- `layoutSFA/layoutSFB` are populated and non-zero

Interpretation / why this is suspicious:

- In the SM120 transposed/swap kernel, `LayoutB` is defined as **ColumnMajor** for the activation operand in that kernel namespace.
- Yet `strideB` looks identical to `strideA` (and “row-major-ish”: leading dimension resembles `K=256`).
- This strongly suggests **the stride values are not consistent with the kernel’s intended operand layout** (even if the C++ types match).

That is exactly the kind of mismatch that can produce an invalid TMA descriptor and trigger a tcgen/TMA illegal-instruction trap.

### 4) Stride sanity check: expected vs actual (fixed + clarified)

We added a debug-only stride sanity check in the SM120 launcher (guarded by `TLLM_SM120_MOE_DUMP_TMA=1`)
that computes the **expected** 2D packed-stride components from the group0 `(M,N,K)` and compares them to
the **actual** stored packed strides.

Important clarification: CUTLASS’ `TagToStrideB` maps B strides in **(N,K)** mode order (see
`cutlass/detail/layout.hpp`), so packed ColumnMajor B has a `(K,1)` packed stride, not `(1,K)`.

After fixing `swap_ab` consistency, for swap mode with `group0 problem_shape=(512,32,256)` it prints:

- `A(RowMajor MxK) stride expected=(256,1) actual=(256,1) OK`
- `B(ColumnMajor NxK) stride expected=(256,1) actual=(256,1) OK`
- `D_T(ColumnMajor MxN) stride expected=(1,512) actual=(1,512) OK`

So the previously-reported “StrideB mismatch” was actually:

- **expectation mismatch** (B stride mode order is (N,K), so ColumnMajor is `(K,1)`), and
- **compiled-vs-runtime `swap_ab` mismatch**, which is what made shapes/strides inconsistent and led to traps.

---

## “Is swapping happening in Python?”

Not as an actual data transpose/copy in Python.

- Python selects the tile and drives JIT compilation with `-DSWAP_AB=1`.
- The *actual swap* is implemented in C++ by rewiring:
  - `ptr_A/ptr_B`
  - `stride_A/stride_B`
  - `sf_A/sf_B` and `sf_stride_A/sf_stride_B`

This is the same *conceptual* pattern as SM100 “swap_ab”: **reinterpret the GEMM** rather than physically transposing tensors.

Why SM100 works but SM120 doesn’t:

- SM100’s swap kernel + its stride/TMA construction are aligned in conventions.
- SM120’s transposed kernel definition (new namespace) appears to be mismatched with the existing stride-fill conventions, so we get “type-correct, semantically wrong” descriptors that hardware rejects.

---

## Root cause (most likely deficiency)

**Compiled-vs-runtime `swap_ab` mismatch**:

- The SM120/121 JIT compiles one module per logical tile, with `-DSWAP_AB={0,1}` baked in.
- The fused-MoE binding defaults to the first heuristic profile when tactics are `[-1, -1]`, which often has
  `swap_ab=false`.
- That means workspace sizing and/or stride/problem-shape generation can run with runtime `swap_ab=false`
  even though the compiled module is swap-mode (`SWAP_AB=1`).

This mismatch can produce inconsistent shapes/strides vs the compiled kernel’s expectation and trigger:

- `compute-sanitizer`: `Illegal instruction` in `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`

### Fix summary (what changed)

- `flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl`
  - Workspace-size query path forces `tma_inputs.swap_ab = SWAP_AB`
  - Execution path requires `tma_inputs.swap_ab == SWAP_AB`
  - Transposed epilogue tile uses `kEpiTileN = min(TILE_N_VAL, 32)` so `TILE_N_VAL=16` compiles
- `flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
  - In `setupTmaWarpSpecializedInputs`, force `gemm*_tma_ws_input.swap_ab = SWAP_AB` (compiled mode)
- `scripts/debug/debug_swap_ab.py`
  - Handle list return value from `cutlass_fused_moe`
  - Use only supported swap tiles in the test list

---

## SM100 vs SM120: `StrideB` / B-layout expectation mismatch (key observation)

This is the conceptual difference that explains why a `(K,1)`-style packed stride can be “fine” on the
SM100 swap path but is wrong for the SM120 transposed swap path.

### SM120 swap/transposed kernel (B is ColumnMajor)

In the SM120 transposed namespace, **B is FP8 activations** and is explicitly declared **ColumnMajor**:

```208:283:/home/swank/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl
// Transpose GEMM: D^T = W^T @ A^T
// - A = FP4 weights
// - B = FP8 activations
// - Output layout: ColumnMajor
#define DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE(NAMESPACE_NAME, TILE_N_VAL)                       \
namespace NAMESPACE_NAME {                                                                        \
  /* Layouts: A (Weight) row-major, B (Act) column-major, C/D column-major for transpose */       \
  using LayoutA = cutlass::layout::RowMajor;                                                      \
  using LayoutB = cutlass::layout::ColumnMajor;                                                   \
  using LayoutC = cutlass::layout::ColumnMajor;                                                   \
  /* ... */                                                                                       \
  using CollectiveMainloop =                                                                      \
      typename cutlass::gemm::collective::CollectiveBuilder<                                      \
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB, \
          ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCount,                        \
          cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;                 \
}
```

So in swap mode for SM120, the kernel’s **B operand is column-major**, but note CUTLASS’ B stride types are in
**(N,K)** mode order, so the packed stride pattern is **\((K, 1)\)** (see `cutlass/detail/layout.hpp`).

### SM100 swap path (manual swap + transpose of input layouts)

In the SM100-era mixed-input launcher (used for the “swap_ab + mixed input” style kernels), the code explicitly
states it “manually swaps and transposes” and then uses **transposed layout tags** for the inputs:

```95:160:/home/swank/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl
// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// ... Mixed Input mainloop uses transposed layouts ...
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
    ArchTag, OperatorClass, cute::tuple<ElementB, ElementScalePacked>, LayoutB_Transpose*,
    AlignmentB, ElementA, LayoutA_Transpose*, AlignmentA, ElementAccumulator, TileShape,
    ClusterShape,
    /* ... */, KernelSchedule>::CollectiveOp;
```

And it wires the grouped stride arrays directly:

```190:203:/home/swank/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl
arguments = Args{
    /* ... */
    {reinterpret_cast<ElementB const**>(hopper_inputs.ptr_weight),
     reinterpret_cast<StrideB*>(hopper_inputs.stride_weight),
     reinterpret_cast<ElementA const**>(hopper_inputs.ptr_act),
     reinterpret_cast<StrideA*>(hopper_inputs.stride_act),
     /* ... */},
    {/* epilogue ... */}};
```

**Observation (updated):** the earlier “StrideB must be \((1,K)\)” hypothesis was incorrect for CUTLASS’ B stride
types (they are in (N,K) mode order, so ColumnMajor B is \((K,1)\)).

The actual root cause was **compiled-vs-runtime `swap_ab` mismatch** (compiled `SWAP_AB=1`, runtime inputs
often had `swap_ab=false` during workspace sizing / stride generation).

## Next steps (recommended order)

### 1) Make the trap attributable (already done)

Always rerun with:

- `CUDA_LAUNCH_BLOCKING=1`

### 2) Keep runtime sanity checks for swap mode (in the launcher)

✅ Implemented: debug-only stride sanity check (guarded by `TLLM_SM120_MOE_DUMP_TMA=1`) that prints
expected vs actual packed strides for A/B/D in swap mode.

### 3) Exact code locations for the fix

The fixes that made `(32,128)` and `(16,128)` swap tiles run:

- `flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl`
  - Workspace-size path forces `tma_inputs.swap_ab = SWAP_AB`
  - Execution path requires `tma_inputs.swap_ab == SWAP_AB`
  - Transposed mode epilogue tile uses `kEpiTileN = min(TILE_N_VAL, 32)` so `TILE_N_VAL=16` compiles
- `flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
  - In `setupTmaWarpSpecializedInputs`, force `gemm*_tma_ws_input.swap_ab = SWAP_AB` (compiled mode)

---

## Appendix: stride construction helper (reference)

This device helper fills `layout_info.stride_act` / `layout_info.stride_weight` based on runtime `swap_ab`:

This is the device helper that fills `layout_info.stride_act` / `layout_info.stride_weight` when `swap_ab` is enabled:

```1156:1193:/home/swank/projects/flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int gemm_m, int gemm_n, int gemm_k,
    int64_t out_idx) {
  if (layout_info.swap_ab) {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  } else {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  }
  // ... D stride fill ...
}
```

If B is column-major in the **transposed** SM120 kernel, but this path produces a `(K,1)`-style stride for the buffer later used as B, this is where the mismatch is created.

### B) Shape vs stride call-site (secondary suspect)

The grouped problem shape is written with swapped `(M,N)` in swap mode, but the stride helper is called with `(gemm_m, gemm*_n, gemm*_k)` regardless:

```1272:1333:/home/swank/projects/flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
  auto const gemm_m = num_tokens_to_expert;

  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info1.swap_ab ? gemm1_n : gemm_m, layout_info1.swap_ab ? gemm_m : gemm1_n,
          gemm1_k);
  // ...
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);
```

This is a classic place to get “right types, wrong values”: the kernel’s problem shape swaps `(M,N)` but the stride builder may still be constructing strides with the unswapped interpretation.

### C) SM120 launcher operand wiring (verify what becomes B)

This is the host-side wiring that decides which stride array becomes **A** vs **B** based on `swap_ab`:

```375:397:/home/swank/projects/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl
  // Operand wiring based on swap_ab (set above from compile-time SWAP_AB):
  // - swap_ab=false => A=act, B=weight, SFA=act_sf, SFB=weight_sf
  // - swap_ab=true  => A=weight, B=act, SFA=weight_sf, SFB=act_sf
  void const** ptr_A = tma_inputs.swap_ab ? tma_inputs.ptr_weight : tma_inputs.ptr_act;
  void const** ptr_B = tma_inputs.swap_ab ? tma_inputs.ptr_act : tma_inputs.ptr_weight;
  void* stride_A = tma_inputs.swap_ab ? tma_inputs.stride_weight : tma_inputs.stride_act;
  void* stride_B = tma_inputs.swap_ab ? tma_inputs.stride_act : tma_inputs.stride_weight;
  // ... SF wiring ...
```

Given the sanity check output, the stride stored in `tma_inputs.stride_act[0]` is what becomes **B** in swap mode, and it currently encodes a row-major-style `(K,1)` stride even though the transposed kernel’s **B layout is ColumnMajor**.

### 4) Use cuda-gdb to map the exact PC → instruction

If needed after stride checks:

- `CUDA_DEVICE_WAITS_ON_EXCEPTION=1`
- `cuda-gdb --args python3 /workspace/scripts/debug/debug_swap_ab.py`

Then map the trapped PC into SASS for the cached JIT `.so`.

### 5) Disassemble the cached JIT artifact to inspect the exact trapped instruction

The memcheck backtrace already prints the `.so` name (example):

- `/root/.cache/flashinfer/0.6.1/121f/cached_ops/fused_moe_120_M32N128_mxfp4min/fused_moe_120_M32N128_mxfp4min.so`

Use `cuobjdump`/`nvdisasm` to find the SASS around the trapping instruction in the SM120 blockscaled mainloop.

---

## Appendix: Known-good debug knobs

- `TLLM_LOG_LEVEL=DEBUG` — enables `TLLM_LOG_DEBUG(...)`
- `TLLM_SM120_MOE_DUMP_TMA=1` — prints group0 stride/layout objects from the SM120 launcher
- `CUDA_LAUNCH_BLOCKING=1` — makes the error surface at the right launch
- `compute-sanitizer --tool memcheck` — confirms device illegal instruction and prints kernel type/backtrace

