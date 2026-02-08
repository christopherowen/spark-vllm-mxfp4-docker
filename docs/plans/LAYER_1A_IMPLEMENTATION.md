# Layer 1A Implementation: Gated FC1 with Fused SwigluBias

## Status: IMPLEMENTED - Pending Hardware Test (2026-02-06)

The gated FC1 kernel is code-complete. Key properties:

- **SwigluBias** applied inline in the gated mainloop (not a custom epilogue)
- **CUDA-graph safe**: no `cudaStreamSynchronize`, no `cudaMemcpy` in production path
- **Device-pointer SwigluBias params**: alpha/beta/limit read on-device, not copied to host
- **Upstream pointer computation**: gated arrays pre-computed by existing stride kernel
- **Default tile**: 64×64×128 (fits SM121 SMEM with 6 TMA planes)

### Current Data Flow (Gated FC1 Path)

```
FC1 Gated GEMM (6-plane TMA: A, B, Aux, SFA, SFB, SFAux)
    → dual-accumulator mma(): A @ W_linear, A @ W_gate
    → inline SwigluBias: gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
    → standard epilogue: BF16 output to intermediate_result
    → doActivation(Identity): BF16 → FP8 + scale factors for FC2
    → FC2 GEMM (standard path, unchanged)
```

### Remaining Verification

| Item | Status | Notes |
|------|--------|-------|
| JIT compile on SM121 | Pending | Previously compiled successfully (2026-02-01) |
| Runtime launch (64×64×128 tile) | Pending | Was "illegal instruction" pre-refactoring; may be resolved |
| Numerical accuracy vs baseline | Pending | Compare fused vs `doGatedActivation` |
| nsys: doGatedActivation eliminated | Pending | |
| CUDA graph capture/replay | Pending | |
| End-to-end benchmark | Pending | |

### Known Risk: 64×64×128 TMA Descriptor

The 64×64×128 tile previously hit "illegal instruction" at runtime (before the gated pointer refactoring). The crash was at `UTMALDG.4D` for plane 3 (SFB), the same TMA plane that fails for CTA_N=16 non-gated tiles. Since then, the pointer computation was completely refactored (moved into the upstream stride kernel, eliminated the separate `computeGatedPointersAndStrides` kernel). Hardware test needed to determine if this is resolved.

If the tile still fails, fallback options:
1. **B/Aux SMEM sharing** - load B and Aux sequentially into the same buffer (saves ~33KB SMEM, enables 64×128×128 tile with N=128)
2. **Debug SFB TMA for small N** - investigate the tensormap descriptor issue for `SFB_NumBlocks=1`

### SMEM Budget

| Tile Size | SMEM (bytes) | Device Max | Status |
|-----------|-------------|------------|--------|
| 128×128×128 | 129,024 | 101,376 | Over by 28KB |
| 64×128×128 | 112,640 | 101,376 | Over by 11KB |
| **64×64×128** | **71,680** | 101,376 | **FITS (29KB margin)** - default tile |
| 128×128×64 | N/A | - | TMA SF layout mismatch (K must be 128) |

**64×64×128 SMEM Breakdown** (6 TMA planes, 2 pipeline stages):
```
smem_A          |      16,384 B | FP8 activations (64×128×2 stages)
smem_B          |      16,384 B | FP4 linear weights (128×64×2 stages, unpacked to 1B)
smem_Aux        |      16,384 B | FP4 gate weights (128×64×2 stages, unpacked to 1B)
smem_SFA        |       1,024 B | E8M0 scale factors for A
smem_SFB        |       1,024 B | E8M0 scale factors for B
smem_SFAux      |       1,024 B | E8M0 scale factors for Aux
-------------------------------------------
Mainloop Total  |      52,224 B |
Epilogue        |      17,408 B |
Kernel Total    |      71,680 B | vs 101,376 B limit → FITS
```

**Design rationale for 6-plane loading:** SM121 is memory IO bound. Loading A (and SFA) once per K-tile and reusing for both linear and gate GEMMs saves ~33% activation-side bandwidth vs a 4-plane approach that would require loading A twice.

### Key Constraints

1. **K dimension must be 128**: Reducing K breaks TMA scale factor layout compatibility
2. **Pipeline stages >= 2**: CUTLASS enforces ≥2 stages via static_assert
3. **FP4 unpacked in SMEM**: MMA instructions require byte-aligned access, so FP4 is stored as 1B not 0.5B
4. **SM121 SMEM limit is 101KB**: Architectural, not a runtime knob

### Fallback: B/Aux SMEM Sharing (if 64×64×128 tile still fails)

If the 64×64×128 tile continues to hit the TMA descriptor issue, the recommended fallback is to share the `smem_B` and `smem_Aux` buffers. This enables the 64×128×128 tile (N=128 avoids the SFB_NumBlocks=1 issue):

- Current 64×128×128 SMEM: 112,640 B (11KB over limit)
- With B/Aux sharing: saves ~33,792 B → ~78,848 B (well under limit)
- Tradeoff: sequential loading (can't prefetch Aux while consuming B)
- Requires pipeline restructuring: two-pass approach per K-tile

---

## Critical Insight from TRT-LLM SM90 Reference (2026-02-02)

After studying the working TRT-LLM implementation at `~/projects/TensorRT-LLM`, the architecture is **fundamentally different** from what we were attempting:

### What TRT-LLM Actually Does

1. **No custom epilogue** - SwiGLU is applied **inline in the kernel's consumer loop**, not in an epilogue:
   ```cpp
   // In GemmUniversalGated::operator() consumer warp group
   Activation elt_op;
   for (int i = 0; i < size(accumulators0); i++) {
       accumulators0[i] = (accumulators0[i] * scale_d0) * elt_op(scale_d1 * accumulators1[i]);
   }
   // Then call STANDARD epilogue with accumulators0
   ```

2. **Gated mainloop computes dual GEMMs** into two accumulators:
   ```cpp
   collective_mainloop.mma(pipeline, state, accum0, accum1, k_tile_count, ...);
   ```

3. **Kernel uses SFINAE enablement** based on mainloop's `isGated` flag:
   ```cpp
   template <...>
   class GemmUniversalGated<..., cute::enable_if_t<CollectiveMainloop_::isGated>>
   ```

### Key Reference Files (TRT-LLM)

| File | Purpose | Lines |
|------|---------|-------|
| `gemm/kernel/sm90_gemm_gated_tma_warpspecialized_pingpong.hpp` | Kernel with inline SiLU | ~600 |
| `gemm/collective/sm90_mma_gated_tma_gmma_ss_warpspecialized_fp8.hpp` | Gated mainloop | ~650 |
| `gemm/collective/collective_builder_gated.hpp` | Builder | ~50 |

### Correct Approach for SM120

1. **Create `CollectiveMmaGated` for SM120 block-scaled** (specialization of CollectiveMma):
   - Add `static constexpr bool isGated = true;`
   - TensorStorage has 6 smem arrays: A, B, Aux, SFA, SFB, SFAux
   - `mma()` takes TWO accumulators and computes both GEMMs
   - `load()` loads all 6 tensors

2. **Create `GemmUniversalGated` kernel specialization**:
   - Enable via SFINAE on `CollectiveMainloop::isGated`
   - Allocate two accumulator tensors
   - Call gated `mma()` with both accumulators
   - Apply SiLU **inline** after mma, before epilogue
   - Use **standard** CUTLASS epilogue (not custom!)

3. **Compile-time validation** (critical for debugging):
   ```cpp
   static_assert(cute::is_tuple_v<GmemTiledCopyPairA>, "Must be tuple");
   static_assert(cute::tuple_size<GmemTiledCopyPairA>::value == 2, "Pair size");
   ```

### What Was Wrong With Our Early Approach (Fixed)

1. ~~Custom epilogue that applies SwiGLU~~ → **Fixed**: SwigluBias is now inline in mainloop's `mma()`
2. ~~Fighting with `GemmUniversalAdapter` type aliases~~ → **Fixed**: using standard `GemmUniversal` with single-accumulator `mma()` overload
3. ~~Complex type inheritance in gated mainloop~~ → **Fixed**: composition pattern with `Base::Params` wrapping

---

## Overview

This document tracks the implementation of Layer 1A - fusing SwiGLU into FC1's epilogue using a two-accumulator gated GEMM pattern.

## Architecture

### Current Flow
```
FC1 GEMM [M, 2*inter] → HBM → doGatedActivationKernel → [M, inter] → HBM → quantize → FC2
```

### Fused Flow (Layer 1A)
```
Gated FC1 GEMM (A @ W_linear, A @ W_gate) → SwiGLU Epilogue → [M, inter] BF16 → HBM → quantize → FC2
```

## Key Implementation Files

| File | Purpose |
|------|---------|
| `cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp` | Gated mainloop: 6-plane TMA, dual-accumulator `mma()`, inline SwigluBias |
| `moe_gemm_sm120_mixed_input_launcher.h` | `GatedFC1SwigluParams` struct (device pointers) |
| `moe_gemm_sm120_mixed_input_launcher.inl` | `sm120_fused_act_moe_gemm_kernelLauncher` |
| `cutlass_fused_moe_kernels.cuh` | Gated dispatch, upstream pointer setup, FP8 quantize via `doActivation(Identity)` |
| `include/moe_gemm_kernels.h` | `TmaWarpSpecializedGroupedGemmInput::GatedFC1` struct |
| `moe_gemm_tma_warp_specialized_input.cu` | Workspace allocation for gated arrays |

---

## Step 1: Add Gated FC1 Kernel Specialization

### 1.1 New Macro for Gated Mode

Add to `moe_gemm_sm120_mixed_input_launcher.inl`:

```cpp
// =============================================================================
// MACRO: Gated Mode (FC1 with fused SwiGLU)
// =============================================================================
// Gated GEMM: computes both A @ W_linear and A @ W_gate simultaneously
// - A = FP8 activations (shared for both GEMMs)
// - B = W_linear [K, N/2] (first half of FC1 weights)
// - Aux = W_gate [K, N/2] (second half of FC1 weights)
// - Output: SwiGLU(gate) * linear = silu(A @ W_gate) * (A @ W_linear)
//
// Key differences from standard mode:
// - Uses CollectiveMmaGated mainloop (dual accumulator)
// - Custom gated epilogue applies SwiGLU and stores BF16
// - Logical N is inter_size (not 2*inter_size)
//
#define DEFINE_SM120_MXFP4_GATED_NAMESPACE(NAMESPACE_NAME, TILE_M_VAL, TILE_N_VAL, TILE_K_VAL) \
namespace NAMESPACE_NAME {                                                                     \
                                                                                               \
using namespace cute;                                                                          \
                                                                                               \
/* Element types: FP8 activations × FP4 weights → BF16 output */                               \
using ElementInputA = cutlass::float_e4m3_t;                                                   \
using ElementInputB = cutlass::float_e2m1_t;                                                   \
using ElementA = cutlass::mx_float8_t<ElementInputA>;                                          \
using ElementB = cutlass::mx_float4_t<ElementInputB>;                                          \
using ElementAux = ElementB;  /* Gate weights same as linear weights */                       \
                                                                                               \
using ElementC = cutlass::bfloat16_t;                                                          \
using ElementD = cutlass::bfloat16_t;                                                          \
                                                                                               \
using ElementAccumulator = float;                                                              \
using ElementCompute = float;                                                                  \
using ElementSF = cutlass::float_ue8m0_t;                                                      \
                                                                                               \
/* ... rest of type definitions ... */                                                         \
                                                                                               \
/* Gated mainloop uses SM90 pattern adapted for SM120 */                                       \
using CollectiveMainloop = CollectiveMmaGated<...>;                                            \
                                                                                               \
/* Custom gated epilogue */                                                                    \
using CollectiveEpilogue = Sm120GatedSwiGLUEpilogue<...>;                                      \
                                                                                               \
}  /* namespace NAMESPACE_NAME */
```

### 1.2 Key Type Changes

| Parameter | Standard FC1 | Gated FC1 |
|-----------|--------------|-----------|
| Logical N | `2 * inter_size` | `inter_size` |
| B operand | `W_fc1 [K, 2*inter]` | `W_linear [K, inter]` |
| Aux operand | N/A | `W_gate [K, inter]` |
| Output | `[M, 2*inter]` BF16 | `[M, inter]` BF16 (SwiGLU applied) |
| Accumulators | 1 | 2 (linear + gate) |

---

## Step 2: Gated Mainloop (DRY Approach)

### Design Principle: Inheritance over Duplication

Instead of duplicating the ~1000-line base mainloop, the gated mainloop:
1. **Inherits type aliases** from `CollectiveMma<MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>>`
2. **Reuses base `load_init()`** by calling it twice (once for B, once for Aux)
3. **Reuses base TMA setup** via `to_underlying_arguments`
4. **Reuses base helper functions** like `partition_fragment_SFA`, `get_layoutSFB_TV`

Only the truly different parts are new code:
- `TensorStorage` with added `smem_Aux`/`smem_SFAux` (~10 lines)
- `Arguments/Params` with Aux fields (~20 lines)
- `load()` that copies 6 tensors instead of 4 (~50 lines)
- `mma()` with dual accumulator pattern (~80 lines)

**Result: ~150 lines of new code instead of ~1000 lines of duplication.**

### 2.1 Adapt CollectiveMmaGated for SM120

The existing `sm90_mma_gated_tma_gmma_ss_warpspecialized_fp8.hpp` provides the pattern:

```cpp
// From existing gated mainloop:
struct SharedStorage {
    cute::array_aligned<ValTypeA, cosize_v<SmemLayoutA>> smem_A;
    cute::array_aligned<ValTypeB, cosize_v<SmemLayoutB>> smem_B;
    cute::array_aligned<ValTypeAux, cosize_v<SmemLayoutAux>> smem_Aux;  // Gate weights
};

// MMA function computes both accumulators:
void mma(..., FrgTensorC& accum0, FrgTensorC& accum1, ...) {
    // Load A, B, Aux tiles
    // Compute: accum0 += A @ B (linear)
    //          accum1 += A @ Aux (gate)
}
```

For SM120, we need to:
1. Use the same SMEM layout for A, B, Aux
2. Adapt TMA descriptors for block-scaled MX types
3. Keep the dual-accumulator MMA pattern

### 2.2 SMEM Budget

```
SM120 Total SMEM: 99,328 bytes

Gated FC1 (128x128 tile, 2 stages):
  - smem_A (activations): 128 × 128 × 1 byte × 2 stages = 32 KB
  - smem_B (W_linear):    128 × 128 × 0.5 byte × 2 stages = 16 KB  
  - smem_Aux (W_gate):    128 × 128 × 0.5 byte × 2 stages = 16 KB
  - Scale factors:        ~4 KB
  - Epilogue scratch:     ~8 KB
  - Total: ~76 KB (fits within 99 KB)
```

---

## Step 3: SwiGLU Epilogue

### 3.1 Create Custom Epilogue

Location: `csrc/nv_internal/.../include/sm120_gated_swiglu_epilogue.hpp`

```cpp
// Custom epilogue that receives two accumulator fragments and applies SwiGLU
template <
    class TileShape,
    class ElementOutput,     // bfloat16_t
    class ElementCompute     // float
>
struct Sm120GatedSwiGLUEpilogue {
    
    struct SharedStorage {
        // Minimal scratch for accumulator tiles
    };
    
    struct Arguments {
        ElementOutput* ptr_aux;    // Output buffer [M, N]
        int64_t stride_aux;        // Row stride
    };
    
    // Main epilogue function - receives both accumulators
    template <class FrgTensorC0, class FrgTensorC1, class BlockCoord>
    CUTLASS_DEVICE void operator()(
        FrgTensorC0 const& accum_linear,  // A @ W_linear result
        FrgTensorC1 const& accum_gate,    // A @ W_gate result
        BlockCoord const& blk_coord,
        Arguments const& args
    ) {
        // FlashInfer convention: output = silu(gate) * linear
        // Process element-by-element
        
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_linear); ++i) {
            ElementCompute linear_val = accum_linear(i);
            ElementCompute gate_val = accum_gate(i);
            
            // SiLU(x) = x * sigmoid(x)
            ElementCompute sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
            ElementCompute silu_gate = gate_val * sigmoid_gate;
            
            // SwiGLU output
            ElementCompute output_val = silu_gate * linear_val;
            
            // Convert to BF16 and store
            // ... coordinate calculation and store ...
        }
    }
};
```

### 3.2 Match FlashInfer Math

From `doGatedActivationKernel` (line 2047-2053):
```cpp
// FlashInfer order:
linear_value = gemm_result_vec[elem_index];           // First N/2
gate_value = gemm_result_vec[elem_index + inter_size]; // Second N/2
gate_act = fn(gate_value, linear_value);  // GLUAdaptor: silu(gate) * linear
```

Our epilogue must match exactly:
- `linear` comes from `accum0` (A @ W_linear)
- `gate` comes from `accum1` (A @ W_gate)
- Output = `silu(gate) * linear`

---

## Step 4: Wire into Call Graph

### 4.1 Add to moe_kernels.h

```cpp
class CutlassMoeFCRunnerInterface {
    // ...
    
    // NEW: Gated FC1 with fused SwiGLU epilogue
    virtual void gemm1_gated(
        void const* const input,
        void* const aux_output,           // [M, inter_size] BF16 SwiGLU output
        int64_t const* const expert_first_token_offset,
        TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
        void const* const fc1_expert_weights,  // [experts, 2*inter, K] - will split internally
        int64_t const* const num_valid_tokens_ptr,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
        int64_t const num_rows,
        int64_t const expanded_num_rows,
        int64_t const hidden_size,
        int64_t const inter_size,
        int const num_experts_per_node,
        cudaStream_t stream,
        cutlass_extensions::CutlassGemmConfig config,
        bool enable_pdl) = 0;
};
```

### 4.2 Modify cutlass_fused_moe_kernels.cuh

```cpp
// In gemm1() function:
if (fuse_swiglu_fc1_aux_bf16) {
    // Use gated FC1 path
    Self::gemm1_gated(
        gemm_runner,
        input,
        aux_output,  // NEW: output goes here instead of intermediate_result
        expert_first_token_offset,
        tma_ws_input_template,
        fc1_expert_weights,
        num_valid_tokens_ptr,
        fc1_fp4_act_flat,
        num_rows,
        expanded_num_rows,
        hidden_size,
        inter_size,
        num_experts_per_node,
        stream,
        config,
        enable_pdl
    );
    // Skip doGatedActivation - output already has SwiGLU applied
} else {
    // Existing path: FC1 GEMM → doGatedActivation
    // ...
}
```

---

## Step 5: Validation

### 5.1 Kernel-Level Oracle

```python
def test_gated_fc1_kernel():
    """Compare fused gated FC1 vs baseline FC1 + doGatedActivation."""
    
    # Baseline path
    fc1_output_raw = run_fc1_gemm(input, fc1_weights)  # [M, 2*inter]
    baseline_aux = run_doGatedActivation(fc1_output_raw)  # [M, inter]
    
    # Fused path
    fused_aux = run_gated_fc1(input, fc1_weights)  # [M, inter]
    
    # Gate 1: Bit-identical (ideal)
    if torch.equal(baseline_aux, fused_aux):
        return "PASS (bit-identical)"
    
    # Gate 2: Numerically equivalent (acceptable)
    max_diff = torch.abs(baseline_aux - fused_aux).max()
    mismatch_rate = (baseline_aux != fused_aux).float().mean()
    
    if max_diff <= 0.001 and mismatch_rate <= 0.001:
        return f"PASS (allclose: max_diff={max_diff}, mismatch={mismatch_rate*100:.2f}%)"
    
    return f"FAIL (max_diff={max_diff}, mismatch={mismatch_rate*100:.2f}%)"
```

### 5.2 Test Shapes

| M | Description | Priority |
|---|-------------|----------|
| 1 | Single token decode | HIGH |
| 16 | Small batch decode | HIGH |
| 64 | Medium batch | MEDIUM |
| 128 | Full tile | HIGH |
| 129 | Tile + remainder | MEDIUM |

---

## Step 6: CUDA Graph Verification

Run immediately after Layer 1A works:

```bash
# Start vLLM with fused path
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --mxfp4-backend CUTLASS \
    --mxfp4-fuse-fc1-swiglu  # NEW FLAG

# Verify CUDA graphs capture
nsys profile --cudagraph-trace=all \
    llama-benchy --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b --pp 512 --tg 32

# Check for:
# 1. No graph capture errors
# 2. Graph replay overhead < 0.5ms
# 3. doGatedActivationKernel not in trace
```

---

## Implementation Order

1. **Create gated epilogue** (`sm120_gated_swiglu_epilogue.hpp`)
   - Simple version: just SwiGLU + BF16 store
   - No EVT complexity yet

2. **Add gated kernel namespace** (in launcher .inl)
   - Adapt SM90 gated mainloop for SM120
   - Dual accumulator pattern

3. **Wire gemm1_gated()** (moe_kernels.h, cutlass_fused_moe_kernels.cuh)
   - Add interface
   - Add dispatch flag

4. **Unit test** 
   - Kernel-level oracle
   - Bit-identical or allclose validation

5. **CUDA graph check**
   - Verify capture works
   - Measure overhead

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SM120 gated mainloop doesn't compile | Start with SM90 pattern, adapt incrementally |
| SMEM overflow | Profile actual usage, reduce stages if needed |
| Epilogue coordinate math wrong | Compare tile-by-tile with baseline |
| CUDA graphs fail | Fall back to eager mode, investigate |

---

## Status (2026-02-06)

### Code Complete

| Step | Status | Notes |
|------|--------|-------|
| Spike validation | ✅ | PyTorch proves approach works |
| Gated mainloop | ✅ | `sm120_blockscaled_mma_gated_array_tma.hpp` - 6-plane TMA, dual-accumulator |
| Inline SwigluBias | ✅ | Full formula in single-accumulator `mma()` overload |
| Dispatch policy | ✅ | `MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated` |
| Standard `GemmUniversal` compatibility | ✅ | Single-accumulator `mma()` works with standard kernel |
| Launcher function | ✅ | `sm120_fused_act_moe_gemm_kernelLauncher` - simplified, CUDA-graph safe |
| Upstream pointer integration | ✅ | Gated arrays in `TmaWarpSpecializedGroupedGemmInput::GatedFC1` |
| cudaStreamSynchronize eliminated | ✅ | Gated pointers pre-computed by upstream stride kernel |
| cudaMemcpy eliminated | ✅ | SwigluBias params as device pointers, read on-device |
| Interface wiring | ✅ | `FLASHINFER_FUSED_ACTIVATION_KERNEL_LAUNCH` compile guard |
| JIT flag integration | ✅ | `FLASHINFER_FUSED_ACTIVATION_LAUNCH=1` env var |
| Weight split validation | ✅ | `test_gated_weight_split.py` - offset calculations correct |

### Pending Hardware Test

| Step | Status | Notes |
|------|--------|-------|
| JIT compile on SM121 | 🔄 TODO | Recompile after recent refactoring |
| Runtime launch (64×64×128) | 🔄 TODO | Was "illegal instruction" pre-refactoring |
| Numerical accuracy | 🔄 TODO | Compare fused vs `doGatedActivation` baseline |
| nsys validation | 🔄 TODO | Verify `doGatedActivationKernel` eliminated |
| CUDA graph capture/replay | 🔄 TODO | |
| End-to-end benchmark | 🔄 TODO | Measure throughput delta |

## Previous Blocking Issue: Epilogue Pointer Array Mismatch (RESOLVED 2026-02-02)

**Original error**: When compiling with `FLASHINFER_FUSED_ACTIVATION=1`, the epilogue expected `ElementD**` 
(array of per-expert pointers) but we passed `ElementOutput*` (single buffer).

**Resolution**: Extended the launcher infrastructure to:
1. Allocate per-expert output pointer and stride arrays in workspace
2. Compute output pointers: `aux_base + expert_first_token_offset[e] * inter_size`
3. Compute output strides using `make_cute_packed_stride((M_e, inter_size, 1))`
4. Pass these device arrays to the epilogue instead of single values

See "Epilogue Pointer Array Fix" section above for implementation details.

## Test Results

```
Passed: 4/4
  M=   1: PASS (bit-identical)
  M=  16: PASS (bit-identical)
  M=  64: PASS (allclose - FP accumulation order)
  M= 128: PASS (bit-identical)

Two-GEMM Overhead (PyTorch, not fused):
  M=1:  32% overhead
  M=16: 59% overhead
  M=64: 55% overhead
  
Note: Fused gated mainloop will eliminate this overhead
```

## Files Created

```
flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/
├── gemm/collective/
│   └── sm120_blockscaled_mma_gated_array_tma.hpp   # Gated mainloop (WIP)
└── epilogue/
    └── sm120_gated_swiglu_epilogue.hpp             # SwiGLU epilogue (WIP)
```

## Weight Layout Analysis

### FC1 Weight Storage

```
FC1 weights per expert: [2*inter_size, K] column-major (FP4 packed)
                        ├── Linear half: columns [0, inter_size)
                        └── Gate half:   columns [inter_size, 2*inter_size)

Scale factors (MXFP4, block=32): [2*inter_size, ceil(K/32)]
                                 ├── Linear SF: rows [0, inter_size)
                                 └── Gate SF:   rows [inter_size, 2*inter_size)
```

### Pointer Offset Calculations

```cpp
// Per expert e:
weight_base = weights + e * (2*inter_size * K);
sf_base = weight_sf + getOffsetWeightSF(e, 2*inter_size, K, MXFPX);

// Linear half (offset 0):
ptr_linear = weight_base;
sf_linear = sf_base;

// Gate half (offset by inter_size columns):
// For FP4 (2 elements/byte): byte_offset = inter_size * K / 2
ptr_gate = weight_base + (inter_size * K / 2);  // bytes
sf_gate = sf_base + (inter_size * ceil(K/32));  // SF elements
```

### Key Gotchas

1. **N in problem shape must be inter_size** (not 2*inter_size)
2. **Scale factor offset**: `inter_size * ceil(K/32)` SF elements, not bytes
3. **Group index mapping**: Must match baseline exactly

---

## Remaining Work

### 1. Gated Namespace ✅ DONE

The `DEFINE_SM120_MXFP4_GATED_NAMESPACE` macro now defines:
- Full CollectiveBuilder-based types matching standard mode
- `kIsGatedMode = true` marker
- Reuses standard kernel for two-GEMM bring-up approach

### 2. Two-GEMM Bring-Up: Implementation Status

**Status**: Flag infrastructure complete. Device-side implementation complex.

**Challenge**: The TMA input structure (`tma_ws_input`) contains device-side pointer
arrays populated by a stride-fill kernel. Modifying for gated path requires:

1. **Device-side offset kernel**: Compute per-expert gate pointer offsets
2. **New TMA input copy**: Allocate device arrays for modified pointers  
3. **Modified problem shapes**: Change N from 2*inter_size to inter_size

**Compile Flags Added**:
- `FLASHINFER_FUSED_ACTIVATION`: Enable gated FC1 path detection
- `FLASHINFER_FUSED_ACTIVATION_TWO_GEMM_BRINGUP`: Enable two-GEMM debug path

**Current Behavior** (with flags enabled):
- Logs activation of gated path with offset calculations
- Falls through to standard path (one GEMM + doGatedActivation)
- No functional change yet

**Implementation Options**:

**Option A: Offset kernel** (~80 lines)
```cpp
__global__ void computeGatedOffsets(
    void const** ptr_weight_src, void const** ptr_weight_gate,
    ElementSF const** sf_weight_src, ElementSF const** sf_weight_gate,
    int64_t inter_size, int64_t hidden_size, int num_experts) {
  int e = blockIdx.x;
  if (e < num_experts) {
    int64_t weight_off = (inter_size * hidden_size) / 2;
    int64_t sf_off = inter_size * ((hidden_size + 31) / 32);
    ptr_weight_gate[e] = (char*)ptr_weight_src[e] + weight_off;
    sf_weight_gate[e] = sf_weight_src[e] + sf_off;
  }
}
```

**Option B: Skip to GemmUniversalGated** (recommended)
- Weight split math validated by `test_gated_weight_split.py`
- Gated mainloop/epilogue infrastructure complete
- Wire GemmUniversalGated directly for actual performance win
- Avoids two-GEMM overhead and complexity

### 3. Validation Tests

1. **Compile test**: Build with `-DFLASHINFER_FUSED_ACTIVATION`
2. **Weight split test**: `test_gated_weight_split.py` ✅ PASSING
3. **One-layer test**: Compare gated path vs baseline (pending)
4. **CUDA graph test**: Capture + replay (pending)

---

## End-to-End Testing

### Enabling the Gated FC1 Kernel

The gated FC1 kernel is controlled by the `FLASHINFER_FUSED_ACTIVATION_LAUNCH` environment variable:

```bash
# Enable gated FC1 kernel launch
export FLASHINFER_FUSED_ACTIVATION_LAUNCH=1
```

When enabled, the JIT compiler adds `-DFLASHINFER_FUSED_ACTIVATION_KERNEL_LAUNCH` to the nvcc flags
and creates a separate cached module with suffix `_fusedact`.

### Testing with vLLM

```bash
# Clear JIT cache to force recompilation
docker exec vllm-dev rm -rf /root/.cache/flashinfer/0.6.1/121f/cached_ops/fused_moe_120*

# Start vLLM with gated FC1 enabled
docker exec -it vllm-dev bash -c '
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export FLASHINFER_FUSED_ACTIVATION_LAUNCH=1
export FLASHINFER_LOGLEVEL=3

vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization mxfp4 \
    --mxfp4-backend CUTLASS \
    --mxfp4-layers moe \
    --tensor-parallel-size 1
'
```

### Expected Log Messages

When gated FC1 is active, you should see:
```
[SM120 MoE] Gated FC1: PRODUCTION mode detected
[SM120 MoE] Gated FC1: KERNEL LAUNCH enabled
[SM120 MoE] Gated FC1 complete - skipping doGatedActivation
```

### nsys Profiling

```bash
# Profile with nsys to verify kernel disappears
nsys profile --output gated_fc1_test \
    python3 -c "
import requests
requests.post('http://localhost:8000/v1/completions', json={
    'model': 'gpt-oss-120b',
    'prompt': 'Hello, world!',
    'max_tokens': 32
})
"

# Check kernels
nsys stats gated_fc1_test.nsys-rep --report gpukernsum
```

**Expected results:**
- `doGatedActivationKernel` should NOT appear
- A new gated GEMM kernel should appear (GemmUniversalGated instantiation)

### CUDA Graph Check

```bash
# Test with CUDA graphs enabled (default)
vllm serve ... # as above

# Verify graphs work
llama-benchy --base-url http://localhost:8000/v1 --model gpt-oss-120b --pp 256 --tg 32

# Compare with eager mode
vllm serve ... --enforce-eager
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "Falling through to standard path" | `FLASHINFER_FUSED_ACTIVATION_LAUNCH` not set | Set env var before JIT |
| JIT cache not rebuilding | Old cache present | Clear `~/.cache/flashinfer/` |
| Kernel not appearing in nsys | Not taking gated path | Check FLASHINFER_LOGLEVEL=3 logs |

---

## Revised Implementation Approach (2026-02-02)

Based on analysis of TRT-LLM's working SM90 implementation, here's the correct approach:

### Phase 1: Create Gated SM120 Mainloop (CollectiveMmaGated)

Location: `cutlass_extensions/gemm/collective/sm120_mma_gated_blockscaled_array_tma.hpp`

**Key requirements:**
1. Specialization of `CollectiveMma` for the gated dispatch policy
2. `static constexpr bool isGated = true;`
3. **TensorStorage** with 6 smem arrays:
   ```cpp
   cute::array_aligned<ValTypeA, cosize_v<SmemLayoutA>> smem_A;
   cute::array_aligned<ValTypeB, cosize_v<SmemLayoutB>> smem_B;
   cute::array_aligned<ValTypeB, cosize_v<SmemLayoutB>> smem_Aux;  // Gate weights
   cute::array_aligned<ElementSF, cosize_v<SmemLayoutSFA>> smem_SFA;
   cute::array_aligned<ElementSF, cosize_v<SmemLayoutSFB>> smem_SFB;
   cute::array_aligned<ElementSF, cosize_v<SmemLayoutSFB>> smem_SFAux;  // Gate scales
   ```
4. **mma() with dual accumulators**:
   ```cpp
   template <class FrgTensorC>
   CUTLASS_DEVICE void mma(
       MainloopPipeline pipeline, PipelineState smem_pipe_read,
       FrgTensorC& accum_linear,  // Output: A @ W_linear
       FrgTensorC& accum_gate,    // Output: A @ W_gate
       int k_tile_count, int thread_idx,
       TensorStorage& shared_tensors, Params const& params);
   ```

**Implementation strategy:**
- Copy SM120 base mainloop (`sm120_blockscaled_mma_array_tma.hpp`) 
- Add `smem_Aux`, `smem_SFAux` to TensorStorage
- Duplicate the MMA loop to compute both products
- Use same TMA patterns as base (block-scaled FP4)

### Phase 2: Create Gated SM120 Kernel (GemmUniversalGated)

Location: `cutlass_extensions/gemm/kernel/sm120_gemm_gated_blockscaled_array_tma.hpp`

**Key requirements:**
1. SFINAE enablement:
   ```cpp
   class GemmUniversalGated<..., cute::enable_if_t<CollectiveMainloop_::isGated>>
   ```
2. Standard type aliases from mainloop/epilogue (no adapter compatibility needed)
3. **Dual accumulator allocation**:
   ```cpp
   Tensor accum_linear = partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));
   Tensor accum_gate = partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));
   ```
4. **Inline SiLU** after mma, before epilogue:
   ```cpp
   collective_mainloop.mma(pipeline, state, accum_linear, accum_gate, ...);
   
   // Apply SwiGLU inline - this is the key insight!
   Activation<float> elt_op;
   for (int i = 0; i < size(accum_linear); i++) {
       accum_linear[i] *= elt_op(accum_gate[i]);
   }
   
   // Use STANDARD epilogue with accum_linear
   collective_epilogue.store(epi_pipe, ..., accum_linear, ...);
   ```

### Phase 3: Integration and Validation

1. **Compile-time validation** in mainloop:
   ```cpp
   static_assert(cute::is_tuple_v<GmemTiledCopyPairA>, "GmemTiledCopyPairA must be tuple");
   static_assert(cute::tuple_size<GmemTiledCopyPairA>::value == 2, "Must be pair");
   static_assert(cute::is_tuple_v<SmemLayoutAtomsA>, "SmemLayoutAtomsA must be tuple");
   ```

2. **Unit test**: Compare gated path vs baseline FC1 + doGatedActivation
3. **nsys validation**: Confirm doGatedActivationKernel disappears
4. **CUDA graph test**: Capture and replay

### Why This Will Work

1. **Copy, don't invent**: Base mainloop structure is proven to work
2. **Inline fusion**: SiLU in kernel, not custom epilogue - matches TRT-LLM
3. **Type safety**: static_assert catches mismatches at compile time
4. **Standard epilogue**: No fighting with EVT or adapter compatibility

---

## Implementation Architecture (Current)

### Design Approach: Gated Mainloop + Standard Kernel

The implementation uses a **gated mainloop** (`CollectiveMma` specialization) with a single-accumulator `mma()` overload that applies SwigluBias inline. This allows the standard `GemmUniversal` kernel to work unchanged.

**Key design decisions:**

1. **Composition, not inheritance**: Gated `Params` wraps `Base::Params` + `AuxParams`
2. **6-plane TMA loading**: A loaded once, reused for both linear and gate GEMMs (saves ~33% activation bandwidth)
3. **Inline SwigluBias**: Full formula with alpha/beta/limit, applied after dual-accumulator `mma()` call
4. **Device-pointer params**: SwigluBias alpha/beta/limit passed as `float const*` device pointers, read on-device
5. **Upstream pointer computation**: Gate weight/SF/output arrays populated by the existing `computeTmaWarpSpecializedInputPointers` kernel

### Key Files

- `sm120_blockscaled_mma_gated_array_tma.hpp` - Gated mainloop
- `moe_gemm_sm120_mixed_input_launcher.inl` - Launcher (simplified, no sync)
- `moe_gemm_sm120_mixed_input_launcher.h` - `GatedFC1SwigluParams` struct
- `cutlass_fused_moe_kernels.cuh` - Dispatch, upstream setup, FP8 quantize
- `include/moe_gemm_kernels.h` - `TmaWarpSpecializedGroupedGemmInput::GatedFC1`

---

## Design Notes

### Why Standard GemmUniversal (Not GemmUniversalGated)

The original plan called for cloning `GemmUniversal` into a `GemmUniversalGated` with SFINAE and dual accumulators. Instead, the implementation uses a **single-accumulator `mma()` overload** in the gated mainloop that:
1. Internally allocates two accumulators
2. Calls the dual-accumulator `mma()` 
3. Applies SwigluBias inline
4. Returns the fused result in the single accumulator

This is simpler and requires zero kernel modifications -- only the mainloop is custom.

### SwigluBias vs SiLU

The original plan discussed simple `SiLU(gate) * linear`. The implementation uses the full **SwigluBias** formula:
```
gate_clamped = min(gate, limit)
linear_clamped = clamp(linear, -limit, limit)
output = gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
```
This matches the `SwigluBiasAdaptor` in `doGatedActivation` and is required by gpt-oss-120b's model config.

---

## Next Steps

1. **Hardware test**: Compile and launch gated kernel on SM121
   - Clear JIT cache: `rm -rf ~/.cache/flashinfer/0.6.*/121*/cached_ops/fused_moe_120*`
   - Set `FLASHINFER_FUSED_ACTIVATION_LAUNCH=1`
   - Check if 64×64×128 tile launches successfully after the pointer refactoring

2. **If 64×64×128 still fails**: Debug TMA descriptor or implement B/Aux SMEM sharing (see Fallback section above)

3. **If launch succeeds**: Run numerical accuracy test, nsys profile, CUDA graph test, benchmark

4. **Layer 1A+ (future)**: Fuse FP8 quantization into the CUTLASS epilogue to eliminate the remaining `doActivation(Identity)` kernel

---

## Session Log

### 2026-02-06: cudaStreamSynchronize and cudaMemcpy elimination

- Refactored `TmaWarpSpecializedGroupedGemmInput` with `GatedFC1` struct
- Moved gated pointer computation into upstream `computeTmaWarpSpecializedInputPointers` kernel
- Removed separate `computeGatedPointersAndStrides` kernel and `GatedFC1WorkspaceLayout`
- Changed `SwigluBiasParams` from scalar values to `float const*` device pointers
- Eliminated all `cudaMemcpy` D2H calls from the gated path
- Result: production path is fully CUDA-graph safe

### 2026-02-02: TMA Debugging and B/Aux Sharing Attempt

- Used cuda-gdb to identify crash at `UTMALDG.4D` for SFB plane (64×64×128 tile)
- Connected to prior CTA_N=16 investigation (same `SFB_NumBlocks=1` issue)
- Added TMA commit/wait to `tensormaps_cp_fence_release()` (necessary but not sufficient)
- Started B/Aux SMEM sharing approach, reverted due to pipeline restructuring complexity
- Key finding: tensormap workspace was all zeros for failing tiles (initialization issue)
