# Fused MoE FC1+FC2 Kernel Design

## Executive Summary

This document outlines a phased approach to fusing the FC1 and FC2 layers of the MoE (Mixture of Experts) GEMM into a single kernel, eliminating HBM round-trips and kernel launch overhead. The optimization is structured as a 4-layer stack, with each layer building on the previous.

**Target improvement: 7-14% decode throughput (62 t/s → 66-71 t/s)**

*Note: Original target was 15-25%, revised after Amdahl's law analysis showing ~96% HBM utilization.*

### What This Does NOT Optimize

The fused FC1+FC2 kernel targets only the GEMM and activation path. These kernels remain unchanged:

| Kernel | Count/Layer | Why Not Fused |
|--------|-------------|---------------|
| `topkGatingSoftmax` | 1 | Routing decision, runs before GEMMs |
| `expandInputRowsKernel` | 1 | Token reordering for expert assignment |
| `finalizeMoeRoutingKernel` | 1 | Output assembly after GEMMs |
| `blockExpertPrefixSumKernel` | ~1 | Expert assignment bookkeeping |
| `fusedBuildExpertMapsSortFirstToken` | ~1 | Expert map construction |
| `computeStridesTmaWarpSpecialized` | 2 | TMA stride setup (may reduce to 1 with fusion) |

**Do not expect 20%+ improvement from FC fusion alone.** The expert routing overhead (~5 kernels/layer) remains.

---

## Current Architecture (Baseline)

### Data Flow

```
Input BF16 ──► mxfp8_quantize ──► FC1 Kernel ──► HBM ──► SwiGLU ──► HBM ──► mxfp8_quantize ──► FC2 Kernel ──► Output BF16
                   │                  │                    │                    │                  │
                   ▼                  ▼                    ▼                    ▼                  ▼
              BF16→FP8+scales    FP8×FP4 GEMM        gate*silu(up)         BF16→FP8+scales    FP8×FP4 GEMM
                                 (BF16 out)            (BF16)                                 (BF16 out)
```

**Critical: FC2 A-operand quantization**
- SwiGLU outputs BF16
- FC2 is FP8×FP4 GEMM, so A-operand must be FP8
- A separate `quantize_with_block_size` kernel (or prologue) converts BF16→FP8+scales before FC2

### Key Observations

| Component | Implementation | Location |
|-----------|----------------|----------|
| FC1 input quantize | `mxfp8_quantize` | `expandInputRowsKernel` + `quantize_with_block_size` |
| FC1 GEMM | CUTLASS SM120 grouped GEMM | `moe_gemm_sm120_mixed_input_launcher.inl` |
| SwiGLU | Separate CUDA kernel | `doGatedActivationKernel()` in `moe_kernels.h` |
| **FC2 input quantize** | `quantize_with_block_size` | **Converts SwiGLU BF16 → FP8+scales** |
| FC2 GEMM | CUTLASS SM120 grouped GEMM | Same launcher as FC1 |

### Current Performance

| Metric | Value |
|--------|-------|
| Decode throughput | 62 t/s |
| Decode latency | 16.13 ms |
| Weight loading | ~15.5 ms (96% of theoretical bandwidth) |
| Activation I/O | ~0.6 ms |

### Kernel Launch Breakdown (per generated token)

Based on nsys profiling of a `--pp 512 --tg 32` benchmark.

*Note: "per decode step" = "per generated token" = one forward pass that produces 1 token.
The tg=32 benchmark generates 32 tokens; counts below are per single token generation.*

| Kernel Type | Count | Notes |
|-------------|-------|-------|
| **MoE-related** | | |
| FC1 input quantize (`quantize_with_block_size`) | 24 | BF16→FP8+scales |
| FC1 grouped GEMM (`GemmUniversal<GroupProblemShape>`) | 24 | Batches all 8 experts |
| SwiGLU activation (`doActivationKernel`) | 24 | Separate kernel |
| **FC2 input quantize** (`quantize_with_block_size`) | **24** | **SwiGLU BF16→FP8+scales** |
| FC2 grouped GEMM (`GemmUniversal<GroupProblemShape>`) | 24 | Batches all 8 experts |
| TMA stride setup (`computeStridesTmaWarpSpecialized`) | 48 | 24 per GEMM × 2 (FC1+FC2) |
| Expert routing (`topkGatingSoftmax`) | 24 | |
| Token expansion (`expandInputRowsKernel`) | 24 | |
| Output combine (`finalizeMoeRoutingKernel`) | 24 | |
| Expert maps/prefix sums | ~24 | Various helper kernels |
| **MoE subtotal** | **~264** | |
| **Other per-layer** | | |
| Attention (FA2/FlashInfer) | 24 | |
| QKV projection (Marlin) | ~24 | |
| O projection (Marlin) | ~24 | |
| RMSNorm (Triton) | 48 | 2 per layer |
| Other elementwise | ~50 | Misc ops |
| **Non-MoE subtotal** | **~170** | |
| **Total** | **~434** | Per generated token |

**Key findings from nsys profile:**
- The grouped GEMM IS working correctly (`GemmUniversal<GroupProblemShape>`)
- SwiGLU runs as a **separate kernel** (`doActivationKernel`), not fused
- TMA stride computation is a separate kernel per GEMM (48 total = 24 FC1 + 24 FC2)
- CUDA graph captures inflate total counts by ~3.5x in profiler output

### HBM Traffic Between FC1 and FC2

Per token, per layer:
- FC1 output writoue: `1 × 4096 × 8 experts × 2 bytes = 64 KB`
- SwiGLU read: 64 KB
- SwiGLU write: 32 KB (output is half size due to gating)
- FC2 read: 32 KB
- **Total: 192 KB per layer × 24 layers = 4.6 MB per token**

---

## Amdahl's Law Reality Check

### Stated Time Breakdown

| Component | Time | Percentage |
|-----------|------|------------|
| Weight loading | ~15.5 ms | 96% |
| Activation I/O | ~0.6 ms | 4% |
| **Total decode** | **16.13 ms** | 100% |

### Maximum Theoretical Gains

| Optimization | Time Saved | New Latency | Speedup |
|--------------|------------|-------------|---------|
| Eliminate ALL activation I/O | 0.6 ms | 15.53 ms | **3.9%** |
| Eliminate 1 ms launch overhead | 1.0 ms | 15.13 ms | **6.6%** |
| Combined maximum | 1.6 ms | 14.53 ms | **~11%** |

*Speedup = 16.13 / (16.13 - saved) - 1*

**Key insight**: If the breakdown above is accurate, achieving 15-25% improvement is impossible through activation/launch optimizations alone. Maximum theoretical is ~11%.

### How Could 15-25% Be Achieved?

For larger gains, one of these must be true:

1. **Weight loading includes bubble time**: The 15.5 ms is not pure bandwidth time; some is stall/scheduling overhead that fusion can eliminate
2. **Fusion improves occupancy**: Better memory-level parallelism from fused kernels increases effective bandwidth utilization
3. **Breakdown misattributes time**: Current profiling may double-count or misattribute time

### Bandwidth Utilization Analysis

From nsys profile of decode (pp=512, tg=32):

| Metric | Value |
|--------|-------|
| GB10 theoretical HBM bandwidth | ~273 GB/s |
| CUTLASS grouped GEMM median time | 305 µs |
| Estimated data per GEMM (decode) | ~80 MB |
| **Observed bandwidth** | **~262 GB/s** |
| **Utilization** | **~96%** |

**Interpretation**: At 96% bandwidth utilization, we are already near the physical limit. Prefetching (Layer 3) cannot "beat physics" - it can only redistribute bandwidth to hide latency if there are bubbles.

### What NCU Profiling Would Reveal

To validate whether optimization headroom exists, we need:

| Metric | What It Tells Us |
|--------|------------------|
| Achieved SM occupancy | Warps active per SM |
| Eligible warps per cycle | Memory-level parallelism |
| % stalled on memory | True bandwidth-bound fraction |
| % stalled on compute | Compute-bound fraction |
| % stalled on sync | Synchronization overhead |

If stalled-on-memory is <80%, there's headroom for fusion to improve.
If stalled-on-memory is >95%, we're hitting physics limits.

### Revised Realistic Estimates

| Layer | Claimed | Realistic | Justification |
|-------|---------|-----------|---------------|
| Layer 1: SwiGLU fusion | 8-15% | **3-4%** | Eliminates SwiGLU kernel overhead, not data |
| Layer 2: SMEM fusion | 5-8% | **1-2%** | After 1A+, only ~34 KB/layer remains |
| Layer 3: Async prefetch | 4-12% | **0-2%** | Only helps if bubbles exist at 96% BW |
| Layer 4: Persistence | 3-5% | **1-3%** | Launch overhead already amortized by CUDA graphs |
| **Combined** | **20-40%** | **7-14%** | Diminishing returns, not additive |

---

## Optimization Stack

The optimizations are structured as layers, where each layer depends on the previous:

```
┌────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION STACK                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Layer 4: LAUNCH OVERHEAD REDUCTION                [+1-3%]        │
│           └─ 4A: CUDA Graph capture (already in use, verify)      │
│           └─ 4B: Persistent tile scheduler (CUTLASS pattern)      │
│           └─ Practical alternatives to mega-kernel                │
│                                                                    │
│  Layer 3: ASYNC FC2 WEIGHT PREFETCH (EXPERIMENTAL) [+0-2%]       │
│           └─ Start FC2 load during FC1 epilogue                   │
│           └─ Only works if warp roles + SMEM layout allow         │
│           └─ CUTLASS already well-pipelined; may get <1%          │
│           └─ Requires: Layer 2 + profiling validation             │
│                                                                    │
│  Layer 2: FC1→SMEM→FC2 FUSION                     [+1-2%]         │
│           └─ Intermediate stays in SMEM, not HBM                  │
│           └─ After 1A+, saves ~34 KB/layer (incremental)          │
│           └─ Requires: Layer 1                                    │
│                                                                    │
│  Layer 1: SwiGLU EPILOGUE FUSION                  [+3-4%]         │
│           └─ 1A: Aux buffer output (low risk, recommended first)  │
│           └─ 1B: Direct N/2 output (optional, more invasive)      │
│           └─ Key challenge: width-halving (N → N/2)               │
│           └─ Reference: TRT-LLM GemmUniversalGated                │
│                                                                    │
│  Layer 0: CURRENT BASELINE                        62 t/s          │
│           └─ Separate FC1, SwiGLU, FC2 kernels                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Cumulative Improvement Estimates (Revised)

| Configuration | Throughput | Improvement | Notes |
|---------------|------------|-------------|-------|
| Baseline | 62 t/s | - | Current, ~96% BW utilization |
| Layer 1 only | 64-65 t/s | +3-4% | Eliminates SwiGLU kernel |
| Layers 1+2 | 65-66 t/s | +5-6% | After 1A+, Layer 2 adds ~1-2% |
| Layers 1+2+3 | 66-69 t/s | +6-11% | Prefetch only helps if bubbles exist |
| Layers 1+2+3+4 | 67-71 t/s | +7-14% | Diminishing returns |

**Note**: These estimates assume we're already at ~96% HBM bandwidth utilization.
If profiling reveals lower utilization (more bubbles), gains could be higher.

---

## Layer 1: SwiGLU Epilogue Fusion

### The SwiGLU "Trap": Width Halving

SwiGLU is **not** a simple pointwise activation. It consumes two halves and produces one half:

```
Input:  [gate (N/2), up (N/2)]  →  Output: silu(gate) * up  →  N/2 elements
```

This is where most "easy EVT fusion" attempts fail:
- Standard epilogues write the same width as the accumulator
- SwiGLU must read N elements and write N/2 elements
- The accumulator slice + write logic is non-trivial

### Recommended Approach: Layer 1A Then 1B

#### Layer 1A: Aux Buffer (Lowest Risk) - RECOMMENDED FIRST

Compute SwiGLU in FC1 epilogue, write to a separate "aux" output buffer (which becomes FC2 input):

```
Current:
  FC1 GEMM → [N] → HBM → SwiGLU kernel → [N/2] → HBM → FC2

Layer 1A (BF16 aux):
  FC1 GEMM → Epilogue(SwiGLU → BF16 aux [N/2]) → HBM → quantize → FC2
           └→ Skip original [N] output

Layer 1A+ (FP8 aux, recommended):
  FC1 GEMM → Epilogue(SwiGLU → quantize → FP8 aux [N/2] + scales) → HBM → FC2
           └→ Fuses SwiGLU AND FC2-input quantization
           └→ Eliminates 2 kernels: SwiGLU + quantize_with_block_size
```

**Output format decision:**

| Option | Aux Buffer Format | FC2 Input | Kernels Eliminated |
|--------|-------------------|-----------|-------------------|
| 1A (BF16) | BF16 [N/2] | Needs separate quantize | 1 (SwiGLU) |
| **1A+ (FP8)** | **FP8 [N/2] + scales** | **Direct to FC2** | **2 (SwiGLU + quantize)** |

**Recommendation**: Target Layer 1A+ (FP8 output) to eliminate both SwiGLU and FC2 quantization in one epilogue.

**Benefits of 1A+:**
- Removes standalone SwiGLU kernel (1 launch eliminated)
- Removes FC2 input quantization kernel (1 more launch eliminated)
- Removes one read/write pass
- Preserves FC2 GEMM unchanged (clean checkpoint)
- Still has intermediate in global memory, but fewer ops

#### Layer 1B: Direct N/2 Output (More Invasive)

Modify FC1 to write only N/2 elements directly:

```
Layer 1B:
  FC1 GEMM(N cols) → Epilogue(SwiGLU + write N/2 cols) → HBM
```

**Challenges of 1B:**
- Requires modifying CUTLASS tile iteration for asymmetric output
- More complex epilogue store pattern
- Higher regression risk

**Recommendation**: Implement 1A first, validate correctness, then either:
- Proceed to Layer 1B if N/2 output is needed, OR
- Skip 1B entirely and jump to Layer 2 (SMEM B2B)

### Reference: TensorRT-LLM Fused Gated MLP

TensorRT-LLM already ships fused gated MLP kernels that handle width-halving:

```cpp
// TRT-LLM uses GemmUniversalGated for GEMM + SwiGLU fusion
cutlass::gemm::kernel::GemmUniversalGated<...>
```

**What to learn from TRT-LLM:**
- How they handle width-halving gated output
- Where they compute/store scales for FP8
- Warp scheduling for epilogue vs mainloop

**Location**: `tensorrt_llm/kernels/cutlass_kernels/` and release notes mention fused GEMM-SwiGLU for FP8 on SM90.

### CRITICAL: SwiGLU Column Layout and Operand Order

**VERIFIED: FlashInfer uses a DIFFERENT layout than cuDNN/TRT-LLM**

From `cutlass_fused_moe_kernels.cuh:2048-2052` (`doGatedActivationKernel`):

```cpp
// FlashInfer column layout: [linear (up), gate]
auto linear_value = gemm_result_vec[elem_index];           // First N/2 columns
auto gate_value = gemm_result_vec[elem_index + inter_size_vec];  // Second N/2 columns
auto gate_act = fn(gate_value, linear_value);  // fn = GLUAdaptor<SiLu>
```

From `cutlass_fused_moe_kernels.cuh:1971-1982` (`GLUAdaptor`):

```cpp
template <template <class> class ActFn>
struct GLUAdaptor {
  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    ActFn<T> fn{};
    return fn(gate) * linear;  // SiLU(gate) * linear
  }
};
```

**Comparison:**

| Framework | Column Layout | Formula | Block Pairing |
|-----------|---------------|---------|---------------|
| **FlashInfer** | `[linear, gate]` contiguous | `SiLU(gate) * linear` | None (split at N/2) |
| **cuDNN/TRT-LLM** | `[up, gate]` interleaved 32-col blocks | `up * SiLU(gate)` | Even/odd 32-col blocks |

**Key difference**: 
- FlashInfer: Simple N/2 split (columns 0:N/2 = linear, N/2:N = gate)
- cuDNN: 32-column block interleaving (blocks 0,2,4 = up, blocks 1,3,5 = gate)

**For the fused epilogue, use FlashInfer's convention:**
- First N/2 columns of accumulator = **linear (up)**
- Second N/2 columns of accumulator = **gate** (apply SiLU)
- Output formula: **SiLU(gate) * linear**
- Output has N/2 columns

**No model config check needed** — FlashInfer's convention is already fixed in the existing `doGatedActivationKernel`.

### Quantization Details (from FlashInfer `mxfp8_quantize`)

**Verified from:**
- `csrc/nv_internal/tensorrt_llm/thop/fp8Quantize.cpp:34-35`
- `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh:624-664`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Block size (SF_VEC_SIZE)** | **32** | Fixed at compile time |
| **Scale dtype** | `float8_e8m0fnu` (E8M0) | Exponent-only format |
| **Scale layout** | SWIZZLED_128x4 (default) | Tiled for GEMM alignment |
| **Row padding** | Multiple of 128 | M dimension padded |
| **Column grouping** | Tiles of 4 | K dimension grouped |

**Swizzled scale layout** (`get_sf_out_offset_128x4`):
```
SF layout: [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4 (kTile)]
  - mTileIdx = mIdx / 128
  - kTileIdx = kIdx / 4
  - outerMIdx = mIdx % 32
  - innerMIdx = (mIdx % 128) / 32
  - innerKIdx = kIdx % 4
```

This layout optimizes memory access for CUTLASS GEMM tiles.

### Dispatch Constraints for Fused Epilogue

**N dimension constraints:**

| Constraint | Reason | Fallback |
|------------|--------|----------|
| **N % 2 == 0** | SwiGLU halves N | Required (always true for MoE) |
| **N/2 >= 32** | Scale factor block size | Required for FP8 output |

**Tile alignment constraints:**

| Parameter | Constraint | Reason |
|-----------|------------|--------|
| **TILE_N** | Must be even | SwiGLU output is TILE_N/2 |
| **TILE_N** | ≥ 64 typical | Common tile sizes (64×128, 128×128) |
| **M padding** | Multiple of 128 | Scale factor swizzled layout |

**When to fall back to separate kernels:**

```python
def should_use_fused_swiglu_epilogue(N, tile_n):
    # Layer 1A (BF16 aux): relaxed constraints
    if output_dtype == torch.bfloat16:
        return N % 2 == 0  # Just need halving
    
    # Layer 1A+ (FP8 aux): stricter constraints
    if output_dtype == torch.float8_e4m3fn:
        n_out = N // 2
        return (N % 2 == 0 and 
                n_out >= 32 and        # Min for SF_VEC_SIZE
                n_out % 32 == 0)       # Clean scale blocks
    
    return False
```

**For gpt-oss-120b MoE:**
- `inter_size` = 4096 (FC1 output N before SwiGLU)
- `N/2` = 2048 (after SwiGLU)
- 2048 % 32 == 0 ✓ — FP8 quantization compatible

### Implementation Phases (Revised)

**Phase 1A: BF16 aux output (simpler, validate correctness first)**
- Fuse SwiGLU into FC1 epilogue
- Write BF16 to aux buffer
- Separate `quantize_with_block_size` kernel for FC2 input
- Eliminates 1 kernel (SwiGLU)

**Phase 1A+: FP8 aux output (full fusion)**
- Add FP8 quantization + scale computation to epilogue
- Write FP8 + scales to aux buffer
- Eliminates 2 kernels (SwiGLU + quantize)
- Must match swizzled scale layout from `get_sf_out_offset_128x4`

### Implementation (Layer 1A)

*Note: CUTLASS uses `Sm90EVT` naming even for SM120 - the epilogue visitor tree API
is shared across Hopper/Blackwell. Use the SM120 collective builders which internally
select appropriate epilogue paths.*

```cpp
// Layer 1A: SwiGLU epilogue with aux output buffer
// Uses Sm90EVT-style API (naming shared with SM120)
//
// FlashInfer column layout (from doGatedActivationKernel):
//   - Columns [0, N/2): linear (up)
//   - Columns [N/2, N): gate (SiLU applied here)
//   - Output = SiLU(gate) * linear

// Step 1: Fetch both halves of accumulator
// NOTE: FlashInfer convention - linear first, gate second
using AccFetchLinear = Sm90AccFetch;  // First N/2 columns (linear/up)
using AccFetchGate = Sm90AccFetch;    // Second N/2 columns (gate)

// Step 2: Apply SiLU to gate half
using GateSiLU = Sm90EVT<
    Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute>,
    AccFetchGate  // gate columns
>;

// Step 3: Multiply silu(gate) * linear (FlashInfer order)
using SwiGLUCompute = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute>,
    GateSiLU,       // silu(gate)
    AccFetchLinear  // linear (up)
>;

// Step 4a (Layer 1A): Store BF16 to aux buffer
using SwiGLUEpilogueBF16 = Sm90EVT<
    Sm90AuxStore<cute::bfloat16_t>,  // Write BF16 to FC2 input buffer
    SwiGLUCompute
>;

// Step 4b (Layer 1A+): Quantize to FP8 + store with scales
// NOTE: Must match swizzled scale layout from get_sf_out_offset_128x4
// This requires custom visitor or integration with quantization logic
using SwiGLUEpilogueFP8 = Sm90EVT<
    Sm90AuxStore<cute::float_e4m3_t>,  // Write FP8 to FC2 input buffer
    Sm90EVT<
        Sm90Compute<BlockScaleQuantize, cute::float_e4m3_t, ElementCompute>,
        SwiGLUCompute
    >
>;
```

### CRITICAL: The Single-Accumulator Problem

**The pseudo-EVT snippet above is aspirational — it won't work with standard tiling.**

With FlashInfer's column layout `[linear | gate]` and normal CUTLASS tiling, most epilogue invocations process a **contiguous column range** that lies entirely in either the linear half OR the gate half, not both.

```
Accumulator layout: [M, N]
  - Columns [0, N/2): linear
  - Columns [N/2, N): gate

Tile processing order (e.g., 128×128 tiles with N=4096):
  Tile 0: columns [0, 128)     → all linear, no gate available
  Tile 1: columns [128, 256)   → all linear, no gate available
  ...
  Tile 16: columns [2048, 2176) → all gate, no linear available
```

**The epilogue does NOT have both operands in registers at the same time.**

The `AccFetchLinear` and `AccFetchGate` in the pseudo-EVT require values from columns that are processed at different times. This is the crux of the implementation challenge.

### Viable Implementation Strategies

| Strategy | Description | Invasiveness | Recommendation |
|----------|-------------|--------------|----------------|
| **1. Two-accumulator gated GEMM** | FC1 as two GEMMs sharing A: `linear = A @ W_linear`, `gate = A @ W_gate`. Both accumulators available in epilogue. | LOW | **Recommended** |
| **2. Interleaved weight layout** | Repack FC1 weights to `[linear_block, gate_block, ...]` so each tile contains both operands. | HIGH | Not recommended (changes all consumers) |
| **3. Write/read intermediate** | Store linear half to SMEM/GMEM, load during gate half processing. | MEDIUM | Defeats purpose (adds memory traffic) |

**Strategy 1 (Two-Accumulator Gated GEMM)** is the right approach:

- TRT-LLM's `GemmUniversalGated` uses this pattern
- cuDNN's GEMM+SwiGLU produces two outputs (`ab12` and `c`) from what is effectively two GEMMs
- Avoids "slice the accumulator" entirely
- Cleanest route with lowest regression risk

### Implementation: Two-Accumulator Approach

Instead of a single FC1 GEMM with N columns, treat FC1 as:

```
FC1_linear: [M, K] × [K, N/2] → [M, N/2]  (linear/up output)
FC1_gate:   [M, K] × [K, N/2] → [M, N/2]  (gate output, apply SiLU)
```

**Weight layout change**: The original `[K, N]` weight is conceptually `[W_linear | W_gate]`.
Either:
- Keep weights as-is and dispatch two GEMMs with different B pointers
- OR use CUTLASS grouped GEMM with 2 groups (same A, different B slices)

**Epilogue** now has both fragments:
```cpp
// Both accumulators available simultaneously
auto linear_value = acc_linear[idx];
auto gate_value = acc_gate[idx];
auto output = silu(gate_value) * linear_value;  // SwiGLU
```

**This is what the pseudo-EVT should actually represent** — not slicing one accumulator, but combining two accumulators from a gated GEMM structure.

### Key Files to Modify

| File | Changes |
|------|---------|
| `moe_gemm_sm120_mixed_input_launcher.inl` | Modify FC1 to two-accumulator gated GEMM structure |
| `moe_kernels.h` | Add gated epilogue, `aux_output` parameter, skip `doGatedActivation()` |
| `cutlass_fused_moe_kernels.cuh` | Update FC1 dispatch to pass both B pointers (W_linear, W_gate) |
| `core.py` | Add flag to enable fused SwiGLU path |

**Weight handling**: The existing `[K, N]` FC1 weight is conceptually `[W_linear | W_gate]`. 
Pass two B pointers: `W_linear = W[:, 0:N/2]`, `W_gate = W[:, N/2:N]`.

### Risk Assessment

| Factor | Layer 1A (Two-Accumulator) | Layer 1A+ (Add FP8 Quantize) |
|--------|----------------------------|------------------------------|
| Technical complexity | MEDIUM-HIGH | MEDIUM |
| Regression risk | MEDIUM | LOW (after 1A works) |
| Implementation time | 3-5 days | 2-3 days |
| CUTLASS expertise needed | Significant | Moderate |

**Key risk for Layer 1A**: The two-accumulator gated GEMM pattern requires modifying FC1 dispatch to maintain two accumulator fragments. This is more invasive than a simple EVT change, but it's the correct architectural approach (matches TRT-LLM/cuDNN).

### Benefits

**HBM Traffic Comparison (including FC2 input quantization):**

| Flow | Writes | Reads | Total |
|------|--------|-------|-------|
| Baseline (FC1 → SwiGLU → quantize → FC2) | 64 KB (FC1 BF16) + 32 KB (SwiGLU BF16) + 16 KB (FP8) | 64 KB + 32 KB + 16 KB | 224 KB |
| Layer 1A (BF16 aux) | 32 KB (BF16 aux) + 16 KB (FP8) | 32 KB + 16 KB | 96 KB |
| **Layer 1A+ (FP8 aux)** | **16 KB (FP8 aux) + scales** | **16 KB** | **~34 KB** |
| **Savings (1A+ vs baseline)** | | | **~190 KB/layer** |

*Note: FP8 is 1 byte vs BF16 2 bytes, plus ~2 KB for block scales*

**Layer 1A+ provides:**
- Eliminates 2 kernel launches per layer: SwiGLU + quantize (48 total)
- Eliminates FC1 full-width output + SwiGLU read/write + quantize read/write
- **~190 KB HBM savings per layer × 24 = 4.5 MB per token**
- Clean checkpoint before more complex fusion

### Success Criteria (Layer 1A+)

**Why bit-identical FC2 input is unlikely with 1A+:**

The fused path has different rounding points than baseline:
- Baseline: FC1 epilogue → BF16 → SwiGLU (BF16 math) → BF16 → quantize → FP8
- Fused 1A+: FC1 epilogue → SwiGLU (FP32 accum) → quantize → FP8

Even with correct implementation, different op ordering produces different FP8 codes/scales.

**Two debug/bring-up modes:**

| Mode | Rounding Path | Use Case |
|------|---------------|----------|
| **Bit-exact** | FP32 → SwiGLU → cast to BF16 → quantize (mirrors baseline) | Phase 1A+ bring-up, `torch.equal()` validation |
| **Fast** | FP32 → SwiGLU → quantize directly | Production (possibly better numerically) |

**Bit-exact mode** explicitly replicates baseline rounding:
1. Compute SwiGLU in FP32 from accumulators
2. Cast result to BF16 (matching baseline's intermediate)
3. Run exact same quantization math to FP8+scales

This gives a real shot at `torch.equal()` for debugging. Once validated, switch to fast mode.

**Tiered success criteria:**

| Mode | Level | Criterion | Status |
|------|-------|-----------|--------|
| Bit-exact | **IDEAL** | `torch.equal(baseline_fc2_input, fused_fc2_input)` | Bring-up validation |
| Fast | **GOOD** | FC2 input: `torch.testing.assert_close()` with tolerance | Acceptable |
| Both | **REQUIRED** | FC2 output matches baseline within 1e-3 relative error | **Gate** |
| Both | **REQUIRED** | End-to-end MoE output matches within 1e-3 | **Gate** |
| Both | **REQUIRED** | Perplexity unchanged (±0.1%) | **Gate** |

**Verification steps:**
1. Start with bit-exact mode, validate `torch.equal()` on FC2 inputs
2. Switch to fast mode, validate `torch.allclose(baseline, fused, rtol=1e-3)` on FC2 outputs
3. Run perplexity test on standard eval set
4. All tile configurations pass without illegal memory access

---

## Layer 2: FC1→SMEM→FC2 Fusion

### Concept

Keep the FC1 output in shared memory instead of writing to HBM. FC2 reads from SMEM instead of HBM.

### Current Flow (after Layer 1)
```
FC1 GEMM → SwiGLU Epilogue → BF16 to HBM → FC2 reads from HBM
```

### Fused Flow
```
FC1 GEMM → SwiGLU Epilogue → FP8 to SMEM → FC2 reads from SMEM
```

### SMEM Layout

```
SM120 Total SMEM: 99,328 bytes

┌─────────────────────────────────────┐
│ FC1 Operands (A, B, SFA, SFB)       │  ~34 KB per stage
├─────────────────────────────────────┤
│ FC1 Operands Stage 1 (pingpong)     │  ~34 KB
├─────────────────────────────────────┤
│ Intermediate Buffer (FC1 output)    │  8-16 KB (FP8)
├─────────────────────────────────────┤
│ FC2 Operands (B, SFB)              │  ~17 KB (union with FC1)
├─────────────────────────────────────┤
│ Epilogue Staging                    │  ~7 KB
└─────────────────────────────────────┘
```

For a 64×128 tile: Intermediate = 64 × 128 × 1 byte (FP8) = 8 KB

### Implementation Challenges

1. **Custom SMEM Store Visitor**: Need to write `Sm120SmemStore` epilogue visitor
   - Similar complexity to existing `Sm90AuxStore`
   - Must coordinate SMEM layout with FC2 mainloop

2. **FC2 A-Operand from SMEM**: Current TMA expects HBM addresses
   - Option A: Bypass TMA for A-operand (use direct SMEM loads)
   - Option B: Write custom collective mainloop
   - Option A is simpler but may lose some TMA benefits

3. **Tile Coupling**: FC1 output N must align with FC2 input K tiles
   - Constrains tile selection flexibility
   - May need fixed tile configurations

### Key Files to Create/Modify

| File | Changes |
|------|---------|
| NEW: `sm120_smem_store_visitor.hpp` | Custom epilogue visitor for SMEM store |
| `moe_gemm_sm120_mixed_input_launcher.inl` | Add fused kernel variant |
| `core.py` | Add fused kernel dispatch |

### Risk Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Technical complexity | MEDIUM | Custom SMEM management |
| Regression risk | MEDIUM | Tight SMEM constraints |
| Implementation time | 1-2 weeks | Custom collective work |

### Benefits

**Important: Benefits depend on baseline**

| Baseline | Layer 2 Saves | Notes |
|----------|---------------|-------|
| Original (no 1A) | 192 KB/layer | Full FC1→SwiGLU→quantize→FC2 path |
| **After 1A+** | **~34 KB/layer** | Only FP8 aux write + read + scales remain |

**If starting from 1A+ (recommended path):**
- Eliminates remaining FP8 aux buffer round-trip (~34 KB/layer)
- Keeps FC2 input in SMEM instead of HBM
- Total: ~0.8 MB HBM savings per token (not 4.6 MB)
- Throughput expectation: +1-2% incremental (not +3-5%)

**Context for savings magnitude:**
After 1A+, the incremental savings are smaller. Compare against measured per-layer weight bytes
(from NCU `dram__bytes_read`) to understand the relative impact.

---

## Layer 3: Async FC2 Weight Prefetch (EXPERIMENTAL)

### Concept

While FC1 epilogue runs (including SwiGLU), start TMA loading FC2 weights. This overlaps memory loads with compute.

### CRITICAL: This Is a Measured Experiment, Not Assumed Benefit

**Overlap is only real if:**
1. **Warp roles allow it**: The warps that would issue FC2 TMA must be idle while FC1 epilogue runs (warp specialization helps)
2. **SMEM layout allows it**: FC2's SMEM tiles cannot collide with FC1 epilogue scratch/intermediate

**CUTLASS already does deep software pipelining**: The existing warp-specialized schedules (Ping-Pong, Cooperative) already achieve high utilization. You may get much less than expected if the kernel is already well-pipelined.

### Prerequisites for Overlap

| Requirement | Description | Check |
|-------------|-------------|-------|
| Idle TMA warps | Producer warps free during epilogue | Profile warp activity |
| SMEM headroom | FC2 tile buffer doesn't overlap FC1 scratch | Static analysis |
| Barrier slots | Enough async barriers for concurrent ops | CUTLASS limits |
| No bank conflicts | FC2 SMEM accesses don't conflict with epilogue | ncu analysis |

### Current Flow (after Layer 2)
```
[FC1 GEMM] → [FC1 Epilogue] → [FC2 Load] → [FC2 GEMM]
                                  ↑
                            Sequential
```

### Overlapped Flow (IF PREREQUISITES MET)
```
[FC1 GEMM] → [FC1 Epilogue  ] → [       ] → [FC2 GEMM]
             [FC2 TMA Load 0] → [FC2 TMA 1] → [ready!]
                  ↑
             Overlapped with epilogue (MEASURE THIS)
```

### Implementation

```cpp
// In fused kernel:

// FC1 mainloop completes
mma_fc1.run();

// Start async TMA for FC2 while epilogue runs
// ONLY if producer warps are free (warp specialization check)
if (is_producer_warp && epilogue_not_using_tma) {
    tma_load_async(fc2_weights_tile_0, smem_fc2_b[0]);
}

// FC1 epilogue (includes SwiGLU, stores to SMEM)
epilogue_fc1.run();

// Sync first FC2 tile
tma_barrier_arrive_expect_tx(fc2_tile_0_bytes);

// FC2 mainloop - first tile already in SMEM!
mma_fc2.run();
```

### Validation: PROVE Overlap Before Claiming Benefit

**Required evidence (nsys + ncu):**

| Metric | Tool | Success Criteria |
|--------|------|------------------|
| FC2 TMA overlaps FC1 epilogue | nsys timeline | Visual confirmation of overlap |
| No SMEM bank conflicts added | ncu `smsp__sass_l1tex_data_bank_conflicts` | Conflicts ≤ baseline |
| Occupancy not reduced | ncu `sm__warps_active` | Occupancy ≥ baseline |
| HBM bandwidth unchanged | ncu `dram__bytes_read` | Not worse than baseline |
| Actual latency reduction | nsys duration | FC1+FC2 faster than sum |

### Benefit Calculation

- FC1 epilogue time: ~5-10 µs per CTA
- FC2 tile load time: ~3-5 µs per tile
- Overlap: 1-2 FC2 tiles loaded "for free" **IF warp roles allow**
- **Theoretical savings**: 3-10 µs × 8 experts × 24 layers = 0.6-2 ms

**Reality check**: At ~96% HBM utilization, prefetching cannot "create" new bandwidth.
It can only redistribute existing bandwidth to hide latency.

**Expected outcome**: 0-2% benefit. If profiling shows <1% benefit after implementation, consider skipping this layer entirely.

### Risk Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Technical complexity | HIGH | Async barrier coordination |
| Regression risk | MEDIUM-HIGH | May reduce occupancy or add conflicts |
| Implementation time | 1-2 weeks | Careful synchronization |
| Expected benefit | LOW | CUTLASS already well-pipelined |

### Decision Point

After Layer 2 is complete, run ncu analysis to check:
1. Are there idle TMA warp cycles during epilogue?
2. Is there SMEM headroom for FC2 tile buffer?

If both are YES, proceed with Layer 3 as experiment. If NO, skip to Layer 4 or stop.

### Hard Stop Condition

**Skip Layer 3 entirely if:**
```
ncu dram__throughput.avg.pct_of_peak_sustained_elapsed > 90%
  AND
FC1 epilogue duration < 5 µs
```

If HBM is already near peak throughput and the epilogue provides minimal slack time, there is no opportunity for overlap. Prefetching cannot beat physics - it can only redistribute existing bandwidth.

---

## Layer 4: Persistent/Warp-Specialized Kernel

### Why "All 24 Layers in One Kernel" Is NOT Feasible

MoE blocks are interleaved with attention/residual/norm in transformer layers:

```
Layer N:  Attention → Add+Norm → MoE → Add+Norm
Layer N+1: Attention → Add+Norm → MoE → Add+Norm
```

A single kernel cannot span multiple layers because:
1. Attention and norms must execute between MoE blocks
2. Residual connections require HBM round-trips
3. The layer dependency chain cannot be hidden

### Practical Launch-Overhead Solutions

Replace the original "mega-kernel" concept with two practical alternatives:

#### Option 4A: CUDA Graph Capture for MoE Subgraph

If the MoE path can be made **shape-stable** (or bucketed), CUDA graphs eliminate launch overhead without mega-kernel debugging pain.

**Current state**: vLLM already captures CUDA graphs for decode:
```python
# vLLM's current approach (cudagraph_mode=FULL_AND_PIECEWISE)
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 24, 32, ...]
```

**The opportunity**: After fusing FC1+SwiGLU+FC2 (Layers 1-2), the MoE subgraph becomes:
```
Single fused kernel per layer × 24 layers = 24 kernels (down from ~264 MoE kernels)
```

With fewer, larger kernels, CUDA graph capture becomes more efficient:
- Less capture overhead per kernel
- Better graph replay performance
- Memory footprint reduction

**What to check:**
1. Are MoE kernel shapes stable across decode steps? (Yes, for batch=1)
2. Are expert assignments causing graph invalidation? (Check dynamic dispatch)
3. Can we bucket by expert count for variable routing?

**Implementation:**
```python
# Ensure fused MoE kernel is captured in CUDA graph
# After Layers 1-2, the MoE is a single kernel:
#   fused_moe_fc1_swiglu_fc2(input, weights, output)
#
# This naturally fits in existing CUDA graph capture
```

**Benefits:**
- Launch overhead already amortized by existing graphs
- No custom kernel development needed
- Stable, production-ready approach

#### Option 4B: Persistent Tile Scheduler Inside One MoE Layer

Use CUTLASS's warp-specialized persistent scheduler to process all expert tiles within a single MoE layer. This is the "practical persistent" that maps to our use case.

**Reference**: CUTLASS changelogs explicitly describe:
- Persistent cooperative designs
- Improved tile scheduling/rasterization
- Warp-specialized kernels with persistent work distribution

```cpp
// CUTLASS 3.x Persistent Grouped GEMM pattern
// Reference: cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp

template <class ProblemShape, class CollectiveMma, class CollectiveEpilogue>
struct PersistentMoeKernel {
    using TileScheduler = typename cutlass::gemm::PersistentScheduler;
    
    struct SharedStorage {
        typename CollectiveMma::SharedStorage mma;
        typename CollectiveEpilogue::SharedStorage epilogue;
        typename TileScheduler::SharedStorage scheduler;
    };
    
    __device__ void operator()(Params const& params, char* smem_ptr) {
        TileScheduler scheduler(params.scheduler);
        
        // Persistent loop: CTAs fetch work until all tiles done
        for (auto work_tile = scheduler.get_current_work();
             work_tile.is_valid();
             work_tile = scheduler.advance_to_next_work()) {
            
            // Determine which expert this tile belongs to
            int expert_idx = work_tile.L;  // Group index = expert
            
            // Process FC1 tile for this expert
            collective_mma_fc1(work_tile, params.fc1[expert_idx], smem_ptr);
            
            // SwiGLU in registers (fused epilogue)
            apply_swiglu_epilogue();
            
            // Process FC2 tile (same expert, same CTA)
            collective_mma_fc2(work_tile, params.fc2[expert_idx], smem_ptr);
        }
    }
};
```

**Benefits:**
- Eliminates per-tile launch overhead within MoE layer
- Reduces MoE kernels from 3 (FC1+SwiGLU+FC2) to 1 per layer
- Aligns with CUTLASS Hopper/Blackwell persistent patterns
- No grid sync or cooperative launch required

**Saves:** ~48 kernel launches per forward → ~240 µs

### Recommendation: 4A First, 4B Only If Needed

**IMPORTANT: 4A is a mandatory check after Layer 1A, not a later optimization.**

See "MANDATORY: CUDA Graph Verification" in the Implementation Roadmap.

| Approach | Effort | Benefit | When to Use |
|----------|--------|---------|-------------|
| **4A check** | **Mandatory** | Determines if 4B needed | **Run after Layer 1A** |
| 4B only | Medium | Extra 1-2% if graphs insufficient | If 4A shows remaining overhead |
| Skip 4B | None | N/A | If graphs already capture cleanly |

**Practical path:**
1. Complete Layer 1A (fused FC1+SwiGLU+FC2 aux buffer)
2. **IMMEDIATELY verify CUDA graphs still capture cleanly (4A)**
3. If graphs work and launch overhead <0.5 ms, **skip 4B entirely**
4. Only implement 4B if graphs fail or show >0.5 ms residual overhead
5. Continue to Layers 2-3 knowing launch overhead status

### Risk Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Technical complexity | HIGH | Requires CUTLASS persistent scheduler |
| Regression risk | MEDIUM | Well-tested CUTLASS patterns exist |
| Implementation time | 1-2 weeks | Build on CUTLASS Ping-Pong examples |

---

## CUTLASS Extension Points

### EVT (Epilogue Visitor Tree)

CUTLASS 3.x provides composable epilogue operations:

```cpp
// Example: Linear combination with activation
using LinearCombinationWithAct = Sm90EVT<
    Sm90Compute<activation_fn, Output, Compute>,  // Apply activation
        Sm90EVT<
            Sm90Compute<multiply_add, Compute, Compute>,  // alpha * acc + beta * C
            Sm90ScalarBroadcast<Scalar>,  // alpha
            Sm90AccFetch,  // accumulator
            Sm90ScalarBroadcast<Scalar>,  // beta
            Sm90SrcFetch<Element>  // C
        >
>;
```

### What's Supported vs Custom

| Operation | Built-in? | Notes |
|-----------|-----------|-------|
| Linear combination | Yes | `Sm90LinearCombination` |
| Activation (ReLU, GELU) | Yes | `Sm90Compute<relu, ...>` |
| Bias addition | Yes | `Sm90ColBroadcast` |
| Scale factors | Yes | `Sm90ScalarBroadcast` |
| Aux store (to separate buffer) | Yes | `Sm90AuxStore` |
| **SwiGLU (gated)** | **Partial** | Need custom EVT tree |
| **Store to SMEM** | **No** | Need custom visitor |

---

## Risk Summary

| Layer | Risk | Probability of Clean Implementation |
|-------|------|-------------------------------------|
| Layer 1A: SwiGLU aux buffer | MEDIUM | 70-80% |
| Layer 1B: SwiGLU N/2 output | HIGH | 40-60% |
| Layer 2: SMEM fusion | MEDIUM | 50-70% |
| Layer 3: Async overlap | HIGH | 30-50% |
| Layer 4: Persistence | HIGH | 40-60% |

### Recommendation

Start with **Layer 1A** (aux buffer approach). It provides:
- Removes SwiGLU kernel without modifying FC1 output shape
- Clean checkpoint that preserves FC2 unchanged
- Natural place to add FC2-input FP8 quantization
- Lower risk than direct N/2 output modification

If Layer 1A works cleanly:
- Either proceed to Layer 1B (if N/2 output needed), OR
- Skip directly to Layer 2 (SMEM B2B) once correctness is validated

### Key References Before Starting

#### cuDNN Frontend (Local: `~/projects/cudnn-frontend/`)

**Primary reference for SwiGLU implementation:**

1. **Dense GEMM+SwiGLU test**: `test/python/fe_api/test_gemm_swiglu.py`
   - Reference implementation in `test_gemm_swiglu_utils.py`
   
2. **Grouped GEMM+SwiGLU (MoE-specific)**: `test/python/fe_api/test_grouped_gemm_swiglu.py`
   - Reference implementation in `test_grouped_gemm_swiglu_utils.py` (lines 452-669)
   - Uses `GroupedGemmSwigluSm100` API
   - Includes `tile_idx_to_expert_idx` mapping for MoE routing

**Key implementation from `test_grouped_gemm_swiglu_utils.py:run_grouped_gemm_swiglu_ref()`:**

```python
# Step 3: Apply SwiGLU with interleaved block layout (lines 516-532)
group = 32
num_blocks = n // group
assert n % group == 0, "N must be divisible by 32"
assert num_blocks % 2 == 0, "Number of 32-col blocks must be even"

cols = torch.arange(n)
block_cols = cols.view(num_blocks, group)
up_idx = block_cols[0::2].reshape(-1)    # Blocks 0,2,4,6,... (even)
gate_idx = block_cols[1::2].reshape(-1)  # Blocks 1,3,5,7,... (odd)

ref_up = ref.index_select(1, up_idx)
ref_gate = ref.index_select(1, gate_idx)

# SwiGLU: up * swish(gate) = up * (gate * sigmoid(gate))
ref_gate = ref_gate * torch.sigmoid(ref_gate)
ref_after_swiglu = ref_up * ref_gate
```

**Grouped GEMM SwiGLU output tensors:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `c_tensor` | (M, N, 1) | Intermediate GEMM result (always BF16) |
| `d_tensor` | (M, N/2, 1) | SwiGLU output (FP8 or BF16) |
| `sfd_row_tensor` | scale shape | Row-major scale factors (if FP8 output) |
| `sfd_col_tensor` | scale shape | Col-major scale factors (if FP8 output) |
| `amax_tensor` | (L, 1) | Per-group max (if BF16 output) |

**Block scaling parameters (from test fixtures):**
- FP8 mode: `sf_vec_size=32`, `sf_dtype=torch.float8_e8m0fnu`
- FP4 mode: `sf_vec_size=16` or `32`, `sf_dtype=float8_e8m0fnu` or `float8_e4m3fn`

#### Other References

3. **cuDNN docs**: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_swiglu.html
4. **TensorRT-LLM `GemmUniversalGated`**: Study their width-halving pattern
5. **CUTLASS EVT tutorials**: `examples/49_hopper_gemm_with_collective_builder`
6. **CUTLASS Sm90EVT docs**: `include/cutlass/epilogue/fusion/sm90_visitor_*.hpp`
7. **Colfax EVT tutorial**: https://research.colfax-intl.com/epilogue_visitor_tree (topological visitors for DAGs)

---

## Measurement Plan

Single reference table for all pass/fail decisions:

| Layer | Mode | Metric | Tool | Pass Threshold | Revert If |
|-------|------|--------|------|----------------|-----------|
| **1A (BF16)** | - | SwiGLU output identical | `torch.equal()` | Binary match | Debug fusion |
| **1A (BF16)** | - | SwiGLU kernel eliminated | `nsys` | Not in trace | Still present |
| **1A (BF16)** | - | Decode throughput | `llama-benchy` | ≥63 t/s (+1%) | <62 t/s (0%) |
| **1A+ (FP8)** | Bit-exact | FC2 input matches (BF16→FP8 path) | `torch.equal()` | Binary match | Debug quantize |
| **1A+ (FP8)** | Fast | FC2 input close (FP32→FP8 path) | `torch.testing.assert_close()` | rtol=1e-3 | **>1e-2 error** |
| **1A+ (FP8)** | Both | FC2 output matches baseline | `torch.allclose(rtol=1e-3)` | Within tolerance | **>1e-3 error** |
| **1A+ (FP8)** | Both | Perplexity unchanged | Eval script | ±0.1% | **>0.5% regression** |
| **1A+ (FP8)** | Both | Decode throughput | `llama-benchy` | ≥64 t/s (+3%) | <63 t/s (<2%) |
| **1A+ (FP8)** | Both | Both kernels eliminated | `nsys` | SwiGLU + quantize gone | Still present |
| **Graph** | - | Graph captures fused kernel | `nsys --cudagraph` | No capture errors | Graph fails |
| **Graph** | - | Launch overhead | `nsys` | <0.5 ms | >1 ms *(consider 4B)* |
| **2** | - | SMEM bounds | Launch success + `compute-sanitizer` | No illegal access | Crash/overflow |
| **2** | - | Decode throughput | `llama-benchy` | ≥65 t/s (+5%) | <64 t/s (<3%) |
| **3** | - | TMA overlap visible | `nsys` timeline | FC2 load during epilogue | No overlap |
| **3** | - | HBM throughput | `ncu dram__throughput` | Not decreased | >5% decrease |
| **3** | - | Decode throughput | `llama-benchy` | ≥66 t/s (+6%) | <65 t/s (<5%) |
| **4B** | - | Persistent scheduler works | Unit test | Correct output | Wrong output |
| **4B** | - | Decode throughput | `llama-benchy` | ≥67 t/s (+8%) | <66 t/s (<6%) |

*Note: Layer 2 expectations reduced since 1A+ already captures most activation savings.*

**Bring-up workflow:**
1. **Phase 1A (BF16)**: Fuse SwiGLU into epilogue with BF16 output
   - Use FlashInfer convention: `SiLU(gate) * linear`, columns `[linear, gate]`
   - Validate `torch.equal()` with baseline `doGatedActivationKernel` output
   - Eliminates 1 kernel (SwiGLU)

2. **Phase 1A+ (FP8 bit-exact)**: Add quantization with BF16 intermediate
   - Cast SwiGLU output to BF16 before quantizing (match baseline rounding)
   - Validate `torch.equal()` on FC2 FP8 inputs
   
3. **Phase 1A+ (FP8 fast)**: Direct FP32→FP8 quantization
   - Remove BF16 cast, validate tolerance-based gates
   - Production mode (possibly better numerically)

**Revert policy:** If "Revert If" condition is met after reasonable debugging effort (max 2 days), revert and document why.

---

## Implementation Roadmap

### Phase 1A: SwiGLU BF16 Aux Buffer Fusion (3-5 days)

**VERIFIED convention (no model config check needed):**
- FlashInfer column layout: `[linear, gate]` (linear first, gate second)
- Formula: `SiLU(gate) * linear`
- Block size for quantization: 32 (SF_VEC_SIZE = 32)
- Scale layout: SWIZZLED_128x4

**Architecture: Two-Accumulator Gated GEMM**

FC1 becomes two logical GEMMs sharing the A operand:
```
FC1_linear: A[M,K] × W_linear[K,N/2] → acc_linear[M,N/2]
FC1_gate:   A[M,K] × W_gate[K,N/2]   → acc_gate[M,N/2]
```

The epilogue has both accumulators available simultaneously:
```cpp
output[i] = silu(acc_gate[i]) * acc_linear[i];
```

**Implementation steps:**

1. **Study existing code:**
   - `doGatedActivationKernel` in `cutlass_fused_moe_kernels.cuh:2003-2055`
   - `GLUAdaptor` struct at line 1971 (confirms `fn(gate) * linear`)
   - FC1 epilogue dispatch in `moe_gemm_sm120_mixed_input_launcher.inl`
   - TRT-LLM `GemmUniversalGated` pattern (two accumulators)

2. **Modify FC1 GEMM to gated structure:**
   - Option A: Two separate GEMM dispatches with same A, different B pointers
   - Option B: CUTLASS grouped GEMM with 2 groups (linear and gate)
   - Option C: Custom collective that maintains two accumulator fragments
   - **Start with Option A** (simplest, validates the approach)

3. **Create gated epilogue:**
   - Receive both `acc_linear` and `acc_gate` fragments
   - Apply SiLU to gate, multiply with linear
   - Store BF16 result to aux buffer

4. **Integration:**
   - Add `aux_output` parameter to FC1 GEMM dispatch
   - Add dispatch flag in `core.py` (`fuse_swiglu=True`)
   - Skip `doGatedActivation()` when fused

5. **Validation (bit-exact mode):**
   - Compare fused BF16 output vs baseline `doGatedActivationKernel` output
   - Use `torch.equal()` — should be bit-identical since both use BF16 output

6. **Benchmark:**
   - Verify SwiGLU kernel eliminated in `nsys` trace
   - Expect ~1-2% improvement (SwiGLU kernel overhead removed)

**Gate criteria**: `torch.equal()` with baseline SwiGLU output, all tile configs working

### Phase 1A+: FP8 Aux Buffer with Scales (additional 2-3 days)

**After Phase 1A is validated:**

1. **Add FP8 quantization to epilogue:**
   - Compute amax over SwiGLU output
   - Compute scale factors (SF_VEC_SIZE = 32)
   - Quantize to FP8 (E4M3)
   - Store FP8 + scales to aux buffers

2. **Match swizzled scale layout:**
   - Use `get_sf_out_offset_128x4` indexing from `quantization.cuh:624-664`
   - Row padding to multiple of 128
   - Column grouping in tiles of 4

3. **Validation:**
   - Bit-exact mode: Cast SwiGLU to BF16 before quantizing (match baseline rounding)
   - Fast mode: Direct FP32→FP8 quantization
   - Compare FC2 output within tolerance

4. **Benchmark:**
   - Verify both SwiGLU + quantize kernels eliminated
   - Expect additional ~1-2% improvement

**Gate criteria**: FC2 output within 1e-3 of baseline, perplexity unchanged

### MANDATORY: CUDA Graph Verification (after Phase 1A)

**Check immediately after Layer 1A works:**
1. Run vLLM with fused kernel, verify CUDA graphs still capture
2. Profile graph replay vs eager execution: `nsys` trace with/without `--enforce-eager`
3. Measure launch overhead: compare graph replay time vs individual kernel launches

**Decision matrix:**

| Graph Status | Launch Overhead | Action |
|--------------|-----------------|--------|
| Graphs capture cleanly | <0.5 ms | **Skip persistent scheduler (4B)** - graphs already solve it |
| Graphs capture cleanly | >0.5 ms | Consider 4B for remaining overhead |
| Graphs fail to capture | Any | Debug graph issues OR implement 4B as workaround |

**If graphs eliminate most launch overhead, down-rank all Layer 4B work.**

### Phase 1B (Optional): Direct N/2 Output (1 week)

1. Modify epilogue to write N/2 columns instead of N
2. Adjust tile iteration for asymmetric output
3. Integration testing
4. Skip if Layer 2 is the next target

**Gate criteria**: Numerical accuracy within 1e-3, no performance regression

### Phase 2: SMEM Fusion (1-2 weeks)

1. Design SMEM layout for intermediate buffer
2. Create `Sm120SmemStore` epilogue visitor
3. Modify FC2 mainloop to read from SMEM
4. Integration testing
5. Benchmark: verify 3-5% improvement (cumulative 6-9%)

**Gate criteria**: All tile configurations working, no SMEM overflow

### Phase 3: Async Overlap (EXPERIMENTAL, 1-2 weeks)

**Pre-implementation profiling (REQUIRED):**
1. Run ncu on Layer 2 kernel to check idle TMA warp cycles during epilogue
2. Analyze SMEM layout for headroom to add FC2 tile buffer
3. **Decision point**: If no idle cycles or no SMEM headroom, SKIP Phase 3

**If proceeding:**
1. Add async TMA loads during epilogue
2. Implement barrier coordination
3. Profile overlap effectiveness with nsys timeline
4. Verify no SMEM bank conflicts added (ncu)
5. Benchmark: verify additional 0-2% improvement

**Gate criteria**: 
- nsys shows visual overlap of FC2 TMA with FC1 epilogue
- ncu shows no increase in bank conflicts or reduction in occupancy
- Actual speedup > 0.5%, otherwise revert

**Expected outcome**: 0-2% benefit. If <0.5% after implementation, consider reverting.

### Phase 4: Launch Overhead Reduction (1-2 weeks)

**Phase 4A: CUDA Graph Verification (Low effort)**
1. After Layers 1-2, verify CUDA graphs still capture cleanly
2. Check that fused MoE kernel shapes are graph-compatible
3. Profile graph replay vs eager execution
4. If graphs work well, Phase 4 may be complete

**Phase 4B: Persistent Tile Scheduler (If needed)**
1. Only proceed if Phase 4A shows remaining launch overhead
2. Implement CUTLASS `PersistentScheduler` for grouped GEMM
3. Integrate with fused FC1+SwiGLU+FC2 kernel
4. Benchmark: verify additional 1-2% improvement

**Gate criteria**: 
- 4A: CUDA graph capture succeeds, replay overhead < 0.5 ms
- 4B: Correct output for all input lengths, stable performance

**Expected outcome**: Most benefit from existing graphs (4A). 4B only if graphs insufficient.

---

## Testing Strategy

### Unit Tests

1. **Numerical accuracy**: Compare fused output vs baseline
   - Relative error < 1e-3 for BF16
   - Test various input shapes (M=1, 16, 64, 128)

2. **SMEM bounds**: Verify no buffer overflow
   - Test all supported tile configurations
   - Check for illegal memory access

### Integration Tests

1. **vLLM end-to-end**: Run `llama-benchy` with fused kernel
   - Compare output quality (perplexity)
   - Verify no hangs or crashes

2. **Context lengths**: Test short (512), medium (2048), long (8192)

### Performance Tests

1. **Decode benchmark**: `llama-benchy --pp 2048 --tg 32 128`
2. **Prefill benchmark**: `llama-benchy --pp 512 2048 8192 --tg 1`
3. **Profile**: `nsys` trace to verify overlap

---

## Appendix: Key Files Reference

### FlashInfer MoE Implementation

| File | Purpose |
|------|---------|
| `flashinfer/fused_moe/core.py` | Python API, tile selection |
| `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | Kernel dispatch |
| `csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl` | SM120 launcher |
| `csrc/nv_internal/.../include/moe_kernels.h` | `gemm1()`, `gemm2()`, `doGatedActivation()` |

### CUTLASS References

| File | Purpose |
|------|---------|
| `cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` | EVT definitions |
| `cutlass/epilogue/fusion/operations.hpp` | Fusion operation base classes |
| `cutlass/examples/13_two_tensor_op_fusion/` | B2B GEMM example (SM80) |

---

## Related Documents

- **[LAYER_1A_SPIKE_PLAN.md](./LAYER_1A_SPIKE_PLAN.md)** - Detailed 6-step spike plan for Layer 1A validation
- **[scripts/debug/test_gated_fc1_spike.py](../../scripts/debug/test_gated_fc1_spike.py)** - Spike validation script

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-01 | AI Assistant | Added spike plan, created validation script |
| 2026-01-31 | AI Assistant | Initial document |
