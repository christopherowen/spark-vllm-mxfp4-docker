# Tencent HPC-Ops Analysis

Deep dive analysis of [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops) for techniques applicable to our SM121 MXFP4 optimization work.

## Executive Summary

HPC-Ops is a production-grade CUDA kernel library from Tencent's Hunyuan AI Infra team, achieving impressive speedups on H20 (SM90):
- **Attention Decode**: 2.22x over FlashInfer/FA2/FA3
- **FusedMoE (FP8)**: 1.49x prefill, 1.14x decode
- **GroupGEMM (FP8)**: 1.88x decode over DeepGEMM

While their kernels are SM90-specific, **some techniques are applicable** to our SM121 work. However, FlashInfer already implements many of the same optimizations.

### Key Finding: FlashInfer Already Optimized

After reviewing both codebases, FlashInfer already has:
- ✅ `math.cuh` with `ptx_exp2`, `ptx_log2`, `ptx_rcp` (same as hpc-ops)
- ✅ SM120 attention uses `exp2f()` with log2 scaling
- ✅ `-use_fast_math` compiler flag enabled
- ❌ MoE activation NOT fused with quantization (gap exists)
- ❌ No specialized small-M warp reductions (gap exists)

---

## Key Techniques to Adopt

### 1. Warp-Specialized Producer-Consumer Pipeline

**Their Approach:**
```cpp
// Load warpgroup (producer)
if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();  // Reduce register pressure
    // ... TMA loads with barrier synchronization
}
// Math warpgroup (consumer)  
else {
    cutlass::arch::warpgroup_reg_alloc<232>();  // More registers for compute
    // ... GEMM with GMMA
}
```

**Key Insight:** They split thread blocks into dedicated load and compute warpgroups:
- **Load warpgroup**: Fewer registers (24), focused on TMA async loads
- **Math warpgroup**: More registers (168-232), focused on tensor core GEMM

**Applicability to SM121:**
Our FlashInfer MoE kernels could benefit from explicit warp specialization. Currently we use CUTLASS's built-in scheduling which may not be as aggressive.

### 2. Multi-Stage Software Pipelining with Barriers

**Their Pattern:**
```cpp
__shared__ uint64_t writable[kStage];  // Producer can write here
__shared__ uint64_t readable[kStage];  // Consumer can read here

// Producer side
wait_barrier(writable[istage], phase);
copy(tma.with(readable[istage]), src, dst);
set_barrier_transaction_bytes(readable[istage], kTransactionBytes);

// Consumer side
wait_barrier(readable[istage], phase);
// ... use data ...
arrive_barrier(writable[istage]);
```

**Key Insight:** They use a **double-buffering/multi-stage** approach with explicit barrier management rather than relying on synchronous loads. This achieves:
- Overlapping of load and compute
- Reduced stalls waiting for memory

**Applicability to SM121:**
Our attention decode could use this pattern to overlap KV cache loading with attention computation.

### 3. Online Softmax with log2 Fast Math

**Their Approach (from smallm_kernels.cuh):**
```cpp
// Use log2 space for numerical stability + speed
float one_over_dk_log2e = 1.0f / (sqrt(dk) * log2(e));

// exp2 is faster than exp (single PTX instruction)
tAttr(im, in) = exp2f_ftz(tAttr(im, in) * one_over_dk_log2e - gMax(in));
```

**FlashInfer Already Does This!**

FlashInfer's `math.cuh` (lines 42-46, 52-56, 83-86):
```cpp
// flashinfer/include/flashinfer/math.cuh
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

And SM120 attention (`csrc/xqa/mla_sm120.cu`, line 925) uses log2 space:
```cpp
x(m, n)(i, j) = exp2f(elem * qkScaleLog2e - maxVal);
```

**Status: ✅ FlashInfer already optimized** - No action needed for attention softmax.

### 4. Specialized Warp Reductions

**Their Approach:**
```cpp
// 4-lane reduction (for small groups)
__device__ __forceinline__ float warp_4lane_reduce_max_xor(float x) {
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 1), x);
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 2), x);
    return x;
}

// 8-lane strided reduction (for specific attention patterns)
__device__ __forceinline__ float warp_8lane_stride4_reduce_max_xor(float x) {
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 4), x);
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 8), x);
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 16), x);
    return x;
}
```

**Key Insight:** They have specialized reduction patterns for different data layouts, avoiding unnecessary shuffles when only partial warp reduction is needed.

**Applicability to SM121:**
For decode attention with small M, we could use 4-lane or 8-lane reductions instead of full warp reductions.

### 5. Vectorized Load/Store Templates

**Their vec_t Utility:**
```cpp
template <typename T, int N>
struct vec_t {
    T data[N];
    __device__ constexpr T& operator[](int idx) { return data[idx]; }
};

template <typename T, int N>
__device__ __forceinline__ auto load(const void *ptr) {
    vec_t<T, N> v;
    constexpr int kBytes = sizeof(T) * N;
    if constexpr (kBytes == 16) {
        *reinterpret_cast<uint4*>(&v) = *reinterpret_cast<const uint4*>(ptr);
    }
    // ...
    return v;
}
```

**Key Insight:** Templated vector loads that compile to optimal width (1/2/4/8/16 bytes) based on type and count.

**Applicability to SM121:**
A unified vec_t approach could simplify our FP8/FP4 data movement code.

### 6. FusedMoE Pipeline Structure

**Their fuse_moe.cu Pattern:**
```cpp
void fuse_moe_pertensor_fp8_async(...) {
    // 0. Count and gather tokens per expert
    count_and_gather_async(...);
    
    // 1. Gate-up GEMM (fused gate + up projection)
    group_gemm_pertensor_fp8_async(...);
    
    // 2. Activation (SiLU) with in-place multiply + quantize
    act_mul_and_quant_async(...);
    
    // 3. Down projection GEMM
    group_gemm_pertensor_fp8_async(...);
    
    // 4. Reduce scattered outputs back to sequence order
    reduce_async(...);
}
```

**Key Insight:** Their MoE is a **5-kernel pipeline** with:
- Explicit token counting/gathering
- GroupGEMM for variable-size batches per expert
- Fused activation + quantization (avoids memory round-trip)
- Final reduction with topk weights

**Applicability to SM121:**
Our FlashInfer MoE could fuse the SiLU activation with quantization to avoid the extra memory traffic we currently have from separate activation kernels.

### 7. Dynamic TMA Descriptor Updates

**Their Grouped GEMM TMA Handling:**
```cpp
// Update TMA descriptors dynamically per-group
__global__ void update_grouped_tma(...) {
    // For each group (expert), update TMA descriptors with correct pointers
    if (idx == 0) {
        auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr), ...);
        update_tma_gtensor<TmaX>(smem_tma_desc[idx], gX);
    }
    // ...
    tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
}
```

**Key Insight:** They dynamically update TMA descriptors for each expert group, allowing a single kernel launch to handle variable per-expert token counts.

**Applicability to SM121:**
Our MoE could benefit from dynamic TMA updates instead of multiple kernel launches or padding.

### 8. Multimem Load-Reduce for All-Reduce

**Their Approach (utils.cuh):**
```cpp
template <typename T, int N>
__device__ __forceinline__ auto multi_load_reduce_add(const void *ptr) {
    if constexpr (std::is_same_v<T, __nv_bfloat162> && N == 4) {
        vec_t<T, N> v;
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2"
            " {%0,%1,%2,%3}, [%4];"
            : "=r"(l->x), "=r"(l->y), "=r"(l->z), "=r"(l->w)
            : "l"(ptr)
            : "memory");
        return v;
    }
}
```

**Key Insight:** They use `multimem.ld_reduce` PTX instructions for **hardware-accelerated all-reduce** operations, avoiding separate load + add + store.

**Applicability to SM121/TP=2:**
This could be used for the final MoE reduction or for TP all-reduce operations, potentially reducing our NCCL overhead.

### 9. Small-M Decode Attention Kernel Variants

**Their Kernel Variants:**
- `m64_dim128.cu` - Standard 64-token decode attention
- `smallm_dim128.cu` - Optimized for very small M (1-16 tokens)
- `smallm_splitk_dim128.cu` - Split-K for memory-bound decode

**Key Insight:** They have **specialized kernels for different M sizes**, not just tile shape selection within one kernel.

**Applicability to SM121:**
Our decode attention (M=1 typically) could benefit from a specialized "M=1" kernel path rather than using the general-purpose attention kernel.

---

## Comparison: HPC-Ops vs FlashInfer (Our Stack)

| Aspect | HPC-Ops (SM90) | FlashInfer (SM121) | Gap? |
|--------|----------------|---------------------|------|
| **Attention** | Custom CUTE/CUTLASS | FlashInfer FA2/MLA | Similar |
| **MoE** | Fused 5-kernel pipeline | CUTLASS grouped GEMM | Similar |
| **Softmax** | log2 space + exp2f_ftz | log2 space + exp2f ✅ | **None** |
| **Fast Math** | Custom PTX intrinsics | `math.cuh` with same PTX ✅ | **None** |
| **Load/Compute Overlap** | Explicit warp specialization | CUTLASS implicit | Minor |
| **TMA** | Dynamic descriptor updates | Fixed descriptors | Minor |
| **Precision** | FP8 E4M3 | FP8×FP4 (MXFP4) | Different |
| **Activation Fusion** | SiLU + quant fused | SiLU + quant fused for FP4/FP8 ✅ | **None** |
| **Small-M Reductions** | 4-lane/8-lane specialized | Full 32-lane warp | **Gap** |

---

## Recommended Actions

### Already Implemented in FlashInfer (No Action Needed)

1. ~~**Adopt log2 Softmax Space**~~ ✅
   - FlashInfer's `math.cuh` already has `ptx_exp2`, `ptx_log2`, `ptx_rcp`
   - SM120 attention uses `exp2f(elem * qkScaleLog2e - maxVal)`

2. ~~**Add Fast Math Utilities**~~ ✅
   - FlashInfer already has `math.cuh` with PTX intrinsics
   - Compiler flag `-use_fast_math` already enabled

### High Priority (Actual Gaps)

1. ~~**Fuse Activation + Quantization in MoE**~~ ✅ **Already Implemented!**
   - FlashInfer's `doActivationKernel` (lines 2081-2287) **already fuses SiLU with FP4/FP8 quantization**
   - See lines 2208-2218:
     ```cpp
     if constexpr (IsNVFP4 || IsMXFP8) {
       auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize>(
           post_act_val, global_scale_val, ...);
       output_vec[elem_index] = res;
     }
     ```
   - The older `doGatedActivationKernel` (BF16→BF16) is only used for non-quantized paths

2. **Small-M Warp Reductions for Decode Attention**
   - FlashInfer uses full 32-lane warp reductions everywhere
   - hpc-ops has specialized 4-lane and 8-lane reductions for M=1 decode
   - For decode attention with M=1, only need 4-lane reduction (2 shuffles vs 5)
   - Expected: Minor speedup (~2-5%) for decode attention

### Medium Priority (Architecture Improvements)

3. **Explicit Warp Specialization for Decode**
   - hpc-ops explicitly splits thread blocks: load warpgroup (24 regs) vs math warpgroup (232 regs)
   - FlashInfer relies on CUTLASS's implicit scheduling
   - May improve decode latency through better resource utilization

4. **Dynamic TMA Descriptor Updates for MoE**
   - hpc-ops updates TMA descriptors per-expert dynamically
   - Could reduce kernel launch overhead for variable expert batch sizes

### Lower Priority (Future Optimization)

5. **Multimem Operations for TP All-Reduce**
   - `multimem.ld_reduce` for hardware-accelerated all-reduce
   - May reduce TP=2 NCCL overhead

---

## Code Snippets to Consider

### FlashInfer Already Has Fast Math (No Porting Needed)

FlashInfer's `include/flashinfer/math.cuh` already contains:
```cpp
// Already exists in FlashInfer!
constexpr float log2e = 1.44269504088896340736f;

__forceinline__ __device__ float ptx_exp2(float x) {
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float ptx_log2(float x) {
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float ptx_rcp(float x) {
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

### Potentially Useful: Small-M Warp Reductions

These could be added to FlashInfer for decode attention optimization:
```cpp
// 4-lane reduction (for M=1 decode attention)
// Only 2 shuffles instead of 5 for full warp reduction
__device__ __forceinline__ float warp_4lane_reduce_max_xor(float x) {
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 1), x);
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 2), x);
    return x;
}

__device__ __forceinline__ float warp_4lane_reduce_sum_xor(float x) {
    x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
    x += __shfl_xor_sync(0xFFFFFFFF, x, 2);
    return x;
}
```

### ~~Main Gap: Fused SiLU + Quantization~~ ✅ Already Implemented

**Correction:** FlashInfer already fuses activation with quantization for FP4/FP8:

```cpp
// FlashInfer csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
// Lines 2206-2218 in doActivationKernel:

auto post_act_val = gate_act * quant_scale;  // SiLU already applied

if constexpr (IsNVFP4 || IsMXFP8) {
  // Fused quantization in same kernel!
  auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize>(
      post_act_val, global_scale_val, num_tokens_before_expert, expert, token, elem_index,
      inter_size, fc2_act_sf_flat,
      IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
              : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
  output_vec[elem_index] = res;
}
```

The `doGatedActivationKernel` (BF16→BF16) is only used for non-quantized paths.
Our MXFP4 path uses `doActivationKernel` which already has fused activation + quantization.

---

## Limitations of HPC-Ops for Our Use

| Limitation | Impact |
|------------|--------|
| **SM90 Only** | Kernels won't run on SM121 without porting |
| **No MXFP4** | They use FP8, not FP4 weights |
| **No Block Scaling** | Their FP8 uses per-tensor/per-row, not block scaling |
| **No Attention Sinks** | GPT-OSS-120B requires attention sinks |
| **Closed MoE** | Their MoE assumes specific activation (SiLU) |

---

## References

- [hpc-ops GitHub](https://github.com/Tencent/hpc-ops)
- Source files analyzed:
  - `src/attention/decode/smallm_kernels.cuh`
  - `src/attention/decode/m64_kernels.cuh`
  - `src/group_gemm/kernels.cuh`
  - `src/fuse_moe/fuse_moe.cu`
  - `src/utils/utils.cuh`
