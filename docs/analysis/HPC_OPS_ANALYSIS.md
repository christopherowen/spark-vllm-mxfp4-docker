# Tencent HPC-Ops Analysis

Deep dive analysis of [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops) for techniques applicable to our SM121 MXFP4 optimization work.

## Executive Summary

HPC-Ops is a production-grade CUDA kernel library from Tencent's Hunyuan AI Infra team, achieving impressive speedups on H20 (SM90):
- **Attention Decode**: 2.22x over FlashInfer/FA2/FA3
- **FusedMoE (FP8)**: 1.49x prefill, 1.14x decode
- **GroupGEMM (FP8)**: 1.88x decode over DeepGEMM

While their kernels are SM90-specific, the **techniques are directly applicable** to our SM121 work.

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

// Scale in log2 space
row_max = warp_reduce(row_max) * one_over_dk_log2e;

// exp2 is faster than exp (single PTX instruction)
tAttr(im, in) = exp2f_ftz(tAttr(im, in) * one_over_dk_log2e - gMax(in));

// Rescale running sum
float scale = exp2f_ftz(last_max - gMax(in));
gSum(in) = gSum(in) * scale + row_sum;
```

**Key Insight:** 
- `exp2f_ftz` maps to a single PTX instruction (`ex2.approx.ftz.f32`)
- Working in log2 space avoids expensive division by log(e)
- `_ftz` (flush-to-zero) versions are faster than standard math

**Fast Math Utilities (utils.cuh):**
```cpp
__device__ __forceinline__ float exp2f_ftz(float x) {
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float silu(float x) {
    return x * rcpf_ftz(1.f + expf_ftz(-x));
}
```

**Applicability to SM121:**
Our FlashInfer attention kernels likely use standard exp/log. Switching to log2 space with `exp2f_ftz` could provide measurable speedup.

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

## Comparison: HPC-Ops vs Our Approach

| Aspect | HPC-Ops (SM90) | Our Current (SM121) |
|--------|----------------|---------------------|
| **Attention** | Custom CUTE/CUTLASS | FlashInfer FA2 |
| **MoE** | Fused 5-kernel pipeline | CUTLASS grouped GEMM |
| **Softmax** | log2 space + exp2f_ftz | Standard exp |
| **Load/Compute Overlap** | Explicit warp specialization | CUTLASS implicit |
| **TMA** | Dynamic descriptor updates | Fixed descriptors |
| **Precision** | FP8 E4M3 | FP8×FP4 (MXFP4) |
| **Activation Fusion** | SiLU + quant fused | Separate kernels |

---

## Recommended Actions

### High Priority (Direct Performance Impact)

1. **Adopt log2 Softmax Space**
   - Replace `exp(x)` with `exp2f_ftz(x * log2e)` in attention
   - Expected: 5-10% attention speedup

2. **Fuse Activation + Quantization**
   - Combine SiLU activation with FP8 quantization
   - Avoid memory round-trip between MoE GEMMs
   - Expected: 5-15% MoE decode speedup

3. **Add Fast Math Utilities**
   - Create `utils.cuh` with `exp2f_ftz`, `rcpf_ftz`, `silu_fast`
   - Use throughout FlashInfer kernels

### Medium Priority (Architecture Improvements)

4. **Implement Warp Specialization for Decode**
   - Separate load and compute warpgroups
   - Explicit register allocation tuning

5. **Small-M Attention Kernel**
   - Specialized kernel for M=1 decode
   - 4-lane reductions instead of full warp

6. **Multi-Stage Pipelining for Attention**
   - Overlap KV cache loads with attention compute
   - Double-buffer KV in shared memory

### Lower Priority (Future Optimization)

7. **Dynamic TMA for MoE**
   - Per-expert TMA descriptor updates
   - Single kernel launch for all experts

8. **Multimem Operations for TP**
   - Explore `multimem.ld_reduce` for all-reduce
   - May help with TP=2 NCCL overhead

---

## Code Snippets to Port

### Fast Math Header (Recommended for FlashInfer)

```cpp
// flashinfer/include/flashinfer/fast_math.cuh

#pragma once

namespace flashinfer {
namespace fast_math {

__device__ __forceinline__ float exp2f_ftz(float x) {
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float log2f_ftz(float x) {
    float r;
    asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rsqrtf_ftz(float x) {
    float r;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float expf_ftz(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    return exp2f_ftz(x * LOG2E);
}

__device__ __forceinline__ float silu_fast(float x) {
    return x * rcpf_ftz(1.f + expf_ftz(-x));
}

} // namespace fast_math
} // namespace flashinfer
```

### Warp Reduction Utilities

```cpp
// Specialized reductions for small attention M
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
