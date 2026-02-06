# Plan: Add MXFP4 Support to CUTLASS Dense GEMM

## Goal

Unify dense layers (QKV, O, lm_head) and MoE to use the same CUTLASS backend, eliminating the Marlin/CUTLASS split.

## Current State

| Backend | NVFP4 (group 16) | MXFP4 (group 32) |
|---------|------------------|------------------|
| CUTLASS | ✓ Works | ✗ Blocked |
| cuDNN | ✓ Works | ✓ Works |
| Marlin | N/A | ✓ Works (used for dense) |

## Technical Difference

```cpp
// NVFP4 - currently used
struct nv_float4_t<float_e2m1_t> {
  using ScaleFactorType = cutlass::float_ue4m3_t;  // E4M3 (8-bit)
  using DataType = float_e2m1_t;                    // E2M1 (4-bit)
  // Block size: 16 elements per scale
};

// MXFP4 - need to add
struct mx_float4_t<float_e2m1_t> {
  using ScaleFactorType = cutlass::float_ue8m0_t;  // E8M0 (8-bit) 
  using DataType = float_e2m1_t;                    // E2M1 (4-bit) - SAME!
  // Block size: 32
};
```

Both formats use the same 4-bit E2M1 data type, just different scale factor formats and block sizes.

## Safe Implementation: Additive Changes Only

**Principle**: Add new code paths for MXFP4 without modifying existing NVFP4 paths.

### Step 1: Add MXFP4 Enum Value (additive)

**File:** `flashinfer/include/flashinfer/gemm/fp4_gemm_cutlass.h`

```cpp
enum class FP4GemmType {
  W4A4_NVFP4_NVFP4,      // Existing - unchanged
  W4A4_MXFP4_MXFP4,      // NEW - added
};
```

### Step 2: Add MXFP4 Kernel Macro (additive)

**File:** `flashinfer/include/flashinfer/gemm/fp4_gemm_template_sm120.h`

Add a NEW macro alongside the existing one:

```cpp
// EXISTING macro - UNCHANGED
#define INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(T, CTA_M_, CTA_N_, CTA_K_, ...) \
  struct DeviceGemmFp4GemmSm120_... { \
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  \
    // ... existing code unchanged ...
  };

// NEW macro - ADDED
#define INSTANTIATE_MXFP4_GEMM_KERNEL_LAUNCHER(T, CTA_M_, CTA_N_, CTA_K_, ...) \
  struct DeviceGemmMxfp4GemmSm120_... { \
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  \
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  \
    // ... rest same structure ...
  };
```

### Step 3: Add MXFP4 Dispatch Branch (additive)

**File:** `flashinfer/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`

```cpp
template <typename T, FP4GemmType fp4GemmType>
size_t dispatchGemmConfigSm120(...) {
  if constexpr (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4) {
    // EXISTING path - UNCHANGED
    return dispatchNVFP4xNVFP4GemmCTAShapeSm120<T>(...);
  } else if constexpr (fp4GemmType == FP4GemmType::W4A4_MXFP4_MXFP4) {
    // NEW path - ADDED
    return dispatchMXFP4xMXFP4GemmCTAShapeSm120<T>(...);
  } else {
    throw std::runtime_error("Unsupported FP4 Gemm type");
  }
}
```

### Step 4: Add MXFP4 C++ Binding (additive)

**File:** `flashinfer/csrc/fp4_gemm_cutlass_sm120.cu`

```cpp
// EXISTING - UNCHANGED
void fp4_gemm(...) { /* NVFP4 path */ }

// NEW - ADDED
void mxfp4_gemm(...) {
  // Same logic but uses CutlassFp4GemmRunner<T, FP4GemmType::W4A4_MXFP4_MXFP4>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_gemm, torch_ext::fp4_gemm);        // existing
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp4_gemm, torch_ext::mxfp4_gemm);    // new
```

### Step 5: Add MXFP4 Python Runner (additive)

**File:** `flashinfer/flashinfer/gemm/gemm_base.py`

```python
# EXISTING requirement check - modify to allow MXFP4
@supported_compute_capability([100, 103, 110, 120, 121])
def _cutlass_gemm_fp4_requirement(..., use_nvfp4: bool = True):
    if use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    # Remove the use_nvfp4 check - CUTLASS now supports both
    return True

# EXISTING runner factory - modify to select correct kernel
def get_cutlass_fp4_gemm_module(major, use_nvfp4=True):
    if use_nvfp4:
        return get_gemm_sm120_module_cutlass_fp4()  # existing
    else:
        return get_gemm_sm120_module_cutlass_mxfp4()  # new
```

### Step 6: JIT Both Variants (additive)

**File:** `flashinfer/flashinfer/jit/gemm/core.py`

Add MXFP4 kernel generation alongside NVFP4:

```python
# EXISTING - unchanged
def gen_gemm_sm120_module_cutlass_fp4():
    # generates NVFP4 kernels

# NEW - added
def gen_gemm_sm120_module_cutlass_mxfp4():
    # generates MXFP4 kernels using INSTANTIATE_MXFP4_GEMM_KERNEL_LAUNCHER
```

## Risk Mitigation

| Change | Risk | Mitigation |
|--------|------|------------|
| Enum addition | None | Additive, doesn't change existing values |
| New macro | None | Parallel to existing, no modification |
| New dispatch branch | None | `else if` doesn't affect existing `if` |
| New binding | None | New function, existing unchanged |
| Python requirement | Low | Only removes a restriction |

## Validation Plan

1. **Existing tests pass** - Run NVFP4 tests to confirm no regression
2. **New MXFP4 tests** - Run with `use_nvfp4=False`
3. **Cross-validate** - Compare CUTLASS MXFP4 vs cuDNN MXFP4

## Benchmark Results (2026-01-28)

Benchmark script: `scripts/benchmarks/benchmark_dense_fp4.py`

### Decode (M=1) - Marlin Wins

| Shape | Marlin MXFP4 | cuDNN MXFP4 | CUTLASS NVFP4 |
|-------|--------------|-------------|---------------|
| QKV (6144×6144) | **37 μs** | 135 μs (0.27x) | 118 μs (0.31x) |
| lm_head (256000×6144) | **3543 μs** | 3954 μs (0.90x) | 3999 μs (0.89x) |

### Prefill (M≥128) - cuDNN/CUTLASS Win

| Shape | Marlin MXFP4 | cuDNN MXFP4 | CUTLASS NVFP4 |
|-------|--------------|-------------|---------------|
| QKV M=128 | 115 μs | **93 μs (1.23x)** | **89 μs (1.29x)** |
| QKV M=512 | 401 μs | **154 μs (2.60x)** | **177 μs (2.27x)** |
| lm_head M=128 | 7491 μs | **4508 μs (1.66x)** | **4539 μs (1.65x)** |

### Key Insight

- **Crossover point: M ≈ 64-128**
- For decode (M=1), Marlin's dequant+BF16 is faster due to lower overhead
- For prefill (M≥128), native FP4 tensor cores are 1.2-2.6x faster
- **Unified CUTLASS MXFP4 would win for prefill, lose for decode**

### Recommendation

1. **Short-term**: Keep Marlin for decode, consider CUTLASS for prefill
2. **Long-term**: Implement CUTLASS MXFP4, then add hybrid selection based on M

## Next Steps

1. [x] Verify CUTLASS builder supports `mx_float4_t` on SM120 ✓ (confirmed in sm1xx_common.inl)
2. [x] Create benchmark script ✓ (`scripts/benchmarks/benchmark_dense_fp4.py`)
3. [x] Benchmark Marlin vs cuDNN/CUTLASS ✓ (see results above)
4. [ ] Add `W4A4_MXFP4_MXFP4` to enum
5. [ ] Add MXFP4 kernel macro  
6. [ ] Add MXFP4 dispatch branch
7. [ ] Add C++ binding
8. [ ] Modify Python requirement
9. [ ] Add JIT for MXFP4
10. [ ] Run correctness tests
11. [ ] Implement hybrid M-based selection in vLLM
