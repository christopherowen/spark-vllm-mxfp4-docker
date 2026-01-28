# MXFP4 v2 Benchmark Results

Live tracking of benchmark results across configurations.

---

## 🏆 Latest Results: TP=2 with RDMA (2026-01-28)

**New Record!** Two-node tensor parallelism with proper RDMA/RoCE transport achieves **72 tok/s decode** - 20% faster than single-node!

| Configuration | Prefill (t/s) | Decode tg32 (t/s) | Decode tg128 (t/s) |
|---------------|---------------|-------------------|---------------------|
| **TP=2 RDMA** | **6,329** | **72.71** | **71.43** |
| TP=1 (single node) | 4,573 | 60.02 | 60.07 |

### Key Fix: `/dev/infiniband` Device Mount

The critical fix was exposing RDMA devices to the Docker container:

```yaml
# docker-compose.dev.yml
devices:
  - /dev/infiniband:/dev/infiniband
cap_add:
  - IPC_LOCK
```

### NCCL Transport Verification

```
# BEFORE (falling back to TCP sockets - slow):
NCCL INFO NET/Socket : Using [0]enp1s0f1np1...

# AFTER (using RDMA/RoCE - fast):
NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE...
```

### Environment Variables

```bash
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1
```

### Coherency Validation

All 16 coherency tests passed across short (2K), medium (16K), and long (80K) contexts:
- Geography: 4/4 ✓
- Math: 4/4 ✓
- Science: 4/4 ✓  
- Code: 4/4 ✓

---

## TP=1 Results (2026-01-17)

Single-node baseline for comparison.

| Context | Prefill (t/s) | Decode tg32 (t/s) | Decode tg128 (t/s) |
|---------|---------------|-------------------|---------------------|
| **Short (512)** | 1,854 | **60.02** | **60.07** |
| **Medium (2048)** | 4,573 | **59.36** | **59.47** |
| **Long (8192)** | 6,628 | **57.52** | **57.81** |

### vs Competition

| Engine | Decode (t/s) | Status |
|--------|--------------|--------|
| SGLang | 52 | ✅ Beat by 10-15% |
| llama.cpp | 58 | ✅ Beat at short/medium context |
| **vLLM TP=1** | **57-60** | **Winner** |
| **vLLM TP=2 RDMA** | **72** | **New Champion** |

### Key Optimizations

1. ✅ **64×128 tile shapes** for decode (small-M optimization)
2. ✅ **CUTLASS FP8×FP4 MoE GEMM** on SM121
3. ✅ **MXFP4 quantization** for MoE, QKV, O, and lm_head layers
4. ✅ **FlashInfer FA2 attention** with sink support
5. ✅ **RDMA/RoCE transport** for multi-node TP (2026-01-28)

---

## Historical Best Results

| Metric | Baseline | Best | Config | Date |
|--------|----------|------|--------|------|
| tg32 (tok/s) | 29.1 | **60.02** | **64×128 tiles + MXFP4 all** | 2026-01-17 |
| tg128 (tok/s) | 31.62 | **60.07** | **64×128 tiles + MXFP4 all** | 2026-01-17 |
| pp512 (tok/s) | 2209 | **1,854** | 64×128 tiles | 2026-01-17 |
| pp2048 (tok/s) | 4341 | **5699** | CUTLASS FP8×FP4 | 2026-01-11 |
| pp8192 (tok/s) | - | **6,628** | 64×128 tiles | 2026-01-17 |

**Milestones achieved**:
1. ✅ Native SM121 CUTLASS FP8×FP4 MoE GEMM - Prefill 31% faster
2. ✅ MXFP4 lm_head with Marlin kernel - Decode +9% (29 → 34 tok/s)
3. ✅ MXFP4 QKV/O quantization - Decode +32% (29.1 → 38.5 tok/s)
4. ✅ Combined MXFP4 all layers - Decode +68% (29.1 → 48.9 tok/s)
5. ✅ **64×128 tile optimization** - Decode +23% (48.9 → 60 tok/s)

---

## 64×128 Tile Optimization (2026-01-17)

### Problem

The default 128×128 CUTLASS tiles were inefficient for decode (M=1 to M=16).

### Solution

Implemented 64×128 tile shapes for small-M workloads:
- Modified `sm120_blockscaled_mma_builder.inl` to use `ceil_div` for scale factor block counts
- Modified `sm120_blockscaled_mma_array_tma.hpp` and `sm120_blockscaled_mma_tma.hpp` to pad TMA tile shapes
- Added tile selection heuristic: 64×128 for M≤16, 128×128 for larger batches

### Results

| Tile Shape | tg32 @ 16 tokens | tg32 @ 1 token |
|------------|------------------|----------------|
| 128×128 | 48.9 tok/s | 60.0 tok/s |
| **64×128** | **59.4 tok/s** | 48.7 tok/s |

64×128 is **19% faster** for typical decode workloads (16 tokens).

See: `docs/plans/SM120_MOE_TILE_EXPANSION.md` for full implementation details.

---

## MXFP4 QKV/O Quantization Results (2026-01-12)

### Configuration

```bash
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --mxfp4-layers moe,qkv,o \
    --load-format fastsafetensors
```

### Benchmark Comparison

| Config | pp2048 (tok/s) | tg32 (tok/s) | Decode Δ |
|--------|----------------|--------------|----------|
| MoE-only | 4472 | 29.1 | baseline |
| MoE + QKV/O | 4470 | 38.5 | +32% |
| **All (MoE+QKV/O+lm_head)** | **4640** | **48.9** | **+68%** |

### Analysis

- QKV/O MXFP4 uses Marlin kernel for fused dequant+GEMM
- Memory bandwidth savings from 4-bit weights (vs BF16)
- No prefill regression (compute-bound)
- Significant decode improvement (memory-bound)

### Next Steps

- Add `lm_head` to `--mxfp4-layers all` for further decode gains
- nsys profile to verify Marlin kernels are used for QKV/O

---

## Critical Finding: lm_head Bottleneck (2026-01-11)

### The Performance Gap Explained

| Engine | tg32 (tok/s) | lm_head format | lm_head size |
|--------|--------------|----------------|--------------|
| **llama.cpp** | **58** | MXFP4 | 0.29 GB |
| **vLLM** | **29** | BF16 | 1.16 GB |
| Difference | **-50%** | **4x larger** | **+3.1 ms/tok** |

### Why vLLM Uses BF16 for lm_head

1. **Model config excludes it**: `quantization_config.modules_to_not_convert: ["lm_head", ...]`
2. **MXFP4 LinearMethod not implemented**: Falls back to `UnquantizedLinearMethod`
3. **Native checkpoint is BF16**: safetensors has `lm_head.weight: dtype=bfloat16`

### Optimization Path

To match llama.cpp decode performance:
1. **Implement MXFP4 for lm_head** - Expected +30-45% decode speedup
2. Use DP4A-based FP4 GEMV for memory-bound decode

See detailed analysis: `docs/analysis/LLAMA_CPP_ANALYSIS.md`

---

## Upstream Baseline (vLLM main, No PR changes)

**Status**: ✅ COMPLETE - 2026-01-09

This is the TRUE baseline - upstream vLLM main without any PR #31740 changes.

### Environment
```yaml
date: 2026-01-09 16:25:01
git_sha_vllm: ac9f9330e (upstream/main)
git_sha_flashinfer: bd2b033f (upstream/main)
cuda_version: (NGC 25.12)
gpu: NVIDIA GB10 (SM121)
```

### vLLM Configuration
```yaml
quantization: mxfp4
tensor_parallel_size: 1
max_model_len: 131072
max_num_seqs: 2
max_num_batched_tokens: 8192
enforce_eager: true
enable_prefix_caching: true
load_format: fastsafetensors
served_model: openai/gpt-oss-120b

env_vars:
  VLLM_ATTENTION_BACKEND: (unset - auto)
```

### Observed Runtime Configuration
```yaml
# From vLLM logs:
attention_backend: TRITON_ATTN  # NOT FlashInfer!
moe_kernel: Marlin
fp4_mode: "Weight-only FP4 compression" (not native SM121)
```

### Workload Parameters
```yaml
prompt_length: 2048
output_lengths: [32]
batch_size: 1
concurrency: 1
test_requests: 10
tool: llama-benchy 0.1.1
latency_mode: generation
```

### Results

#### Prefill Throughput (varying context length)
```yaml
pp512_tps: 2209.10 ± 97.25   # TTFT: 332ms
pp1024_tps: 3385.55 ± 81.00  # TTFT: 418ms  
pp2048_tps: 4340.70 ± 12.22  # TTFT: 555ms
pp4096_tps: 4008.20 ± 23.05  # TTFT: 1051ms
```

#### Decode Throughput (pp=2048, varying output length)
```yaml
tg16_tps: 31.53 ± 0.06
tg32_tps: 31.58 ± 0.06
tg64_tps: 31.62 ± 0.03
tg128_tps: 31.62 ± 0.05
tg256_tps: 31.12 ± 0.35
```

#### Latency (pp=2048, tg=32)
```yaml
ttfr_ms: 457.13 ± 13.94
est_ppt_ms: 417.48 ± 13.94
e2e_ttft_ms: 554.80 ± 14.04
```

#### Variance Analysis
| Metric | Mean | Std Dev | CV% | Notes |
|--------|------|---------|-----|-------|
| pp512 | 2209 | ±97 | 4.4% | Lower prefill shows higher relative variance |
| pp1024 | 3386 | ±81 | 2.4% | |
| pp2048 | 4341 | ±12 | 0.3% | Suspiciously low - may be warmed state |
| pp4096 | 4008 | ±23 | 0.6% | |
| tg32 | 31.58 | ±0.06 | 0.2% | **Very stable** - decode is memory-bound |
| tg128 | 31.62 | ±0.05 | 0.2% | |
| tg256 | 31.12 | ±0.35 | 1.1% | Slight degradation at longer context |

**Key insight**: Decode throughput (tg*) is highly stable (~0.2% CV). Prefill throughput (pp*) can vary 2-7% between runs due to prefix caching and warmup effects.

### Key Findings

1. **Attention**: Using `TRITON_ATTN`, NOT FlashInfer
2. **MoE**: Using Marlin kernel
3. **FP4**: Weight-only compression, NOT native SM121 FP4 compute
4. **Warning**: "Your GPU does not have native support for FP4 computation"

This confirms PR #31740 is needed to enable:
- FlashInfer attention backend
- CUTLASS MoE kernels
- Native SM121 FP4 path

---

## nsys Kernel Profiling (Marlin + FlashInfer FA2)

**Status**: ✅ COMPLETE - 2026-01-10

Full nsys profiling to identify bottlenecks. See detailed analysis in `docs/analysis/VLLM_BASELINE_ANALYSIS.md`.

### Configuration
```yaml
moe_kernel: marlin
attention_backend: FLASHINFER (native CUTLASS FA2 on SM121)
kv_cache_layout: HND
prompt_tokens: 2048
output_tokens: 64
enforce_eager: true
```

### Key Findings

**Attention is NOT the bottleneck.** FlashInfer attention accounts for only **1.5%** of GPU time.

| Category | GPU Time % | Primary Kernels |
|----------|------------|-----------------|
| **Dense GEMV/GEMM** | **38.4%** | `gemvx::kernel`, `cutlass_80_tensorop` |
| **MoE GEMM (Marlin)** | **33.6%** | `marlin_moe_wna16::Marlin` |
| **Memory/Data Movement** | **12.4%** | `elementwise_kernel`, `CatArrayBatchedCopy` |
| **Activations/Norms** | **3.1%** | `swigluoai`, `fused_add_rms_norm` |
| **MoE Overhead** | **2.7%** | `marlin_repack`, `topkGatingSoftmax` |
| **Attention** | **1.5%** | `BatchPrefillWithPagedKVCacheKernel` |
| **Other** | **8.3%** | Various PyTorch ops |

### Performance
```yaml
decode_throughput: 30.3 tok/s
tokens_generated: 64
runs: 3
warmup: 2
```

### Optimization Priority (Based on Profile)

1. **MoE GEMM** (34%): Test CUTLASS grouped GEMM vs Marlin
2. **Dense GEMV** (38%): Consider quantized lm_head
3. **Memory** (12%): MXFP8 persistent activations
4. ~~Attention~~ (1.5%): **LOW PRIORITY** - already efficient

### Artifacts
- Profile: `docs/perf_artifacts/marlin_flashinfer_profile.nsys-rep`
- Database: `docs/perf_artifacts/marlin_flashinfer_profile.sqlite`

---

## PR #31740 with Baseline Config (Marlin + TRITON_ATTN)

**Status**: ✅ COMPLETE - 2026-01-09

Running PR #31740 with forced baseline configuration to verify parity.

### Environment
```yaml
date: 2026-01-09 17:27:17
git_sha_vllm: 77bf5a554 (pr-31740)
git_sha_flashinfer: bd2b033f (upstream/main -> mxfp4_v2)
```

### Configuration Override
```bash
# Force baseline configuration on PR branch
export VLLM_MXFP4_MOE_KERNEL=marlin
vllm serve ... --attention-backend TRITON_ATTN
```

### Results
```yaml
throughput:
  pp2048_tps: 4701.82 ± 362.71
  tg32_tps: 31.87 ± 0.05
```

### Comparison to Upstream Baseline
| Metric | Upstream | PR+Baseline (3 runs) | Notes |
|--------|----------|---------------------|-------|
| pp2048 | 4341 ± 12 | 4200-4500 ± 300 | High variance due to prefix caching, warmup |
| tg32 | 31.63 ± 0.04 | 31.82-31.89 ± 0.05 | **Stable, matches baseline** ✓ |

**Conclusion**: Decode (tg32) performance matches. Prefill (pp2048) has high run-to-run variance but is in the same range. The decode stability is what matters for our optimization goal.

---

## PR #31740 with Native Config (FlashInfer + CUTLASS)

**Status**: PENDING - Requires FlashInfer SM12x support

### Expected Configuration
```yaml
attention_backend: FLASHINFER  # Native FA2
moe_kernel: CUTLASS_BLACKWELL_FP4FP8  # Native CUTLASS FP8×FP4
```

### Blocker
FlashInfer `upstream/main` lacks SM12x JIT compilation support:
```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

Need FlashInfer `mxfp4_wip` branch with SM12x kernels.

---

## Eagle3 Speculative Decoding (Baseline Config)

**Status**: ✅ COMPLETE - 2026-01-09

Testing Eagle3 with baseline backends (Marlin + TRITON_ATTN) to establish Eagle3 baseline.

### Configuration
```yaml
attention_backend: TRITON_ATTN
moe_kernel: marlin (via VLLM_MXFP4_MOE_KERNEL=marlin)
speculative_config:
  method: eagle3
  num_speculative_tokens: 3
```

### Results

| Eagle3 Model | tg32 (tok/s) | pp2048 (tok/s) | Acceptance Rate |
|--------------|--------------|----------------|-----------------|
| **short-context** | 13.97 ± 0.08 | 5153 ± 455 | ~31% |
| **long-context** | 14.42 ± 0.09 | 5193 ± 465 | ~31% |
| **throughput** | 14.71 ± 0.07 | 5241 ± 395 | ~31% |
| *Baseline (no spec)* | *31.6* | *4341* | *N/A* |

### Analysis

**Eagle3 with baseline config is 2x SLOWER than no speculation!**

| Metric | No Spec | With Eagle3 | Delta |
|--------|---------|-------------|-------|
| tg32 | 31.6 tok/s | 14.4 tok/s | **-54%** |

**Root Cause**: ~31% draft acceptance rate (should be 60-70%+)
- Per-position acceptance: 0.55, 0.27, 0.13 (very low)
- Drafted tokens are being rejected, wasting compute

**Hypothesis**: Eagle3 was trained with different numerical characteristics than Marlin + TRITON_ATTN produces. The FlashInfer + CUTLASS backends (used in mxfp4_wip where 61 tok/s was achieved) likely have outputs that match better.

### Next Steps
Port FlashInfer sink support from mxfp4_wip to enable FlashInfer attention, then re-test Eagle3.

---

## Attention Backend Testing (Sinks Disabled)

**Status**: ✅ COMPLETE - 2026-01-09

Added `VLLM_ATTENTION_SINKS` env var to control attention sink behavior for testing.

### Configuration
```yaml
# Marlin + FLASH_ATTN (sinks disabled)
env: VLLM_ATTENTION_SINKS=false, VLLM_MXFP4_MOE_KERNEL=marlin
attention_config: {"backend": "FLASH_ATTN"}
```

### Results

| Backend | tg32 (tok/s) | Output Quality | Notes |
|---------|--------------|----------------|-------|
| **TRITON_ATTN** (env=false) | **31.5** | ✅ Correct | ⚠️ Sinks NOT actually disabled |
| **FLASH_ATTN** (sinks disabled) | **31.9** | ❌ Garbage | Bug in FA2 path on SM121 |
| **FLASHINFER** | **31.7** | ✅ Correct | Uses native FA2 sink module on SM121 |

### Analysis

**IMPORTANT CORRECTION**: TRITON_ATTN does NOT check `VLLM_ATTENTION_SINKS` env var!
When we tested "TRITON_ATTN with sinks disabled", sinks were **actually still enabled**.

- **TRITON_ATTN**: Has native sink support, ignores env var - sinks always active
- **FLASH_ATTN**: Respects env var, but produces garbage without sinks on SM121

The raw FA2 kernel is actually 2x faster than TRITON's unified_attention in microbenchmarks, but something in vLLM's FLASH_ATTN backend path corrupts the output.

### Root Cause: Multiple Issues with FLASH_ATTN on SM121

#### Issue 1: Non-standard Head Dimension

gpt-oss-120b uses **head_dim=45** which is NOT in FA2's supported list:
- FA2 supports: hdim32, 64, 96, 128, 160, 192, 256
- **hdim45 is NOT supported!**

Even with native SM121 FA2 kernels, the model's non-standard head dimension causes garbage output.
TRITON_ATTN is more flexible and handles arbitrary head dims correctly.

#### Issue 2: Dao-AILab flash-attention Lacks SM121 Support

**This is a known issue**: [flash-attention#1969 - Support for compute capability 12.1 (sm121) - NVIDIA GB10 GPU](https://github.com/Dao-AILab/flash-attention/issues/1969)

The vLLM FLASH_ATTN backend uses Dao-AILab's flash-attention library, which:
- **Does NOT officially support SM121 (GB10 / DGX Spark)**
- Only compiles kernels for SM80 (see `cuobjdump` output below)
- Relies on PTX JIT compilation from SM80 → SM121 at runtime
- PTX forward compatibility fails silently, producing numerically incorrect results

```bash
# Proof: vLLM FA2 binary only contains SM80 cubins
$ cuobjdump -lelf /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
ELF file    1: _vllm_fa2_C.abi3.1.sm_80.cubin
ELF file    2: _vllm_fa2_C.abi3.2.sm_80.cubin
... (all sm_80, no sm_120 or sm_121)
```

**Why SM121 can't be easily added**:
The vLLM flash-attn source has **88 CUDA files hardcoded for SM80** (`*_sm80.cu`). Adding SM121 to CMakeLists only produces PTX from SM80 source, which JIT compiles incorrectly at runtime.

```bash
$ ls /workspace/vllm/.deps/vllm-flash-attn-src/csrc/flash_attn/src/*.cu | grep -oE 'sm[0-9]+' | sort -u
sm80  # Only SM80 kernels exist!
```

**Workarounds**:
1. Use **TRITON_ATTN** (works correctly on SM121)
2. Use **FlashInfer backend** (has native SM121a JIT compilation with Blackwell-specific fixes)
3. Wait for Dao-AILab to add SM121 support (open issue since Oct 2025)

### Conclusion
For SM121 with gpt-oss-120b (head_dim=45):
- Use **TRITON_ATTN** for correct output (handles non-standard head dims)
- **FLASH_ATTN fails** due to unsupported head_dim=45, even with native SM121 build
- **FlashInfer** (from mxfp4_wip with native FA2 sink support) is the path to 61+ tok/s

**Note**: We built FA2 with native SM121 support (see `docs/FLASH_ATTN_SM121_BUILD.md`) and verified the kernel runs without NaN/Inf, but gpt-oss-120b's head_dim=45 is outside FA2's supported dimensions.

---

## FlashInfer Native Attention with Sinks (SM121)

**Status**: ✅ WORKING - 2026-01-10

Successfully ported FlashInfer native FA2 attention with sink support from mxfp4_wip branch.

### Configuration
```yaml
# Marlin + FlashInfer + Sinks
env: VLLM_MXFP4_MOE_KERNEL=marlin
attention_config: {"backend": "FLASHINFER"}
# VLLM_ATTENTION_SINKS defaults to "auto" (sinks enabled)
```

### Results

| Config | tg32 (tok/s) | Output Quality | Notes |
|--------|--------------|----------------|-------|
| **Marlin + FlashInfer + Sinks** | **32.6** | ✅ Coherent | FA2 sink module working |
| **Marlin + FlashInfer (no sinks)** | **32.6** | ❌ Garbage | Model requires sinks |
| **Marlin + TRITON_ATTN + Sinks** | **32.3** | ✅ Coherent | Reference baseline |

### Key Fixes Applied

1. **SM12x TRTLLM exclusion** (`vllm/utils/flashinfer.py`):
   - TRTLLM cubins only work on SM100/SM103, not SM12x
   - Added capability check to prevent auto-detection on SM121

2. **o_data_type parameter bug** (`vllm/v1/attention/backends/flashinfer.py`):
   - `fast_plan_decode` was missing `o_data_type` param in positional args
   - Caused argument shift, breaking `seq_lens` interpretation
   - **Pre-existing bug in vLLM main** - needs upstream PR

3. **non_blocking=None fix** (`flashinfer/decode.py`):
   - PyTorch doesn't accept `None` for `non_blocking` parameter
   - Added fallback: `nb = non_blocking if non_blocking is not None else True`

4. **Sinks passed to run()** (`flashinfer.py`):
   - `_use_fa2_sinks` was on builder, not impl
   - Changed to simply pass `sinks=self.sinks` (None if not available)

### Analysis

FlashInfer native attention achieves **same performance as TRITON_ATTN** (~32 tok/s) with correct output when sinks are enabled.

**Key finding**: gpt-oss-120b **requires sinks** for coherent output:
- FlashInfer without sinks → garbage
- TRITON_ATTN always uses sinks (ignores `VLLM_ATTENTION_SINKS` env var)
- All coherent results we measured were with sinks enabled

### Next Steps
Test Eagle3 with FlashInfer attention to verify speculative decoding still works.

---

## Comparison Matrix

| Config | tg32 | tg128 | tg256 | pp2048 | TTFT p50 | Notes |
|--------|------|-------|-------|--------|----------|-------|
| **Upstream Baseline** | **31.6** | **31.6** | **31.1** | **4341** | ~555ms | TRITON_ATTN, Marlin, no native FP4 |
| **PR #31740 + Baseline** | **31.9** | - | - | **4702** | ~579ms | Same backends, parity confirmed |
| **Marlin + FlashInfer + Sinks** | **32.6** | - | - | - | - | ✅ Coherent, FA2 sink module |
| **CUTLASS FP8×FP4 + FlashInfer** | **29.0** | **28.8** | - | **5699** | **~532ms** | 🎉 Native SM121! |
| **Eagle3 short-context** | **14.0** | - | - | 5153 | ~662ms | 31% acceptance rate - REGRESSION |
| **Eagle3 long-context** | **14.4** | - | - | 5193 | ~577ms | 31% acceptance rate - REGRESSION |
| **Eagle3 throughput** | **14.7** | - | - | 5241 | ~655ms | 31% acceptance rate - REGRESSION |
| Eagle3 + FlashInfer | - | - | - | - | - | TODO |
| CUTLASS + Eagle3 | - | - | - | - | - | TODO |
| CUTLASS + MXFP4 lm_head | - | - | - | - | - | **HIGH PRIORITY** |
| | | | | | | |
| **llama.cpp** | **57.85** | - | - | 2449 | - | Target (MXFP4 lm_head) |
| **SGLang** | **52.37** | - | - | - | 49.87ms | Target |

### Key Observations

1. **Prefill**: CUTLASS FP8×FP4 is **31% faster** than Marlin (5699 vs 4341 tok/s)
2. **Decode**: CUTLASS slightly slower than Marlin (29 vs 31.6 tok/s) due to FP8 quantization overhead
3. **Gap to targets**: 29 tok/s vs 52-58 tok/s - **BF16 lm_head is the bottleneck**
4. **lm_head optimization** is highest priority for decode improvement

---

## Reference: Competing Engines

### llama.cpp (llama-bench, build f5acfb2ff)

| Test | Throughput (t/s) |
|------|------------------|
| pp2048 | 2449.83 ± 10.27 |
| tg32 | **57.85 ± 0.44** |
| pp2048 @ d4096 | 2293.59 ± 8.99 |
| tg32 @ d4096 | 54.81 ± 0.30 |

### SGLang

| Metric | Value |
|--------|-------|
| Output throughput | **52.37 tok/s** |
| TTFT | 49.87 ms |
| TPOT | 18.83 ms |

---

## Detailed Run Logs

Each benchmark run should link to:
1. Full metadata file in `docs/TEST_LOGS/`
2. nsys profile (if captured) in `docs/perf_artifacts/`
3. Feature porting doc in `docs/plans/`

| Run ID | Date | Config | tg32 | Notes |
|--------|------|--------|------|-------|
| baseline_upstream_v1 | 2026-01-09 | TRITON_ATTN + Marlin | 31.6 | nsys profile captured |

---

## Configuration Reference

Environment variables and CLI arguments for consistent benchmarking.

### vLLM Runtime Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VLLM_MXFP4_MOE_KERNEL` | `auto`, `marlin`, `gemm`, `gemv`, `triton` | `auto` | MoE kernel selection. `marlin` for baseline, `gemm` for CUTLASS. |
| `VLLM_ATTENTION_SINKS` | `auto`, `true`, `false` | `auto` | Attention sink control. `false` disables sinks for testing. |

### Attention Backend Selection

Use `--attention-config` CLI argument (not env var):

```bash
# TRITON_ATTN (baseline fallback)
vllm serve ... --attention-config '{"backend": "TRITON_ATTN"}'

# FLASH_ATTN (FlashAttention 2)
vllm serve ... --attention-config '{"backend": "FLASH_ATTN"}'

# FLASHINFER (requires SM12x kernel support)
vllm serve ... --attention-config '{"backend": "FLASHINFER"}'
```

### FlashInfer Build Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASHINFER_CUDA_ARCH_LIST` | `12.1f` | Target SM121 with FP4 hardware path (family mode) |
| `FLASHINFER_NVCC_THREADS` | `4` | Parallel JIT compilation threads |
| `FLASHINFER_LOGLEVEL` | `0-5` | Logging verbosity (0=quiet) |
| `FLASHINFER_JIT_VERBOSE` | `0/1` | JIT compilation logging |

### Baseline Configuration

For reproducible baseline benchmarks, use:

```bash
# Environment
export VLLM_MXFP4_MOE_KERNEL=marlin
unset VLLM_ATTENTION_BACKEND  # Don't use deprecated env var

# Server command
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --enforce-eager \
    --load-format fastsafetensors \
    --attention-config '{"backend": "TRITON_ATTN"}'
```

### Testing Different Configurations

```bash
# Test with CUTLASS MoE (requires FlashInfer SM12x support)
export VLLM_MXFP4_MOE_KERNEL=gemm

# Test with sinks disabled (for Eagle3 or FlashInfer testing)
export VLLM_ATTENTION_SINKS=false

# Test with FlashInfer attention
vllm serve ... --attention-config '{"backend": "FLASHINFER"}'
```

### Legacy Variables (Deprecated)

These are no longer set by default in Docker configs:

| Variable | Notes |
|----------|-------|
| `VLLM_ATTENTION_BACKEND` | Use `--attention-config` CLI instead |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | Set in docker-compose.yml for production |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | Set in docker-compose.yml for production |
| `VLLM_FLASHINFER_MOE_BACKEND` | Set in docker-compose.yml for production |
| `VLLM_USE_CUDA_GRAPH` | Set in docker-compose.yml for production |

---

## How to Run Benchmarks

```bash
# Start vLLM server
docker compose -f docker-compose.dev.yml --profile serve up serve

# In another terminal, run llama-benchy
llama-benchy \
  --model gpt-oss-120b \
  --base-url http://localhost:8000 \
  --num-requests 10 \

# Collect metadata
scripts/collect_benchmark_metadata.sh > docs/TEST_LOGS/run_$(date +%Y%m%d_%H%M%S).yaml
```

---

## FlashInfer FA2 Sink Module Investigation (2026-01-10)

### Root Cause Analysis

Systematic validation of 6 potential issues with the FlashInfer FA2 attention sink implementation:

| Issue | Status | Finding |
|-------|--------|---------|
| **#1: Backend mismatch (prefill vs decode)** | **CRITICAL BUG** | Prefill has `use_sinks` parameter, decode has NONE |
| **#2: Argument ordering mismatch** | OK | Sink module expects `[sinks, sm_scale]`, code passes exactly that |
| **#3: Cache key missing compile-time knobs** | **BUG** | `use_logits_soft_cap=False`, `use_fp16_qk_reduction=False` hardcoded |
| **#4: Sink value space (log vs linear)** | OK | Model stores natural log, kernel converts with `* log2e` |
| **#5: LSE renormalization** | OK | Sinks applied once (first tile), properly incorporated into running sum |
| **#6: Shape/layout mismatch (QO vs KV heads)** | OK | Model: `[64]` (QO heads), kernel: `sink[qo_head_idx]` - matches |

### Critical Bugs Found

1. **Decode path has NO sink support**
   - Location: `flashinfer/decode.py` - no `use_sinks` parameter anywhere
   - Impact: Prefill uses sinks, decode doesn't → model outputs garbage after first token
   
2. **Sink module cache key missing compile-time knobs**
   - Location: `flashinfer/jit/attention/modules.py:1122-1124`
   - Hardcoded: `use_logits_soft_cap=False`, `use_fp16_qk_reduction=False`
   - Impact: If model uses logits soft cap, wrong kernel is compiled/used

### Model Sink Values (gpt-oss-120b)

```
Shape: [64] (one per QO head)
Dtype: bfloat16
Range: -2.06 to 5.75 (natural log space)
Example: [1.56, 1.08, 1.25, 1.24, 1.05]
```

### Required Fixes

1. Add `use_sinks` parameter to decode path (`BatchDecodeWithPagedKVCacheWrapper.plan()`)
2. Add `use_logits_soft_cap` and `use_fp16_qk_reduction` to sink module generation
3. Wire decode sinks through vLLM's FlashInfer backend

---

## FlashInfer FA2 Sink Implementation Results (2026-01-10)

**Status**: ✅ COMPLETE - Implementation working with sinks

After implementing all three fixes above, FlashInfer FA2 with sinks is now functional.

### Configuration

```yaml
moe_kernel: Marlin (VLLM_MXFP4_MOE_KERNEL=marlin)
attention_backend: FLASHINFER (--attention-config '{"backend": "FLASHINFER"}')
sinks: enabled (default, no VLLM_ATTENTION_SINKS override)
```

### Results

| Configuration | Coherent | tg=32 (tok/s) | tg=128 (tok/s) |
|---------------|----------|---------------|----------------|
| Marlin + FA2 (no sinks) | ❌ GARBAGE | N/A | N/A |
| Marlin + FA2 + Sinks | ✅ | **31.9** | **32.3** |

### Comparison to Baseline

| Config | tg=32 (tok/s) | Notes |
|--------|---------------|-------|
| Baseline (Marlin + TRITON_ATTN) | 31.6 | From upstream baseline |
| Marlin + FlashInfer FA2 + Sinks | 31.9 | **Matches baseline** |

### Key Findings

1. **gpt-oss-120b requires sinks** - Without sinks, output is garbage regardless of attention backend
2. **FlashInfer FA2 sink module works** - Performance matches TRITON_ATTN baseline (~32 tok/s)
3. **No regression** - New sink implementation does not degrade performance

### Test Output Sample

```
Prompt: "The capital of France is"
Without sinks: " a is the is a is the is a is the is a is the is a is the is"
With sinks: " Paris.\n\nGreat! If you have any more questions or need further assistance, feel free to ask"
```

---

## FlashInfer Speculative Decoding Support (2026-01-10)

### Problem

FlashInfer backend failed with Eagle3 speculative decoding due to:
```
AssertionError: FlashInfer backend currently only supports models in which 
all layers share the same values for: `window_left`, `logits_soft_cap`, `sm_scale`.
```

**Root cause**: Main model (gpt-oss-120b) and draft model (Eagle3) have different hyperparameters:
- Main model: `window_left=128`, `has_sinks=True`
- Draft model: `window_left=-1`, `has_sinks=False`

### Solution: Per-Hyperparameter-Group Wrappers

Implemented Option A: Create separate FlashInfer wrappers for each unique set of hyperparameters.

**Changes**:
1. `vllm/v1/attention/backends/utils.py`:
   - Added `group_layers_by_hyperparameters()` to group layers with identical hyperparameters

2. `vllm/v1/attention/backends/flashinfer.py`:
   - Added `HyperparamKey` type alias
   - Modified `FIPrefill` and `FIDecode` to store per-group wrappers
   - Modified `FlashInferMetadataBuilder` to create wrappers per group
   - Modified `FlashInferImpl.forward()` to look up correct wrapper by layer's hyperparameters

### Results

| Configuration | tg32 (tok/s) | tg64 (tok/s) | tg128 (tok/s) | Notes |
|--------------|--------------|--------------|---------------|-------|
| Baseline (no spec) | 32 | - | - | Marlin + TRITON_ATTN |
| FlashInfer (no spec, with sinks) | 31.9 | - | 32.3 | Matches baseline |
| **Eagle3-long + FlashInfer** | **19.1** | **20.5** | **22.1** | Speculative decoding working! |
| Eagle3-long + TRITON_ATTN | ~24-25 | - | - | For comparison |

### Key Findings

1. **FlashInfer + Eagle3 speculative decoding now works** - No more assertion failures
2. **Performance is similar to TRITON_ATTN** - Neither backend achieves the target 61 tok/s
3. **Both backends are slower than no speculation** - Indicates low acceptance rate issue, not backend-specific
4. **The 61 tok/s result from mxfp4_wip likely used different hyperparameters or model configuration**

### Next Steps

- Investigate why acceptance rate is low with both TRITON_ATTN and FlashInfer
- Compare numerical outputs between backends to identify any precision mismatches
- Test with BF16 (no quantization) to isolate MXFP4 as potential culprit

---

## Native CUTLASS MXFP4 MoE on SM121 🎉

**Status**: ✅ WORKING - 2026-01-11

Successfully implemented SM120/SM121 CUTLASS FP8×FP4 MoE GEMM kernel.

### Configuration

```yaml
date: 2026-01-11 18:40:37
moe_kernel: CUTLASS (FlashInfer FP8×FP4 for SM12x)
attention_backend: FLASHINFER
sinks: enabled (default)
quantization: mxfp4
enforce_eager: true
load_format: fastsafetensors
```

### Benchmark Results

```
llama-benchy 0.1.1
model: gpt-oss-120b @ http://localhost:8000/v1
latency_mode: generation
runs: 3
```

| Test | Throughput (tok/s) | TTFR (ms) | E2E TTFT (ms) |
|------|-------------------|-----------|---------------|
| **pp2048** | **5699.57 ± 38.47** | 427.37 ± 2.41 | 532.46 ± 2.85 |
| **tg32** | **29.00 ± 0.10** | - | - |
| **pp2048** | **5675.14 ± 36.37** | 428.91 ± 2.32 | 535.82 ± 1.72 |
| **tg128** | **28.78 ± 0.07** | - | - |

### Comparison to Baselines

| Config | pp2048 (tok/s) | tg32 (tok/s) | Notes |
|--------|---------------|--------------|-------|
| **CUTLASS FP8×FP4 + FlashInfer** | **5699** | **29.0** | ✅ Native SM121 |
| Marlin + FlashInfer | 4341 | 31.9 | Marlin dequant path |
| Upstream Baseline | 4341 | 31.6 | TRITON_ATTN + Marlin |
| **llama.cpp** | 2449 | **57.85** | Target |

### Key Findings

1. **Prefill is 31% faster** with CUTLASS (5699 vs 4341 tok/s)
2. **Decode is slightly slower** (29 vs 31.6 tok/s) - see analysis below
3. **6008 successful CUTLASS kernel runs** confirmed via debug logs
4. **CUTLASS kernel working correctly**: `run=Success` for all expert groups

### Decode Performance Gap Analysis

**Why 29 tok/s (CUTLASS) vs 31.6 tok/s (Marlin)?**

The CUTLASS FP8×FP4 kernel requires activation quantization overhead:
- BF16 → FP8 quantization per forward pass
- Scale factor computation per block (32 elements)

For decode (M=1), this overhead may outweigh the memory bandwidth savings.

**Why 29 tok/s (vLLM) vs 58 tok/s (llama.cpp)?**

**Critical finding: lm_head format difference** (see `docs/analysis/LLAMA_CPP_ANALYSIS.md`)

| | llama.cpp | vLLM |
|--|-----------|------|
| **lm_head format** | MXFP4 (FP4) | BF16 |
| **lm_head size** | 0.29 GB | 1.16 GB |
| **Memory read time** | ~1.1 ms | ~4.2 ms |

The BF16 lm_head adds **3.1 ms per token** - a ~10 tok/s penalty at this throughput level.

### GPU Time Breakdown (from nsys profile)

| Component | % GPU Time | Kernel |
|-----------|-----------|--------|
| Dense GEMV (lm_head) | ~36% | `gemvx::kernel` |
| MoE GEMM | ~34% | CUTLASS FP8×FP4 |
| Memory/Elementwise | ~18% | Various |
| Attention | ~1.5% | FlashInfer FA2 |

**lm_head is the bottleneck**, not MoE or attention.

### Implementation Details

The SM120/SM121 CUTLASS kernel was implemented with:
- Kernel types at **namespace scope** (critical fix for `Error Internal` bug)
- `OpClassBlockScaledTensorOp` with wrapped `mx_float8_t`/`mx_float4_t` types
- `KernelPtrArrayTmaWarpSpecializedPingpong` schedule
- 128×128×128 tile shape, 1×1×1 cluster
- Autotuner filtering to exclude unsupported `swap_ab=true` and `finalize_fusion` tactics

### Next Steps for Decode Optimization

1. **Implement MXFP4 for lm_head** - Potentially +30-45% decode speedup
2. **Fuse BF16→FP8 activations into MoE expand (remove standalone `mxfp8_quantize()`)** - Recover CUTLASS decode regression on SM121  
   - Plan: `docs/plans/SM121_MOE_FUSE_BF16_TO_FP8_EXPAND_PLAN.md`
2. **Enable CUDA graphs** - Remove `--enforce-eager`
3. **Test speculative decoding** - Eagle3 with CUTLASS backend
4. **Profile dense GEMV kernels** - Investigate cuBLAS vs custom implementations

---

## MXFP4 lm_head + CUTLASS MoE (2026-01-12)

**Status**: ✅ COMPLETE - lm_head MXFP4 implemented with Marlin kernel

### Configuration

```yaml
date: 2026-01-12
moe_kernel: CUTLASS (FlashInfer FP8×FP4 for SM12x)
lm_head: MXFP4 (Marlin fused dequant+GEMM)
attention_backend: FLASHINFER
sinks: enabled (default)
quantization: mxfp4
enforce_eager: true
load_format: safetensors
```

### Benchmark Results

```
llama-benchy 0.1.1
model: gpt-oss-120b @ http://localhost:8000/v1
latency_mode: api
runs: 10
```

| Test | Throughput (tok/s) | TTFR (ms) | Notes |
|------|-------------------|-----------|-------|
| **pp2048** | **4380.72 ± 315.12** | 405.33 ± 35.47 | Prefill |
| **tg32** | **34.44 ± 0.18** | - | Decode |

### Manual Profiling Results

| Output Length | Tokens | Time (ms) | Throughput (tok/s) |
|---------------|--------|-----------|-------------------|
| tg16 | 16 | 543.2 | 29.45 |
| tg32 | 32 | 961.9 | 33.27 |
| tg64 | 64 | 1879.6 | 34.05 |

### Comparison to Previous Results

| Config | tg32 (tok/s) | Change |
|--------|--------------|--------|
| Upstream Baseline (BF16 lm_head) | 31.6 | - |
| CUTLASS MoE (BF16 lm_head) | 29.0 | -8% |
| **CUTLASS MoE + MXFP4 lm_head** | **34.4** | **+9%** |
| **Target (llama.cpp)** | **52-58** | - |

### lm_head MXFP4 Quantization Log

```
[MXFP4] lm_head quantized: torch.Size([201088, 2880]) BF16 -> torch.Size([201088, 1440]) FP4 (4x smaller)
```

Memory savings: ~1.1 GB → ~0.28 GB (4x reduction)

### Gap Analysis

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| tg32 (tok/s) | 34.4 | 52 | **37% below** |
| tg32 (tok/s) | 34.4 | 58 | **41% below** |

### Components Active

- ✅ MoE: CUTLASS SM120 FP8×FP4 MXFP4 kernel
- ✅ lm_head: Marlin FP4 dequant+GEMM (4x smaller weights)
- ✅ Attention: FlashInfer CUTLASS (SM12x) with sinks

### Remaining Bottlenecks (Estimated)

| Component | Est. % of Decode Time | Optimization Opportunity |
|-----------|----------------------|--------------------------|
| MoE GEMM | ~34% | Already using CUTLASS FP8×FP4 |
| Dense GEMV | ~30-35% | Attention projections still BF16 |
| lm_head | ~6% | ✅ Now using Marlin MXFP4 |
| Attention | ~1.5% | Already optimized |
| Other | ~25% | Memory/elementwise ops |

### Notes on Marlin vs Native FP8×FP4

The lm_head uses Marlin (weight-only FP4 → dequant to BF16 → BF16 GEMM) rather than native FP8×FP4 MMA because:

1. Marlin is a proven, stable kernel for small-M GEMV
2. Native FP8×FP4 grouped GEMM has small-M inefficiency
3. lm_head is only ~6% of decode time - not the priority bottleneck

A native FP8×FP4 small-M kernel for lm_head could provide additional gains but is not the critical path to 52+ tok/s.

---

## Historical: CUTLASS Investigation (Pre-Fix)

**Status**: ⚠️ SUPERSEDED by above - 2026-01-10

Previous attempts at CUTLASS MXFP4 failed due to:
1. Missing SM12x dispatch in `moe_gemm_template_dispatch_tma_ws.h`
2. `Error Internal` from kernel types defined inside function templates
3. Incorrect tile shapes (128×128×256 not supported on SM120)
4. Missing autotuner filters for unsupported configurations

All issues resolved in the 2026-01-11 implementation.
