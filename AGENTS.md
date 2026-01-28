# AGENTS.md - Project Context for AI Assistants

## Mission

**Extend vLLM's lead as the fastest inference engine for gpt-oss-120b** on NVIDIA GB10 (SM121).

### Current Performance (2026-01-17)

| Context | Prefill (t/s) | Decode (t/s) |
|---------|---------------|--------------|
| Short (512) | 1,854 | **60.0** |
| Medium (2048) | 4,573 | **59.4** |
| Long (8192) | 6,628 | **57.5** |

### vs Competition

| Engine | Decode (t/s) | Status |
|--------|--------------|--------|
| SGLang | 52 | ✅ Beat by 10-15% |
| llama.cpp | 58 | ✅ Beat at short/medium context |
| **vLLM (this)** | **57-60** | **Winner** |

**Next Goal**: Further optimization through fused quantization, CUTLASS dense layers, and CUDA graph improvements.

---

## notes
- we do not require time estimates in plans
- avoid self-promotion like 'ai-assisted'
- **NEVER modify existing copyright headers** in any file
- **NEVER add copyright headers** to new or existing files
- **NEVER stop, restart, or recreate Docker containers without asking first** - the operator may have long-running processes (compilation, training, etc.) in the container. Always ask before executing `docker stop`, `docker restart`, `docker compose down`, or suggesting container recreation.
- **NEVER revert work** - even if a fix is incomplete or has issues, keep the progress and iterate forward. The user strongly dislikes reverting changes. Document limitations instead of undoing work.
  - **Git reverts**: Don't use `git checkout --`, `git reset`, `git checkout -B <branch> <upstream>` that would lose commits
  - **Code reverts**: When hitting an error after making a change, do NOT immediately revert the code change. Instead:
    1. **Investigate**: Is the constraint real (hardware/fundamental) or artificial (conservative defaults, untested assumptions)?
    2. **Ask**: If unsure, ask the user before reverting
    3. **Document**: If the constraint is real, document why and THEN consider alternatives
    4. **Iterate forward**: Try to fix the error with the new code, not by reverting to old code
  - When asked to "sync with upstream" or "switch branches", ALWAYS preserve existing patches:
    1. First, identify what patches/commits exist on the current branch
    2. Rebase onto upstream rather than resetting (preserves commits)
    3. Or cherry-pick/re-apply patches after any reset
  - **Example of bad behavior**: Hit "Ambiguous scatter" error with EpiN=8, immediately reverted to EpiN=16. Should have investigated: Is this a hardware limit? A swizzle pattern choice? Can the swizzle be changed?
  - When in doubt, ASK before any operation that could lose work
  - If work must be redone, use the previous implementation as a reference rather than reimplementing from scratch

### Code Search Warning

**ripgrep (`rg`) and internal IDE/agent search tools regularly return stale or incomplete results** in this project. This appears to be due to filesystem caching, Docker volume mounts, or index staleness.

**Always use traditional shell commands for code search:**

```bash
# Find files by name
find /workspace/flashinfer -name "*.py" | xargs grep -l "pattern"

# Search file contents
grep -rn "pattern" /workspace/flashinfer/flashinfer/

# Search with context
grep -rn -A3 -B3 "function_name" /workspace/flashinfer/

# Find and read specific symbols
grep -rn "^def select_tile" /workspace/flashinfer/flashinfer/fused_moe/
grep -rn "^class " /workspace/flashinfer/flashinfer/fused_moe/core.py
```

**Do NOT rely on:**
- `rg` (ripgrep) - returns stale/cached results
- IDE "Find in Files" - may use stale index
- Agent Grep/Glob tools - may show outdated file state

**When in doubt:** Use `cat` to read the actual file and verify content matches what search tools report.

### FlashInfer/CUTLASS C++ Coding Restrictions

FlashInfer and CUTLASS use **C++17** (not C++20). This affects template metaprogramming patterns:

**❌ C++20 features that will NOT compile:**

```cpp
// Non-type template parameters of class type (C++20 only)
template<auto Shape>  // FAILS: class type NTTP requires C++20
struct MyKernel { ... };

// Direct use of constexpr class objects as template args
using TileShape_SF = cute::Shape<cute::Int<128>, cute::Int<128>>;  // FAILS
```

**✓ C++17 workarounds that DO compile:**

```cpp
// Use decltype(make_shape(...)) pattern for type aliases
using TileShape_SFA = decltype(cute::make_shape(cute::Int<TileM_SFA>{}, cute::size<2>(TileShape{})));

// Use static constexpr for values, then wrap in Int<>
static constexpr int TileM_SFA = cutlass::ceil_div(...) * 128;
using sSFA_shapeM = decltype(prepend(Int<TileM_SFA>{} / Blk_MN{}, ...));

// Use if constexpr for conditional compile-time logic
if constexpr (!IsCtaM64) { CUTE_STATIC_ASSERT_V(...); }
```

**Key patterns in CUTLASS/CUTE:**

| Pattern | Usage |
|---------|-------|
| `cute::Int<N>{}` | Compile-time integer constant |
| `cute::size<I>(Shape{})` | Extract Ith dimension of a shape |
| `cute::shape<I>(Shape{})` | Same as size<I> |
| `decltype(cute::make_shape(...))` | Create shape type from components |
| `cutlass::ceil_div(a, b)` | Ceiling division for padding |
| `CUTE_STATIC_ASSERT_V(...)` | Compile-time shape validation |

**Compilation flags used:**
- `-std=c++17` for all JIT compilation
- `-gencode=arch=compute_121a,code=sm_121a` for SM121
- CUTLASS uses `CMAKE_CXX_STANDARD 17`

## Implementation Stack

### WE ARE USING

| Component | Implementation | Location |
|-----------|----------------|----------|
| **MoE GEMM** | CUTLASS SM120/121 grouped GEMM | `flashinfer/fused_moe/` |
| **Attention** | FlashAttention-2 (FA2) | `flashinfer/prefill.py`, `flashinfer/decode.py` |
| **FP4 Quantization** | FlashInfer `mxfp4_quantize()` | `flashinfer/fp4_quantization.py` |
| **Framework** | vLLM with FlashInfer backend | Local repos mounted in Docker |

### WE ARE NOT USING

| ❌ Component | Why Not |
|-------------|---------|
| **TensorRT-LLM** | We use CUTLASS directly |
| **trtllm_fp4_block_scale_moe** | SM100-only, crashes on SM121 |
| **FA3** | SM90 only; SM121 uses FA2 |
| **nvfp4** | We use MXFP4 (group size 32), not NVFP4 (group size 16) |

---

## Architecture: SM121 (GB10)

- **Compute Capability**: 12.1
- **Memory**: 120GB
- **Block-scaled MMA**: FP8×FP4 (not BF16×FP4 directly)
- **MXFP4 Path**: BF16 activations → quantize to FP8 → FP8×FP4 kernel

---

## Repository Layout

```
~/projects/
├── flashinfer/          # Local FlashInfer (CUTLASS kernels) - mxfp4_v2 branch
├── vllm/                # Local vLLM (uses FlashInfer backend) - mxfp4_v2 branch
└── ai/mxfp4/            # THIS REPO - Docker config + benchmarking
    ├── AGENTS.md        # This file
    ├── Dockerfile       # Production image (pinned SHAs)
    ├── docker-compose.yml        # Production deployment
    ├── docker-compose.dev.yml    # Development with mounted repos
    ├── docs/
    │   ├── MXFP4_V2_PLAN.md      # Optimization plan
    │   ├── BENCHMARK_RESULTS.md   # Benchmark tracking
    │   ├── UPSTREAM_TODOS.md      # Upstream improvement opportunities
    │   ├── analysis/              # Competitor analysis (llama.cpp, SGLang, TRT-LLM)
    │   ├── plans/                 # Feature plans (implemented and future)
    │   ├── reference/             # Technical reference, code reviews
    │   └── archive/               # Historical investigations
    └── scripts/
        ├── benchmarks/            # Benchmark scripts
        ├── tests/                 # Test scripts
        ├── profiling/             # Profiling scripts
        ├── utils/                 # Utility scripts
        └── debug/                 # Debug scripts
```

---

## Docker Environment

### Starting the Dev Container

```bash
cd ~/projects/ai/mxfp4
docker compose -f docker-compose.dev.yml up -d
docker exec -it vllm-dev bash
```

### Volume Mounts

Local repos are mounted into the container:
- `~/projects/vllm` → `/workspace/vllm`
- `~/projects/flashinfer` → `/workspace/flashinfer`

### Critical: PYTHONPATH

FlashInfer requires explicit PYTHONPATH to use local repo over installed wheel:

```bash
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
```

### Verify Correct Imports

```bash
docker exec vllm-dev python3 -c "import flashinfer; print(flashinfer.__file__)"
# ✓ Should print: /workspace/flashinfer/flashinfer/__init__.py
# ❌ NOT: /usr/local/lib/python3.x/site-packages/flashinfer/...
```

---

## vLLM Server Configuration

### Current Best Command

```bash
docker exec -it vllm-dev bash -c '
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gpt-oss-120b \
  --quantization mxfp4 \
  --mxfp4-backend CUTLASS \
  --mxfp4-layers moe,qkv,o,lm_head \
  --attention-backend FLASHINFER \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --load-format fastsafetensors
'
```

**Key flags:**
- `--mxfp4-layers moe,qkv,o,lm_head` - Quantize all layers (MoE, attention, LM head)
- `--attention-backend FLASHINFER` - Use FlashInfer for attention
- `--kv-cache-dtype fp8` - FP8 KV cache for memory efficiency
- No `--enforce-eager` - Enable CUDA graphs for better decode perf

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `VLLM_MXFP4_BACKEND` | **Unified backend selector** (recommended) |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` or `FLASH_ATTN` |
| `FLASHINFER_LOGLEVEL` | 0-5 for debug logging |

### MXFP4 Backend Options

**CLI argument (recommended):**
```bash
vllm serve ... --quantization mxfp4 --mxfp4-backend CUTLASS
```

**Environment variable:**
```bash
export VLLM_MXFP4_BACKEND=CUTLASS
```

| Backend | Description | SM Support |
|---------|-------------|------------|
| `auto` | Hardware-based auto-selection | - |
| `MARLIN` | Marlin dequant→BF16 | All GPUs |
| `CUTLASS` | FlashInfer CUTLASS FP8×FP4 | SM100, SM12x |
| `TRITON` | OpenAI Triton | SM90-SM100 |
| `TRTLLM` | TRT-LLM BF16×FP4 | SM100 only |
| `TRTLLM_MXFP8` | TRT-LLM FP8×FP4 | SM100 only |

### SM12x Configuration (GB10)

For SM12x native CUTLASS MXFP4 (once Phase 2 is complete):
```bash
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --mxfp4-backend CUTLASS
```

**Data Flow**:
1. vLLM receives BF16 activations from model
2. vLLM calls `mxfp8_quantize()` to convert BF16 → FP8
3. FlashInfer CUTLASS kernel executes FP8×FP4 GEMM on SM12x tensor cores

### CLI Arguments

#### `--mxfp4-layers` - Layer Selection

Control which layer types are quantized with MXFP4:

| Token | Layers Matched | Description |
|-------|----------------|-------------|
| `moe` | `*.experts.*` | MoE expert weights (default) |
| `qkv` | `*.qkv_proj` | Fused QKV projection |
| `o` | `*.o_proj` | Attention output projection |
| `lm_head` | `lm_head` | Output logits projection |
| `all` | All above | Shorthand for full quantization |

**Default**: `--mxfp4-layers moe` (backwards compatible)

**Recommended**: `--mxfp4-layers moe,qkv,o,lm_head` (full quantization, best performance)

**Compatibility notes:**
- LoRA: When enabled, QKV/O/lm_head fall back to BF16
- Tied embeddings: lm_head falls back to BF16 when `tie_word_embeddings=True`

### Environment Variables Reference

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_MXFP4_BACKEND` | `auto`, `MARLIN`, `CUTLASS`, `TRITON` | Backend selection (recommended over env vars below) |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER`, `FLASH_ATTN` | Attention backend |
| `FLASHINFER_LOGLEVEL` | `0`-`5` | API logging verbosity |
| `FLASHINFER_JIT_VERBOSE` | `0`, `1` | JIT compilation output |
| `FLASHINFER_NVCC_THREADS` | number | Parallel JIT threads |

### Deprecated Variables

The following are deprecated - use `--mxfp4-backend` CLI instead:
- `VLLM_MXFP4_MOE_KERNEL`
- `VLLM_MXFP4_ACTIVATION`
- `VLLM_USE_FLASHINFER_MOE_MXFP4_*`

---

## Correct API Usage

### MoE GEMM on SM121

```python
# ✓ CORRECT: Use cutlass_fused_moe
from flashinfer.fused_moe import cutlass_fused_moe

output = cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_indices,
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=torch.bfloat16,
    activation_type=ActivationType.Swiglu,
)
```

```python
# ❌ WRONG: Do NOT use on SM121
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe  # SM100 ONLY!
```

### MXFP4 Quantization

```python
# ✓ CORRECT: MXFP4 (group size 32)
from flashinfer import mxfp4_quantize
weight_fp4, weight_scale = mxfp4_quantize(weight_bf16)
```

```python
# ❌ WRONG: NVFP4 (group size 16) - different format
from flashinfer import nvfp4_quantize
```

---

## Development Workflow

### Editing Kernels

1. Edit on host: `vim ~/projects/flashinfer/include/flashinfer/moe/...`
2. Clear JIT cache (see below for selective clearing)
3. Run test - FlashInfer JIT recompiles automatically

### Editing Python

Changes are immediate - no cache clearing needed.

### FlashInfer JIT Cache Structure

The JIT cache is organized by architecture and operation type:

```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120/           # SM120 MoE GEMM kernels (~10min to rebuild all)
│   ├── moe_gemm_kernels_bf16_fp4.cuda.o   # MXFP4 (BF16 act)
│   ├── moe_gemm_kernels_fp16_fp4.cuda.o   # MXFP4 (FP16 act)
│   ├── moe_gemm_kernels_fp4_fp4.cuda.o    # NVFP4
│   └── fused_moe_120.so                   # Final library
├── fp4_quantization_121/    # FP4 quantization ops
└── ...                      # Attention, other ops
```

### Selective Cache Clearing (Faster Testing)

**Full clear** (slow - rebuilds everything ~10+ min):
```bash
rm -rf ~/.cache/flashinfer/0.6.0/121a/cached_ops/*
```

**MoE-only clear** (fast - rebuilds only MoE kernels ~2-5 min):
```bash
# Clear ONLY SM120 MoE GEMM kernels
rm -rf ~/.cache/flashinfer/0.6.0/121a/cached_ops/fused_moe_120/
```

**Inside Docker**:
```bash
docker exec vllm-dev rm -rf /root/.cache/flashinfer/0.6.0/121a/cached_ops/fused_moe_120/
```

Use selective clearing when iterating on MoE launcher code to save significant build time.

### After Container Recreation

```bash
cd /workspace/flashinfer && uv pip install --no-build-isolation -e .
cd /workspace/vllm && python3 use_existing_torch.py && \
  uv pip install -r requirements/build.txt && \
  uv pip install --no-build-isolation -e .
```

---

## Testing Protocol

| Test | Script | Purpose |
|------|--------|---------|
| Smoke | `scripts/tests/smoke_test_basic.py` | Basic kernel compilation |
| Quantize | `scripts/tests/smoke_test_proper_quantize.py` | Full MXFP4 quantization path |
| Numerical | `scripts/tests/test_numerical_accuracy.py` | Compare vs BF16 reference |
| Benchmark | `scripts/benchmarks/benchmark_tile_shapes.py` | Tile performance comparison |
| E2E | `llama-benchy --base-url ... --pp 2048 --tg 32 128` | Full vLLM benchmark |

---

## Quick Reference

### Verify Kernel Path

```bash
# Check which MoE kernel is being used
FLASHINFER_LOGLEVEL=3 vllm serve ... 2>&1 | grep -i "moe\|kernel\|cutlass"
```

### Clear All Caches

```bash
# On host (outside Docker):
rm -rf ~/projects/ai/mxfp4/.cache/flashinfer/*
rm -rf ~/projects/ai/mxfp4/.cache/vllm/*

# Inside Docker container:
rm -rf ~/.cache/flashinfer/*
rm -rf ~/.cache/vllm/*
```

**Note:** The cache directories are Docker mounts. Delete the *contents* (`/*`), not the directories themselves, or Docker bind mounts will break.

### Run Baseline Benchmark

```bash
llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --tokenizer openai/gpt-oss-120b \
  --pp 2048 \
  --tg 32 128 \
  --runs 5
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/MXFP4_V2_PLAN.md` | Full optimization plan and methodology |
| `docs/BENCHMARK_RESULTS.md` | Live benchmark tracking |
| `docs/UPSTREAM_TODOS.md` | FlashInfer improvement opportunities |
| `docs/plans/SM120_MOE_TILE_EXPANSION.md` | Small tile implementation (64×128) |
| `docs/reference/SM121_TECHNICAL_GUIDE.md` | Architecture deep dive |
| `docs/reference/SM121_OPTIMIZATION_ANALYSIS.md` | Competitor analysis and optimization gaps |
| `docs/analysis/` | llama.cpp, SGLang, TensorRT-LLM analysis |

---

## Summary for AI Agents

1. **We achieved 57-60 tok/s decode** - beating SGLang (52) and llama.cpp (58)
2. **We use CUTLASS via FlashInfer**, not TensorRT-LLM
3. **Correct MoE API is `cutlass_fused_moe`**
4. **SM121 uses FA2** (not FA3)
5. **MXFP4 uses group size 32**, goes through FP8×FP4 kernel
6. **Always verify PYTHONPATH** points to local repos
7. **KV cache is HND layout** on SM121

If debugging crashes: verify which FlashInfer you're importing first.
