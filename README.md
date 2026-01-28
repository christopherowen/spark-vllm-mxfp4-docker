# GPT-OSS-120B with MXFP4 on DGX Spark (SM121/GB10)

**Fastest gpt-oss-120b inference on DGX Spark** - **72 tok/s decode with TP=2**, beating all competitors.

## Benchmark Results

### TP=2 (Two-Node with RDMA) - New Record!

| Context | Prefill (t/s) | Decode tg32 (t/s) | Decode tg128 (t/s) |
|---------|---------------|-------------------|---------------------|
| **2048** | **6,329** | **72.71** | **71.43** |

### TP=1 (Single Node)

| Context | Prefill (t/s) | Decode tg32 (t/s) | Decode tg128 (t/s) |
|---------|---------------|-------------------|---------------------|
| **Short (512)** | 1,854 | **60.02** | **60.07** |
| **Medium (2048)** | 4,573 | **59.36** | **59.47** |
| **Long (8192)** | 6,628 | **57.52** | **57.81** |

### Key Observations

- ✅ **TP=2 with RDMA achieves 72 tok/s** - 20% faster than single-node
- ✅ Proper `/dev/infiniband` device mount enables RoCE transport
- ✅ Decode consistently **57-60 tok/s** on TP=1 across all context lengths
- ✅ Prefill scales well: 1.8K → 4.6K → 6.6K t/s as batch size increases

### vs Competitors

| Engine | Decode (t/s) | Status |
|--------|--------------|--------|
| SGLang | 52 | ✅ Beat by 38% (TP=2) |
| llama.cpp | 58 | ✅ Beat by 24% (TP=2) |
| **vLLM TP=1** | **57-60** | Previous best |
| **vLLM TP=2 RDMA** | **72** | **New Champion** |

See the [discussion on NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651/18) for more details.

---

## What's Included

### SM120/SM121 MXFP4 MoE Kernel
- **First implementation** of CUTLASS block-scaled MXFP4 MoE GEMM for DGX Spark (GB10)
- Automatic tile and schedule selection:
  - **Decode**: 64×128 tiles with PingPong schedule (double-buffered, better for small batches)
  - **Prefill**: 128×128 tiles with Cooperative schedule (warps share tiles, better for large batches)
- CUTLASS patches enabling small-tile compilation (previously broken)
- 2x decode throughput improvement over baseline

### Full MXFP4 Quantization Stack  
- **MoE layers** - CUTLASS FP8×FP4 grouped GEMM
- **QKV projections** - MXFP4 attention inputs
- **O projections** - MXFP4 attention outputs
- **LM head** - MXFP4 logits computation with Blackwell detection

### Unified Configuration API
- `--mxfp4-backend` - Single flag to select CUTLASS, MARLIN, TRITON, or auto
- `--mxfp4-layers` - Fine-grained control over which layers to quantize
- Clean deprecation of legacy environment variables

### FP8 KV Cache with Attention Sinks
- FP8 E4M3 KV cache support for memory efficiency
- Compatible with GPT-OSS-120B attention sink mechanism

### Production-Ready Docker
- 30-minute initial build, instant rebuilds on updates
- Git/ccache/pip caching with BuildKit
- Optimized docker-compose with tuned vLLM settings

---

## Quick Start

### Build

```bash
docker build -t vllm-mxfp4-spark .
```

Note: First build takes ~30 minutes (compiling CUDA kernels). Subsequent builds are faster due to ccache.

### Download the Model

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Install huggingface hub CLI
uv tool install huggingface_hub

# Download gpt-oss-120b (~240GB)
hf download openai/gpt-oss-120b
```

### Run with Docker Compose (Recommended)

```bash
# Start the server
docker compose up -d

# View logs
docker compose logs -f

# Enter the container
docker compose exec vllm-mxfp4 bash

# Stop
docker compose down
```

### Run with Docker

```bash
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./.cache/flashinfer:/root/.cache/flashinfer \
    vllm-mxfp4-spark
```

The volume mounts persist the model cache and JIT-compiled kernels between runs.

---

## Replicate Benchmark Results

Once the server is running, run the benchmark:

```bash
docker compose exec vllm-mxfp4 llama-benchy \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --pp 512 2048 8192 \
    --tg 32 128 \
    --runs 5
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Are we fast yet?"}]
  }'
```

---

## Configuration

The server starts with optimized settings for gpt-oss-120b:

| Setting | Value |
|---------|-------|
| Quantization | MXFP4 with CUTLASS backend |
| Attention | FlashInfer with FP8 KV cache |
| Max context | 131072 tokens |
| GPU memory | 70% utilization |

See `docker-compose.yml` for full configuration.

---

## Development

For development with local FlashInfer/vLLM repos:

```bash
# Clone the repos (one-time setup)
mkdir -p ~/projects && cd ~/projects
git clone -b mxfp4_v2 https://github.com/christopherowen/vllm.git
git clone -b mxfp4_v2 https://github.com/christopherowen/flashinfer.git

# Initialize submodules and switch CUTLASS to our fork
cd flashinfer
git submodule update --init --recursive
cd 3rdparty/cutlass
git remote set-url origin https://github.com/christopherowen/cutlass.git
git fetch origin
git checkout mxfp4_v2
cd ~/projects

# Start development container
cd ~/projects/ai/mxfp4
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml exec dev bash
```

---

## Build Cache

The Dockerfile uses Docker BuildKit cache mounts to speed up rebuilds:
- **git-flashinfer, git-vllm, git-cutlass**: Git repo caches
- **uv-cache**: Python package cache
- **ccache**: C++/CUDA compilation cache

First build takes ~30 minutes. Subsequent builds are much faster.

To clear caches:

```bash
# Clear Docker BuildKit caches (git repos, ccache, pip)
docker builder prune --filter type=exec.cachemount

# Clear FlashInfer JIT cache (volume mount)
sudo rm -rf .cache/
```

---

## Future Work

**Upstreaming**
- Contribute small-tile CUTLASS patches (64×128) to FlashInfer upstream
- Contribute MXFP4 layer selection (`--mxfp4-layers`) and SM121 fixes to vLLM upstream
- Work with NVIDIA on CUTLASS Blackwell block-scaled improvements

**Performance Optimizations**
- Use FlashInfer autotuner to dynamically select optimal tile shapes and CUTLASS schedules (PingPong vs Cooperative) based on workload ([plan](docs/plans/SMALL_TILE_SUPPORT_PLAN.md))
- Benchmark additional tile shapes (64×64, 64×256, 128×64) for different workload sizes
- Benchmark small-M tiles (M=8, 16, 32) and small-N tiles for decode-oriented workloads
- Fuse activation quantization directly into MoE GEMM kernel ([plan](docs/plans/SM121_MOE_FUSE_BF16_TO_FP8_EXPAND_PLAN.md))
- Native FP4×FP4 block-scale MMA (llama.cpp uses `mxf4.block_scale` instruction, we use FP8×FP4)
- Low-M CUDA core dispatch for dense layers (TRT-LLM uses CUDA cores for M≤4, we always use tensor cores)
- Evaluate [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) as alternative to CUTLASS (simpler tile primitives, Blackwell MXFP8/NVFP4 support)
- Use CUTLASS for dense layers (QKV, O projections, LM head) instead of Marlin fallback

**Multi-Node / Ray Cluster**
- Enable async scheduling for Ray executor backend - currently disabled because Ray Compiled DAGs are compiled with `enable_asyncio=False`. The Ray executor already has `max_concurrent_batches=2` logic; enabling async would allow batch pipelining (schedule next batch while current executes). Requires: (1) change `enable_asyncio=True` in `ray_executor.py`, (2) add `"ray"` to `executor_supports_async_sched` in `vllm/config/vllm.py`, (3) test for deadlocks/race conditions in multi-node setting
- Install `ray[default]` in container for full dashboard/metrics support (currently using minimal Ray without `aiohttp_cors`)

**New Features**
- Speculative decoding with Eagle3 tree-based verification
- Support for multiple quantization modes: native BF16, MXFP8, MXFP4
- NVFP4 format support (group size 16, vs MXFP4's group size 32)
- LoRA adapter support

**Numerical Accuracy**
- Full-scale activation quantization (currently using identity scales for simplicity)

---

## Documentation

- [AGENTS.md](AGENTS.md) - Project context and AI assistant guide
- [docs/reference/](docs/reference/) - Technical deep dive (SM121 architecture, CUTLASS details)
- [docs/plans/](docs/plans/) - Feature implementation documentation

### Competitor Analysis

- [SGLang Analysis](docs/analysis/SGLANG_ANALYSIS.md) - How SGLang achieves 52 tok/s
- [llama.cpp Analysis](docs/analysis/LLAMA_CPP_ANALYSIS.md) - How llama.cpp achieves 58 tok/s
- [TensorRT-LLM Analysis](docs/analysis/TENSORRT_LLM_ANALYSIS.md) - Techniques to learn from TRT-LLM
- [vLLM Baseline Analysis](docs/analysis/VLLM_BASELINE_ANALYSIS.md) - Why vLLM started at 30 tok/s

---

## Pinned Versions

| Component | SHA | Repository |
|-----------|-----|------------|
| vLLM | `45954168` | [christopherowen/vllm](https://github.com/christopherowen/vllm/tree/mxfp4_v2) |
| FlashInfer | `1660ee8d` | [christopherowen/flashinfer](https://github.com/christopherowen/flashinfer/tree/mxfp4_v2) |
| CUTLASS | `11af7f02` | [christopherowen/cutlass](https://github.com/christopherowen/cutlass/tree/mxfp4_v2) |
