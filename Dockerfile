# =============================================================================
# Dockerfile for GPT-OSS-120B with MXFP4 on DGX Spark (SM121/GB10)
# =============================================================================
#
# Optimized MXFP4 inference achieving 59.4 tok/s decode on GB10.
#
# Build:
#   docker build -t vllm-mxfp4-spark .
#
# Run with docker-compose (recommended):
#   docker compose up -d
#
# Run standalone:
#   docker run --gpus all -p 8000:8000 \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       vllm-mxfp4-spark \
#       vllm serve openai/gpt-oss-120b --quantization mxfp4
#
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL maintainer="MXFP4 Optimization Project"
LABEL description="vLLM with optimized MXFP4 for DGX Spark (SM121/GB10)"

# =============================================================================
# Pinned versions - our tested configurations
# =============================================================================

ARG VLLM_SHA=459541683f2d8c21f9c0e2f44954b04f59611cbe
ARG FLASHINFER_SHA=f349e52496a72a00d8c4ac02c7a1e38523ff7194
ARG CUTLASS_SHA=11af7f02ab52c9130e422eeb4b44042fbd60c083

ARG VLLM_REPO=https://github.com/christopherowen/vllm.git
ARG FLASHINFER_REPO=https://github.com/christopherowen/flashinfer.git
ARG CUTLASS_REPO=https://github.com/christopherowen/cutlass.git

# Build parallelism
ARG BUILD_JOBS=16

# =============================================================================
# Environment configuration
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# UV package manager
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

# Ccache for faster rebuilds
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# FlashInfer settings
ENV FLASHINFER_CUDA_ARCH_LIST="12.1f"
ENV FLASHINFER_JIT_VERBOSE=0
ENV FLASHINFER_LOGLEVEL=0
ENV FLASHINFER_NVCC_THREADS=4

# CUDA architecture - SM120/SM121 only (DGX Spark)
# This avoids compiling for SM80/SM90 which wastes build time
# TORCH_CUDA_ARCH_LIST: Used by PyTorch/vLLM extension builds
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"

# Model cache (HF_HOME is the modern unified path)
ENV HF_HOME=/root/.cache/huggingface

# Use local repos
ENV PYTHONPATH=/workspace/flashinfer:/workspace/vllm

WORKDIR /workspace

# =============================================================================
# System dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv \
    && pip uninstall -y flash-attn 2>/dev/null || true  # Remove NGC flash-attn to avoid operator conflicts

# =============================================================================
# Clone repositories at pinned versions
# =============================================================================

# Clone FlashInfer (cached for faster rebuilds)
RUN --mount=type=cache,id=git-flashinfer,target=/git-cache/flashinfer \
    if [ -d /git-cache/flashinfer/.git ]; then \
        echo "=== Using cached FlashInfer repo ===" && \
        cp -a /git-cache/flashinfer /workspace/flashinfer && \
        cd /workspace/flashinfer && \
        git fetch origin; \
    else \
        echo "=== Cloning FlashInfer (first build) ===" && \
        git clone ${FLASHINFER_REPO} /workspace/flashinfer && \
        cp -a /workspace/flashinfer /git-cache/flashinfer; \
    fi && \
    cd /workspace/flashinfer && git checkout ${FLASHINFER_SHA}

# Clone spdlog submodule (small, no caching needed)
RUN cd /workspace/flashinfer && \
    git submodule update --init 3rdparty/spdlog

# Clone CUTLASS directly (skip submodule, use our fork)
RUN --mount=type=cache,id=git-cutlass,target=/git-cache/cutlass \
    cd /workspace/flashinfer && \
    rm -rf 3rdparty/cutlass && \
    if [ -d /git-cache/cutlass/.git ] && [ -d /git-cache/cutlass/.git/objects ]; then \
        echo "=== Using cached CUTLASS repo ===" && \
        cp -a /git-cache/cutlass 3rdparty/cutlass && \
        cd 3rdparty/cutlass && \
        git fetch origin; \
    else \
        echo "=== Cloning CUTLASS (first build) ===" && \
        rm -rf /git-cache/cutlass/* /git-cache/cutlass/.* 2>/dev/null || true && \
        git clone ${CUTLASS_REPO} 3rdparty/cutlass && \
        cp -a /workspace/flashinfer/3rdparty/cutlass/. /git-cache/cutlass/; \
    fi && \
    cd /workspace/flashinfer/3rdparty/cutlass && git checkout ${CUTLASS_SHA}

# Clone vLLM (cached for faster rebuilds)
RUN --mount=type=cache,id=git-vllm,target=/git-cache/vllm \
    if [ -d /git-cache/vllm/.git ]; then \
        echo "=== Using cached vLLM repo ===" && \
        cp -a /git-cache/vllm /workspace/vllm && \
        cd /workspace/vllm && \
        git fetch origin; \
    else \
        echo "=== Cloning vLLM (first build) ===" && \
        git clone ${VLLM_REPO} /workspace/vllm && \
        cp -a /workspace/vllm /git-cache/vllm; \
    fi && \
    cd /workspace/vllm && git checkout ${VLLM_SHA} && \
    git submodule update --init --recursive

# =============================================================================
# Build FlashInfer
# =============================================================================

WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    uv pip install --no-build-isolation -e .

# =============================================================================
# Build vLLM
# =============================================================================

WORKDIR /workspace/vllm

# Prepare build (use existing torch from NGC container)
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt 2>/dev/null || true

# Install build dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

# Install runtime dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/cuda.txt -r requirements/common.txt 2>/dev/null || \
    uv pip install -r requirements.txt 2>/dev/null || true

# Build and install vLLM
# Note: This step compiles CUDA kernels and takes 20-40 minutes on first build.
# Ccache speeds up subsequent rebuilds significantly.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=vllm-build,target=/workspace/vllm/build \
    uv pip install --no-build-isolation --no-deps -e .

# Install additional tools
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install fastsafetensors llama-benchy

# =============================================================================
# Download tiktoken encodings (cached)
# =============================================================================

RUN --mount=type=cache,id=tiktoken-cache,target=/tiktoken-cache \
    mkdir -p /workspace/tiktoken_encodings && \
    if [ -f "/tiktoken-cache/o200k_base.tiktoken" ] && [ -f "/tiktoken-cache/cl100k_base.tiktoken" ]; then \
        echo "=== Using cached tiktoken encodings ===" && \
        cp /tiktoken-cache/*.tiktoken /workspace/tiktoken_encodings/; \
    else \
        echo "=== Downloading tiktoken encodings ===" && \
        wget -q -O /tiktoken-cache/o200k_base.tiktoken \
            "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
        wget -q -O /tiktoken-cache/cl100k_base.tiktoken \
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" && \
        cp /tiktoken-cache/*.tiktoken /workspace/tiktoken_encodings/; \
    fi

# =============================================================================
# Remove opencv (causes circular import with typing module)
# =============================================================================

RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true && \
    rm -f /usr/local/lib/python3.12/dist-packages/*opencv*.pth \
          /usr/local/lib/python3.12/dist-packages/*cv2*.pth 2>/dev/null || true

# =============================================================================
# Create entrypoint script (validation only)
# =============================================================================

RUN cat > /workspace/entrypoint.sh << 'ENTRYPOINT_SCRIPT'
#!/bin/bash
set -e

echo "=============================================="
echo "MXFP4 vLLM for DGX Spark (SM121/GB10)"
echo "=============================================="
echo ""
echo "Git SHAs:"
echo "  vLLM:       $(cd /workspace/vllm && git rev-parse --short HEAD)"
echo "  FlashInfer: $(cd /workspace/flashinfer && git rev-parse --short HEAD)"
echo "  CUTLASS:    $(cd /workspace/flashinfer/3rdparty/cutlass && git rev-parse --short HEAD)"
echo ""

# Validate GPU
python3 -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability()
print(f'Compute Capability: SM{cc[0]}{cc[1]}')
"

# Check if model is downloaded
MODEL_PATH="/root/.cache/huggingface/hub/models--openai--gpt-oss-120b"
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "WARNING: Model not found at $MODEL_PATH"
    echo ""
    echo "Download the model first (240GB):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  uv tool install huggingface_hub"
    echo "  hf download openai/gpt-oss-120b"
    echo ""
    echo "Or vLLM will attempt to download it (slow)."
    echo ""
fi

echo ""
echo "=============================================="
echo ""

exec "$@"
ENTRYPOINT_SCRIPT

RUN chmod +x /workspace/entrypoint.sh

# =============================================================================
# Runtime configuration
# =============================================================================

WORKDIR /workspace

RUN mkdir -p ${HF_HOME}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
