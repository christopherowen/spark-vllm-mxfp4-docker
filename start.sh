#!/bin/bash
# =============================================================================
# Start vLLM with optimized MXFP4 settings for DGX Spark (SM121/GB10)
# Achieves 59.4 tok/s decode with CUTLASS MXFP4 kernel
#
# Environment variables:
#   VLLM_ENFORCE_EAGER=1    - Disable CUDA graphs
#   FLASHINFER_LOGLEVEL=3   - Debug logging (0=off, 3=debug, 5=trace)
# =============================================================================

set -e

# Configuration
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
FLASHINFER_LOGLEVEL="${FLASHINFER_LOGLEVEL:-0}"

# Host-side expansion for flags that must reach the container CLI.
# (Don't rely on expanding $VLLM_* inside the container; those vars are not
# exported into the docker exec environment by default.)
EAGER_FLAG=""
if [[ "${ENFORCE_EAGER}" = "1" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

# Optional: write vLLM stdout/stderr to a file INSIDE the container.
# Example (used by sweep script):
#   VLLM_LOG_FILE=/tmp/vllm_64x128.log ./start.sh
LOG_FILE="${VLLM_LOG_FILE:-}"

# Optional: vLLM python logging dictConfig path (inside container).
# If set, vLLM will use this config for logging (APIServer + EngineCore).
VLLM_LOGGING_CONFIG_PATH="${VLLM_LOGGING_CONFIG_PATH:-}"
VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"

# Allow non-interactive / detached runs (useful for automation).
# Examples:
#   VLLM_DOCKER_EXEC_FLAGS="-d" ./start.sh         # detach vLLM inside container
#   VLLM_DOCKER_EXEC_FLAGS="-i" ./start.sh         # no TTY
DOCKER_EXEC_FLAGS="${VLLM_DOCKER_EXEC_FLAGS:--it}"

echo "=============================================="
echo "Starting vLLM MXFP4 Server"
echo "=============================================="
echo "Model: $MODEL"
echo "Endpoint: http://$HOST:$PORT"
echo "=============================================="

docker exec ${DOCKER_EXEC_FLAGS} "${CONTAINER}" bash -c "
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
# Disable GPU Direct Storage for fastsafetensors (not available on this platform)
export VLLM_FASTSAFETENSORS_NOGDS=1
# FlashInfer log level (0=off, 1=API calls, 3=debug, 5=trace)
export FLASHINFER_LOGLEVEL=\"${FLASHINFER_LOGLEVEL}\"

LOG_FILE=\"${LOG_FILE}\"
VLLM_LOGGING_CONFIG_PATH=\"${VLLM_LOGGING_CONFIG_PATH}\"
VLLM_CONFIGURE_LOGGING=\"${VLLM_CONFIGURE_LOGGING}\"

export VLLM_CONFIGURE_LOGGING
if [[ -n \"\$VLLM_LOGGING_CONFIG_PATH\" ]]; then
  export VLLM_LOGGING_CONFIG_PATH
fi

if [[ -n \"\$LOG_FILE\" ]]; then
  mkdir -p \"\$(dirname \"\$LOG_FILE\")\"
  exec > >(tee -a \"\$LOG_FILE\") 2>&1
fi

vllm serve $MODEL \\
    --host $HOST \\
    --port $PORT \\
    --served-model-name gpt-oss-120b \\
    --quantization mxfp4 \\
    --mxfp4-backend CUTLASS \\
    --mxfp4-layers moe,qkv,o,lm_head \\
    --attention-backend FLASHINFER \\
    --kv-cache-dtype fp8 \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.70 \\
    --max-model-len 131072 \\
    --max-num-seqs 2 \\
    --max-num-batched-tokens 8192 \\
    --enable-prefix-caching \\
    --load-format fastsafetensors \\
    ${EAGER_FLAG} \\
    ${EXTRA_ARGS}
"
