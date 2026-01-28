#!/bin/bash
# =============================================================================
# Start vLLM with optimized MXFP4 settings for DGX Spark (SM121/GB10)
# Achieves 59.4 tok/s decode with CUTLASS MXFP4 kernel
# =============================================================================

set -e

# Configuration
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

echo "=============================================="
echo "Starting vLLM MXFP4 Server"
echo "=============================================="
echo "Model: $MODEL"
echo "Endpoint: http://$HOST:$PORT"
echo "=============================================="

docker exec -it vllm-dev bash -c "
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
# Disable GPU Direct Storage for fastsafetensors (not available on this platform)
export VLLM_FASTSAFETENSORS_NOGDS=1

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
    --load-format fastsafetensors
"
