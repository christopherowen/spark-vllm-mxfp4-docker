#!/bin/bash
# =============================================================================
# Start vLLM in cluster mode with TP=2 across two nodes
# Run ray-master.sh on this node and ray-slave.sh on the other node FIRST
# =============================================================================

set -e

# Configuration
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

echo "=============================================="
echo "Starting vLLM MXFP4 Cluster Server (TP=2)"
echo "=============================================="
echo "Model: $MODEL"
echo "Endpoint: http://$HOST:$PORT"
echo "Tensor Parallel: 2 (across Ray cluster)"
echo "=============================================="
echo ""
echo "Make sure ray-master.sh is running on this node"
echo "and ray-slave.sh is running on the other node!"
echo ""

# Determine host IP based on current hostname
CURRENT_HOST=$(hostname)
case "$CURRENT_HOST" in
    dgx1*)
        INTERNAL_HOST="dgx1-internal"
        ;;
    dgx2*)
        INTERNAL_HOST="dgx2-internal"
        ;;
    *)
        echo "ERROR: Unknown hostname '$CURRENT_HOST'. Expected dgx1* or dgx2*"
        exit 1
        ;;
esac

docker exec -it vllm-dev bash -c "
# Get this host's IP for VLLM_HOST_IP
export VLLM_HOST_IP=\$(getent hosts ${INTERNAL_HOST} | awk '{print \$1}')
echo \"VLLM_HOST_IP: \$VLLM_HOST_IP\"

# Ray IP address environment variables - ensures all Ray components use correct IP
export RAY_NODE_IP_ADDRESS=\$VLLM_HOST_IP
export RAY_OVERRIDE_NODE_IP_ADDRESS=\$VLLM_HOST_IP

export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1

export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1
export NCCL_DEBUG=INFO

# Opt into future Ray behavior for accelerator env var handling (silences FutureWarning)
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Disable Ray memory monitor refresh (reduces overhead)
export RAY_memory_monitor_refresh_ms=0
# Note: metrics exporter agent errors will appear but are harmless
# Container has minimal Ray without dashboard deps - metrics will not be exported
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
    --tensor-parallel-size 2 \\
    --distributed-executor-backend ray \\
    --gpu-memory-utilization 0.70 \\
    --max-model-len 131072 \\
    --max-num-seqs 2 \\
    --max-num-batched-tokens 8192 \\
    --enable-prefix-caching \\
    --load-format fastsafetensors
"
