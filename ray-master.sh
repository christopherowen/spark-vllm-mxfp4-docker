#!/bin/bash
# =============================================================================
# Start Ray head node for multi-node vLLM cluster
# Run this on the master node (the one that will run vllm serve)
# =============================================================================

set -e

# Determine internal hostname
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

echo "=============================================="
echo "Starting Ray Head Node"
echo "=============================================="
echo "Hostname: $CURRENT_HOST"
echo "Internal: $INTERNAL_HOST"
echo "=============================================="

docker exec -i vllm-dev bash -c "
# Set VLLM_HOST_IP for consistent IP usage
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

# Opt into future Ray behavior for accelerator env var handling (silences FutureWarning)
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Disable Ray memory monitor refresh (reduces overhead)
export RAY_memory_monitor_refresh_ms=0
# Note: metrics exporter agent errors will appear but are harmless
# Container has minimal Ray without dashboard deps - metrics will not be exported
# Disable GPU Direct Storage for fastsafetensors (not available on this platform)
export VLLM_FASTSAFETENSORS_NOGDS=1

# Stop any existing Ray instance
ray stop 2>/dev/null || true

# Start Ray head
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=\$VLLM_HOST_IP

echo ''
echo '=== Ray Head Started ==='
echo 'Workers should connect now, then run vllm serve on this node'
echo ''
ray status
"
