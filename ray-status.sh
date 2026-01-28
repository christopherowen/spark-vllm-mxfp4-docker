#!/bin/bash
# =============================================================================
# Check Ray cluster status
# =============================================================================

echo "=============================================="
echo "Ray Cluster Status on $(hostname)"
echo "=============================================="

docker exec -i vllm-dev bash -c "
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
ray status
"
