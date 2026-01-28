#!/bin/bash
# =============================================================================
# Stop Ray on this node
# Run on both master and worker nodes to fully shut down the cluster
# =============================================================================

echo "=============================================="
echo "Stopping Ray on $(hostname)"
echo "=============================================="

docker exec -i vllm-dev bash -c "
ray stop --force
echo 'Ray stopped.'
"
