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

# Clean up stale Ray state that --force leaves behind
rm -rf /tmp/ray/session_* 2>/dev/null && echo 'Cleaned up stale Ray sessions'
rm -f /dev/shm/sem.loky-* 2>/dev/null && echo 'Cleaned up stale semaphores'

echo 'Ray stopped.'
"
