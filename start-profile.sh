#!/bin/bash
# =============================================================================
# Start vLLM under nsys, run bench.sh with profiling, collect results.
#
# Uses vLLM's built-in CUDA profiler support:
#   - Server: --profiler-config.profiler cuda (enables cudaProfilerStart/Stop)
#   - nsys: --capture-range=cudaProfilerApi (only records during profiler window)
#   - Client: curl /start_profile, bench.sh, curl /stop_profile
#
# This captures kernel data from the EngineCore child process because nsys
# with --trace-fork-before-exec injects into all children, and the profiler
# API calls happen inside the worker process.
#
# IMPORTANT: Run a clean benchmark first (start.sh + bench.sh) to warm the
# FlashInfer JIT cache. Cold JIT compilation spawns many nvcc/ptxas forks
# that interfere with nsys's --trace-fork-before-exec tracing.
#
# Uses --gpu-memory-utilization 0.60 (not 0.70) to leave room for CUPTI
# overhead. Without this, nsys causes CUDA OOM during model loading.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-/tmp/nsys_profile/vllm_inference}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
FLASHINFER_LOGLEVEL="${FLASHINFER_LOGLEVEL:-0}"

echo "=============================================="
echo "vLLM MXFP4 Profiler (cudaProfilerApi capture)"
echo "=============================================="
echo "Model: $MODEL"
echo "Endpoint: http://$HOST:$PORT"
echo "Profile output: $PROFILE_OUTPUT"
echo "gpu-memory-utilization: 0.60 (reduced for nsys CUPTI overhead)"
echo "=============================================="

# ---- Pre-flight: check for memory consumers inside the container -------------
GPU_PROCS=$(docker exec "${CONTAINER}" nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)
MEM_HOGS=$(docker exec "${CONTAINER}" bash -c 'ps aux --sort=-%mem | awk "NR>1 && \$6 > 1048576 { printf \"  PID %-8s  RSS %-6dMB  %s\n\", \$2, \$6/1024, \$11 }"' 2>/dev/null || true)

if [[ -n "${GPU_PROCS}" || -n "${MEM_HOGS}" ]]; then
  echo "ERROR: Memory-hungry processes detected inside the container."
  echo "On GB10 (unified memory), these steal from the GPU memory pool."
  echo ""
  if [[ -n "${GPU_PROCS}" ]]; then
    echo "GPU compute processes:"
    echo "${GPU_PROCS}" | sed 's/^/  /'
    echo ""
  fi
  if [[ -n "${MEM_HOGS}" ]]; then
    echo "Host memory hogs (>1 GB RSS):"
    echo "${MEM_HOGS}"
    echo ""
  fi
  echo "To kill all: docker exec ${CONTAINER} bash -c 'pkill -9 -f \"ptxas|nvcc|vllm\" 2>/dev/null; true'"
  exit 1
fi

# ---- Start nsys + vllm serve (detached inside container) ---------------------
echo ""
echo "=== Launching vLLM under nsys (detached) ==="

docker exec -d "${CONTAINER}" bash -c "
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export VLLM_FASTSAFETENSORS_NOGDS=1
export VLLM_MXFP4_FUSE_ACTIVATION=${VLLM_MXFP4_FUSE_ACTIVATION:-0}
export FLASHINFER_LOGLEVEL=${FLASHINFER_LOGLEVEL}
mkdir -p /tmp/nsys_profile

nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --output=${PROFILE_OUTPUT} \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=true \
    -- vllm serve $MODEL \
        --host $HOST \
        --port $PORT \
        --served-model-name gpt-oss-120b \
        --quantization mxfp4 \
        --mxfp4-backend CUTLASS \
        --mxfp4-layers moe,qkv,o,lm_head \
        --attention-backend FLASHINFER \
        --kv-cache-dtype fp8 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.60 \
        --max-model-len 131072 \
        --max-num-seqs 2 \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --enable-layerwise-nvtx-tracing \
        --profiler-config.profiler cuda \
        2>&1 | tee /tmp/vllm_server.log
"

# ---- Wait for server ---------------------------------------------------------
echo ""
echo "=== Waiting for server ==="

MAX_WAIT=900
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker exec "$CONTAINER" curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 60)) -eq 0 ]; then
        echo "  Still waiting... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

sleep 2

# ---- Warmup ------------------------------------------------------------------
echo ""
echo "=== Warmup ==="

docker exec "$CONTAINER" curl -sf http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"gpt-oss-120b","prompt":"Hello","max_tokens":16,"temperature":0}' \
    > /dev/null
echo "Warmup done"
sleep 2

# ---- Start profiler, run benchmark, stop profiler ----------------------------
echo ""
echo "=== Starting profiler ==="

docker exec "$CONTAINER" curl -sf -X POST http://localhost:$PORT/start_profile
echo "Profiler started"
sleep 1

echo ""
echo "=== Benchmark (bench.sh) ==="

VLLM_DOCKER_EXEC_FLAGS="-i" "$SCRIPT_DIR/bench.sh"

echo ""
echo "=== Stopping profiler ==="

docker exec "$CONTAINER" curl -sf -X POST http://localhost:$PORT/stop_profile
echo "Profiler stopped"
sleep 2

# ---- Stop server (nsys writes profile on exit) ------------------------------
echo ""
echo "=== Stopping server ==="

"$SCRIPT_DIR/stop.sh" 2>&1 | tail -3

# Wait for nsys to finish writing the profile
sleep 5

echo "=============================================="
echo "Profile saved to: ${PROFILE_OUTPUT}.nsys-rep"
echo "=============================================="
