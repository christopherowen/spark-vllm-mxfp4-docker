#!/bin/bash
# Benchmark all validated tiles with vLLM and llama-benchy.
# Tests both prefill and decode for each tile.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
PORT="${VLLM_PORT:-8000}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"

# Validated tiles only (N >= 128)
TILES=("128x128" "64x128" "32x128" "16x128" "32x256")

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BENCH_OUT_DIR:-./results/tile_bench_${STAMP}}"
mkdir -p "${OUT_DIR}"

echo "=============================================="
echo "Tile Benchmark Suite"
echo "=============================================="
echo "Output: ${OUT_DIR}"
echo "Tiles: ${TILES[*]}"
echo "=============================================="

# Ensure container is up
./up.sh >/dev/null 2>&1 || true

wait_ready() {
  local timeout_s="${1:-300}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if docker exec "${CONTAINER}" curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      # Also verify inference works
      if docker exec "${CONTAINER}" curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' >/dev/null 2>&1; then
        return 0
      fi
    fi
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout_s )); then
      return 1
    fi
    sleep 5
  done
}

run_benchmark() {
  local tile="$1"
  local out_prefix="$2"
  
  echo "  Running llama-benchy..."
  docker exec -i "${CONTAINER}" bash -c "
    llama-benchy \
      --base-url http://localhost:${PORT}/v1 \
      --model gpt-oss-120b \
      --tokenizer openai/gpt-oss-120b \
      --latency-mode api \
      --pp 512 2048 \
      --tg 32 128 \
      --runs 5
  " > "${out_prefix}_bench.log" 2>&1 || true
  
  # Extract key metrics
  if grep -q "tok/s" "${out_prefix}_bench.log"; then
    echo "  Results:"
    grep -E "^\|.*tok/s" "${out_prefix}_bench.log" | head -10 || true
  else
    echo "  Benchmark may have failed - check ${out_prefix}_bench.log"
  fi
}

for tile in "${TILES[@]}"; do
  echo ""
  echo "=============================================="
  echo "TILE: ${tile}"
  echo "=============================================="
  
  # Stop any running server
  ./stop.sh >/dev/null 2>&1 || true
  sleep 2
  
  # Set tile override
  docker exec "${CONTAINER}" sh -c "echo '${tile}' > /tmp/flashinfer_moe_tile"
  echo "Set tile override: ${tile}"
  
  # Start vLLM with tile override enabled and enforce-eager for stability
  echo "Starting vLLM..."
  docker exec -d "${CONTAINER}" bash -c "
    export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
    export VLLM_FASTSAFETENSORS_NOGDS=1
    export FLASHINFER_DEBUG_TILE_OVERRIDE=1
    nohup vllm serve openai/gpt-oss-120b \
      --host 0.0.0.0 \
      --port ${PORT} \
      --served-model-name gpt-oss-120b \
      --quantization mxfp4 \
      --mxfp4-backend CUTLASS \
      --mxfp4-layers moe,qkv,o,lm_head \
      --attention-backend FLASHINFER \
      --kv-cache-dtype fp8 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.70 \
      --max-model-len 131072 \
      --max-num-seqs 2 \
      --max-num-batched-tokens 8192 \
      --enable-prefix-caching \
      --load-format fastsafetensors \
      --enforce-eager \
      > /tmp/vllm_${tile}.log 2>&1 &
  "
  echo "vLLM started in background"
  
  # Wait for ready
  echo "Waiting for server..."
  if ! wait_ready "${READY_TIMEOUT_S}"; then
    echo "ERROR: Server failed to start for tile ${tile}"
    docker exec "${CONTAINER}" sh -c "cat /tmp/vllm*.log 2>/dev/null | tail -50" > "${OUT_DIR}/${tile}_error.log" 2>&1 || true
    continue
  fi
  echo "Server ready!"
  
  # Run benchmark
  run_benchmark "${tile}" "${OUT_DIR}/${tile}"
  
  # Stop server
  ./stop.sh >/dev/null 2>&1 || true
done

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="

for tile in "${TILES[@]}"; do
  log="${OUT_DIR}/${tile}_bench.log"
  if [[ -f "${log}" ]]; then
    echo ""
    echo "--- ${tile} ---"
    # Extract decode performance (tg column with highest values)
    grep -E "^\|.*\|.*\|.*tok/s" "${log}" 2>/dev/null | tail -5 || echo "No results"
  fi
done

echo ""
echo "Full results in: ${OUT_DIR}"
