#!/bin/bash
# Run benchmark for a single tile. Usage: ./run_tile_benchmark.sh 128x128
set -e

TILE="${1:-128x128}"
OUT_DIR="./results/tile_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

echo "=============================================="
echo "Benchmarking tile: ${TILE}"
echo "Output: ${OUT_DIR}"
echo "=============================================="

# Set tile override
docker exec vllm-dev sh -c "echo '${TILE}' > /tmp/flashinfer_moe_tile"

# Start vLLM in foreground (this script blocks until vLLM exits)
echo "Starting vLLM..."
docker exec -i vllm-dev bash -c "
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export VLLM_FASTSAFETENSORS_NOGDS=1
export FLASHINFER_DEBUG_TILE_OVERRIDE=1

vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
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
  2>&1 &

# Wait for server to be ready
for i in {1..120}; do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo 'Server ready!'
    break
  fi
  sleep 5
done

# Run benchmark
echo 'Running benchmark...'
llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --tokenizer openai/gpt-oss-120b \
  --latency-mode api \
  --pp 512 2048 \
  --tg 32 128 \
  --runs 5

# Kill vLLM
pkill -f 'vllm serve' || true
" 2>&1 | tee "${OUT_DIR}/${TILE}.log"

echo "Done. Results in ${OUT_DIR}/${TILE}.log"
