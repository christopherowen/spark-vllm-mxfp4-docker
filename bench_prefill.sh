#!/bin/bash
# Prefill-focused benchmark for MoE tile evaluation.
# Tests large prompt processing (prefill phase) where native tiles excel.
set -e

DOCKER_EXEC_FLAGS="${VLLM_DOCKER_EXEC_FLAGS:--it}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
RUNS="${BENCH_RUNS:-10}"

echo "=============================================="
echo "Prefill Benchmark"
echo "=============================================="
echo "Focus: Prompt processing (prefill phase)"
echo "Expected tile: (128,128) for large batches"
echo "=============================================="

docker exec ${DOCKER_EXEC_FLAGS} "${CONTAINER}" bash -c "
llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --tokenizer openai/gpt-oss-120b \
  --latency-mode api \
  --pp 512 1024 2048 4096 8192 \
  --tg 1 8 16 \
  --runs ${RUNS}
"
