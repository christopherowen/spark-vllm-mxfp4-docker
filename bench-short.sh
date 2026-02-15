#!/bin/bash
# Short benchmark for profiling: 1 run, 2 shapes (pp2048+tg32, pp8192+tg32)
set -e

DOCKER_EXEC_FLAGS="${VLLM_DOCKER_EXEC_FLAGS:--it}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"

docker exec ${DOCKER_EXEC_FLAGS} "${CONTAINER}" bash -c "
llama-benchy \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --latency-mode api \
    --pp 2048 8192 \
    --tg 32 \
    --runs 1
"
