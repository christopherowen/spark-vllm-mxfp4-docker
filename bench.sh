#!/bin/bash
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
    --tg 32 128 512 \
    --runs 3
"
