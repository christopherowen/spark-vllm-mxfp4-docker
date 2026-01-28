#!/bin/bash
docker exec -it vllm-dev bash -c "
llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --tokenizer openai/gpt-oss-120b \
  --pp 2048 \
  --tg 32 128 \
  --runs 5
"
