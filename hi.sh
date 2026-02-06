#!/bin/bash
set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
MODEL="${VLLM_MODEL:-gpt-oss-120b}"
MAX_TOKENS="${VLLM_MAX_TOKENS:-64}"
TIMEOUT_SECS="${VLLM_TIMEOUT_SECS:-30}"

curl -sS --fail-with-body \
  --connect-timeout 2 \
  --max-time "$TIMEOUT_SECS" \
  "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer none" \
  -d '{
    "model": "'$MODEL'",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "max_tokens": '$MAX_TOKENS',
    "temperature": 0,
    "stream": false
  }'
