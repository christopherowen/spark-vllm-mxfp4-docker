#!/bin/bash
# Decode-focused benchmark for MoE tile evaluation.
# Tests small batch generation (decode phase) where swap tiles may help.
set -e

DOCKER_EXEC_FLAGS="${VLLM_DOCKER_EXEC_FLAGS:--it}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
RUNS="${BENCH_RUNS:-10}"

echo "=============================================="
echo "Decode Benchmark"
echo "=============================================="
echo "Focus: Token generation (decode phase)"
echo "Expected tile: (64,128) or swap tiles like (32,128)"
echo "=============================================="

docker exec ${DOCKER_EXEC_FLAGS} "${CONTAINER}" bash -c "
# See bench.sh for why llama-benchy may return empty results for chat completions.
BACKEND=\"${VLLM_BENCH_BACKEND:-completions}\"

if [[ \"${BACKEND}\" == \"llama-benchy\" ]]; then
  llama-benchy \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --latency-mode api \
    --pp 128 256 512 \
    --tg 32 64 128 256 512 \
    --runs ${RUNS}
else
  BENCH_RUNS=\"${RUNS}\" python3 - <<'PY'
import json
import os
import time
import urllib.request

from transformers import AutoTokenizer

BASE_URL = os.environ.get(\"VLLM_BASE_URL\", \"http://localhost:8000/v1\").rstrip(\"/\")
MODEL = os.environ.get(\"VLLM_MODEL_NAME\", \"gpt-oss-120b\")
TOKENIZER_NAME = os.environ.get(\"VLLM_TOKENIZER\", \"openai/gpt-oss-120b\")
RUNS = int(os.environ.get(\"BENCH_RUNS\", \"10\"))

PPS = [128, 256, 512]
TGS = [32, 64, 128, 256, 512]

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
BASE_TEXT = \"We hold these truths to be self-evident, that all men are created equal. \"
BASE_IDS = tok.encode(BASE_TEXT, add_special_tokens=False)

def make_prompt(target_tokens: int) -> str:
    ids = []
    while len(ids) < target_tokens:
        take = min(len(BASE_IDS), target_tokens - len(ids))
        ids.extend(BASE_IDS[:take])
    return tok.decode(ids, skip_special_tokens=True)

def post_json(url: str, payload: dict, timeout_s: int = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(\"utf-8\"),
        headers={\"Content-Type\": \"application/json\"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read().decode(\"utf-8\", \"replace\")
    return json.loads(body)

def run_one(pp: int, tg: int) -> float:
    payload = {
        \"model\": MODEL,
        \"prompt\": make_prompt(pp),
        \"max_tokens\": tg,
        \"temperature\": 0,
        \"cache_prompt\": False,
    }
    t0 = time.time()
    out = post_json(f\"{BASE_URL}/completions\", payload)
    t1 = time.time()
    if \"error\" in out:
        raise RuntimeError(out[\"error\"])
    usage = out.get(\"usage\") or {}
    ct = usage.get(\"completion_tokens\")
    return ct / max(t1 - t0, 1e-9)

print(f\"Decode benchmark (completions): model={MODEL} base_url={BASE_URL} runs={RUNS}\")
for pp in PPS:
    for tg in TGS:
        vals = [run_one(pp, tg) for _ in range(RUNS)]
        print(f\"pp={pp:4d} tg={tg:4d}  gen_tok/s={sum(vals)/len(vals):8.2f}\")
PY
fi
"
