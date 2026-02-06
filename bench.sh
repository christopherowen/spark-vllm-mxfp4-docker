#!/bin/bash
set -e

# Allow non-interactive runs (useful for automation).
# Examples:
#   VLLM_DOCKER_EXEC_FLAGS="-i" ./bench.sh
DOCKER_EXEC_FLAGS="${VLLM_DOCKER_EXEC_FLAGS:--it}"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"

docker exec ${DOCKER_EXEC_FLAGS} "${CONTAINER}" bash -c "
# NOTE: vLLM (0.14.0rc1.dev*) currently returns chat completions with
# message.content = null for gpt-oss-120b, which breaks llama-benchy result parsing
# (it reports: 'No results collected').
#
# Workaround: benchmark via /v1/completions and use usage.completion_tokens.
#
# You can force llama-benchy by setting:
#   VLLM_BENCH_BACKEND=llama-benchy ./bench.sh
BACKEND=\"${VLLM_BENCH_BACKEND:-completions}\"

if [[ \"${BACKEND}\" == \"llama-benchy\" ]]; then
  llama-benchy \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --latency-mode api \
    --pp 2048 4096 8192 16384 \
    --tg 32 128 256 512 1024 \
    --runs 10
else
  python3 - <<'PY'
import json
import os
import time
import urllib.request

try:
    from transformers import AutoTokenizer
except Exception as e:
    raise SystemExit(f\"transformers is required in the container for token-accurate prompts: {e}\")

BASE_URL = os.environ.get(\"VLLM_BASE_URL\", \"http://localhost:8000/v1\").rstrip(\"/\")
MODEL = os.environ.get(\"VLLM_MODEL_NAME\", \"gpt-oss-120b\")
TOKENIZER_NAME = os.environ.get(\"VLLM_TOKENIZER\", \"openai/gpt-oss-120b\")
RUNS = int(os.environ.get(\"BENCH_RUNS\", \"10\"))

PPS = [2048, 4096, 8192, 16384]
TGS = [32, 128, 256, 512, 1024]

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

BASE_TEXT = \"\"\"Sherlock Holmes was always the woman. I have seldom heard him mention her
under any other name. In his eyes she eclipses and predominates the whole of her sex.
It was not that he felt any emotion akin to love for Irene Adler.
\"\"\"
BASE_IDS = tok.encode(BASE_TEXT, add_special_tokens=False)
if len(BASE_IDS) < 32:
    raise SystemExit(\"tokenizer produced too few base tokens; cannot build prompts reliably\")

def make_prompt(target_tokens: int) -> str:
    # Build exactly target_tokens (by tokenizer tokens), then detokenize.
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

def run_one(pp: int, tg: int, run_idx: int) -> dict:
    prompt = make_prompt(pp)
    payload = {
        \"model\": MODEL,
        \"prompt\": prompt,
        \"max_tokens\": tg,
        \"temperature\": 0,
        # vLLM extension: disable prompt caching (keeps runs comparable)
        \"cache_prompt\": False,
    }
    t0 = time.time()
    out = post_json(f\"{BASE_URL}/completions\", payload)
    t1 = time.time()

    if \"error\" in out:
        raise RuntimeError(out[\"error\"])

    usage = out.get(\"usage\") or {}
    return {
        \"elapsed_s\": (t1 - t0),
        \"prompt_tokens\": usage.get(\"prompt_tokens\"),
        \"completion_tokens\": usage.get(\"completion_tokens\"),
        \"total_tokens\": usage.get(\"total_tokens\"),
    }

print(f\"Benchmarking model: {MODEL} at {BASE_URL}\")
print(f\"Tokenizer: {TOKENIZER_NAME}\")
print(f\"Runs per test: {RUNS}\")
print()

rows = []
for pp in PPS:
    for tg in TGS:
        per = []
        for r in range(RUNS):
            m = run_one(pp, tg, r)
            per.append(m)
        # Aggregate
        gen_tok = [x[\"completion_tokens\"] for x in per if x[\"completion_tokens\"] is not None]
        el = [x[\"elapsed_s\"] for x in per]
        # Use reported completion tokens for throughput (robust vs text decoding)
        tok_s = [gt / e for gt, e in zip(gen_tok, el)] if gen_tok else []
        rows.append((pp, tg, sum(tok_s) / len(tok_s) if tok_s else float(\"nan\")))
        print(f\"pp={pp:5d} tg={tg:4d}  gen_tok/s={rows[-1][2]:8.2f}\")

print()
print(\"Summary (mean gen tok/s):\")
for pp, tg, v in rows:
    print(f\"  pp={pp:5d} tg={tg:4d}  {v:8.2f}\")
PY
fi
"
