#!/bin/bash
# Run one tile end-to-end with logs + syslog capture.
#
# This is the “don’t fight the sweep” approach: run tiles one at a time in a
# clean process, collect artifacts, and stop immediately if the GPU faults.
#
# Usage:
#   ./run_one_tile.sh 64x128
#   ./run_one_tile.sh 32x256
#
set -euo pipefail

TILE="${1:-}"
if [[ -z "${TILE}" ]]; then
  echo "Usage: $0 <TILE_MxTILE_N> (e.g. 64x128)"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
PORT="${VLLM_PORT:-8000}"
SYSLOG_PATH="${SYSLOG_PATH:-/var/log/syslog}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-./results/tile_${TILE}_${STAMP}}"
mkdir -p "${OUT_DIR}"

syslog_mark() {
  if [[ -f "${SYSLOG_PATH}" ]]; then
    wc -l "${SYSLOG_PATH}" | awk '{print $1}'
  else
    echo 0
  fi
}

syslog_delta() {
  local start_line="${1:-0}"
  local out_file="${2:-/dev/null}"
  if [[ -f "${SYSLOG_PATH}" ]] && [[ "${start_line}" -gt 0 ]]; then
    sed -n "$((start_line+1)),\$p" "${SYSLOG_PATH}" > "${out_file}" || true
  else
    : > "${out_file}"
  fi
}

syslog_has_gpu_fault() {
  local file="${1:-}"
  [[ -f "${file}" ]] || return 1
  grep -qiE 'NVRM: Xid|Xid \(PCI:|cudaErrorIllegalInstruction|illegal instruction' "${file}"
}

wait_ready_chat() {
  local timeout_s="${1:-900}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
import json, urllib.request
base = 'http://127.0.0.1:${PORT}/v1'
payload = {
  'model': 'gpt-oss-120b',
  'messages': [{'role': 'user', 'content': 'ping'}],
  'max_tokens': 1,
  'temperature': 0,
}
req = urllib.request.Request(
    f'{base}/chat/completions',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=10) as r:
    r.read()
print('OK')
PY" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - start_ts > timeout_s )); then
      return 1
    fi
    sleep 2
  done
}

echo "OUT_DIR=${OUT_DIR}"
echo "TILE=${TILE}"

./up.sh >/dev/null

SYSLOG_L0="$(syslog_mark)"

VLLM_CONTAINER="${CONTAINER}" ./stop.sh >"${OUT_DIR}/stop_before.log" 2>&1 || true

docker exec "${CONTAINER}" bash -lc "echo '${TILE}' > /tmp/flashinfer_moe_tile"

CONTAINER_LOG="/tmp/vllm_tile_${TILE}.log"
PY_LOG="/tmp/vllm_python_${TILE}.log"
LOG_CFG="/tmp/vllm_logging_${TILE}.json"
docker exec "${CONTAINER}" bash -lc "rm -f '${CONTAINER_LOG}' '${PY_LOG}' '${LOG_CFG}' || true"

docker exec "${CONTAINER}" bash -lc "cat > '${LOG_CFG}' <<'JSON'
{
  \"version\": 1,
  \"disable_existing_loggers\": false,
  \"formatters\": {
    \"standard\": {
      \"format\": \"%(levelname)s %(asctime)s [%(processName)s %(process)d] %(name)s:%(lineno)d - %(message)s\"
    }
  },
  \"handlers\": {
    \"console\": {
      \"class\": \"logging.StreamHandler\",
      \"level\": \"INFO\",
      \"formatter\": \"standard\",
      \"stream\": \"ext://sys.stdout\"
    },
    \"file\": {
      \"class\": \"logging.FileHandler\",
      \"level\": \"INFO\",
      \"formatter\": \"standard\",
      \"filename\": \"${PY_LOG}\",
      \"mode\": \"a\"
    }
  },
  \"root\": {
    \"level\": \"INFO\",
    \"handlers\": [\"console\", \"file\"]
  }
}
JSON"

VLLM_CONTAINER="${CONTAINER}" \
VLLM_PORT="${PORT}" \
VLLM_LOG_FILE="${CONTAINER_LOG}" \
VLLM_CONFIGURE_LOGGING="1" \
VLLM_LOGGING_CONFIG_PATH="${LOG_CFG}" \
VLLM_DOCKER_EXEC_FLAGS="-d" \
  ./start.sh >"${OUT_DIR}/start.log" 2>&1

echo "Waiting for inference-ready..."
if ! wait_ready_chat 1200; then
  echo "Server never became inference-ready."
fi

VLLM_CONTAINER="${CONTAINER}" VLLM_DOCKER_EXEC_FLAGS="-i" ./bench.sh >"${OUT_DIR}/bench.log" 2>&1 || true

docker cp "${CONTAINER}:${CONTAINER_LOG}" "${OUT_DIR}/vllm.log" >/dev/null 2>&1 || true
docker cp "${CONTAINER}:${PY_LOG}" "${OUT_DIR}/vllm_python.log" >/dev/null 2>&1 || true

VLLM_CONTAINER="${CONTAINER}" ./stop.sh >"${OUT_DIR}/stop_after.log" 2>&1 || true

syslog_delta "${SYSLOG_L0}" "${OUT_DIR}/syslog.log"
if syslog_has_gpu_fault "${OUT_DIR}/syslog.log"; then
  echo "GPU fault detected during this tile run. See ${OUT_DIR}/syslog.log"
  exit 3
fi

echo "Done: ${OUT_DIR}"
