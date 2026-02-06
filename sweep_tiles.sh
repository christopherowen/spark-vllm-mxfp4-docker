#!/bin/bash
# Sweep vLLM decode benchmarks across MoE tile overrides.
#
# This script:
#  - writes /tmp/flashinfer_moe_tile inside the container (read once per process)
#  - runs ./stop.sh to ensure a clean slate
#  - starts vLLM via ./start.sh (detached)
#  - waits for /v1/models to respond
#  - runs ./bench.sh
#  - stops vLLM via ./stop.sh
#
# Usage:
#   ./sweep_tiles.sh                    # uses default 5 tiles
#   ./sweep_tiles.sh 64x128 32x256 ...  # custom tile list
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
PORT="${VLLM_PORT:-8000}"
SKIP_BENCH="${SWEEP_SKIP_BENCH:-0}"
READY_TIMEOUT_S="${SWEEP_READY_TIMEOUT_S:-600}"

# Validated tiles only - tiles with N < 128 fail with TMA/shared memory errors
DEFAULT_TILES=("128x128" "64x128" "32x128" "16x128" "32x256")
TILES=("$@")
if [[ ${#TILES[@]} -eq 0 ]]; then
  TILES=("${DEFAULT_TILES[@]}")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SWEEP_OUT_DIR:-./results/tile_sweep_${STAMP}}"
mkdir -p "${OUT_DIR}"

echo "Output dir: ${OUT_DIR}"
echo "Tiles: ${TILES[*]}"

echo "Ensuring dev container is up..."
./up.sh >/dev/null

SYSLOG_PATH="${SYSLOG_PATH:-/var/log/syslog}"

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
    # New lines since mark (best-effort; syslog rotation will break this).
    sed -n "$((start_line+1)),\$p" "${SYSLOG_PATH}" > "${out_file}" || true
  else
    : > "${out_file}"
  fi
}

syslog_has_gpu_fault() {
  local file="${1:-}"
  [[ -f "${file}" ]] || return 1
  # Xid 13: illegal instruction; Xid 43 often follows after a fatal fault.
  grep -qiE 'NVRM: Xid|Xid \(PCI:|cudaErrorIllegalInstruction|illegal instruction' "${file}"
}

wait_ready() {
  local timeout_s="${1:-600}"
  local tile="${2:-unknown}"
  local log_path="${3:-}"
  local py_log_path="${4:-}"
  local start_ts
  start_ts="$(date +%s)"
  local last_print_ts="${start_ts}"
  while true; do
    # If the container isn't running, there's no point waiting.
    if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "${CONTAINER}"; then
      echo "Container ${CONTAINER} is not running while waiting (tile=${tile})."
      return 1
    fi

    # Fast-fail heuristics: if EngineCore already crashed, don't wait full timeout.
    if [[ -n "${py_log_path}" ]]; then
      if docker exec "${CONTAINER}" bash -lc "test -f '${py_log_path}' && grep -qiE 'EngineCore failed to start|Engine core initialization failed|torch\\.AcceleratorError: CUDA error: an illegal instruction|cudaErrorIllegalInstruction' '${py_log_path}'"; then
        return 1
      fi
    fi

    # If EngineCore process is missing, we won't become inference-ready.
    if ! docker exec "${CONTAINER}" bash -lc "ps aux | grep -E 'VLLM::EngineCore' | grep -v grep >/dev/null"; then
      # API server may still be starting, so only treat this as fatal after a short grace period.
      if (( $(date +%s) - start_ts > 30 )); then
        return 1
      fi
    fi

    # IMPORTANT: /v1/models can return 200 even when the EngineCore is dead
    # (it is served by the API process). For sweep readiness we need to verify
    # *inference* works, so we probe a tiny chat completion too.
    if docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
import json, urllib.request

base = 'http://127.0.0.1:${PORT}/v1'

# 1) API process is listening
with urllib.request.urlopen(f'{base}/models', timeout=2) as r:
    r.read()

# 2) EngineCore can actually run inference
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
    body = r.read()
    if r.status != 200:
        raise RuntimeError(f'status={r.status} body={body[:200]!r}')

print('READY')
PY" >/dev/null 2>&1; then
      return 0
    fi
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout_s )); then
      return 1
    fi

    # Periodic progress + diagnostics so we don't look "stuck".
    if (( now_ts - last_print_ts >= 15 )); then
      echo "  still waiting... elapsed=$((now_ts-start_ts))s tile=${tile}"
      docker exec "${CONTAINER}" bash -lc "ps aux | grep -E 'vllm serve|EngineCore|APIServer' | grep -v grep || true" \
        2>/dev/null | head -n 5 || true
      if [[ -n "${log_path}" ]]; then
        docker exec "${CONTAINER}" bash -lc "test -f '${log_path}' && tail -n 20 '${log_path}' || true" \
          2>/dev/null || true
      fi
      if [[ -n "${py_log_path}" ]]; then
        docker exec "${CONTAINER}" bash -lc "test -f '${py_log_path}' && tail -n 20 '${py_log_path}' || true" \
          2>/dev/null || true
      fi
      last_print_ts="${now_ts}"
    fi
    sleep 2
  done
}

for tile in "${TILES[@]}"; do
  echo "================================================================================"
  echo "TILE ${tile}"
  echo "================================================================================"

  SYSLOG_L0="$(syslog_mark)"

  # Always stop any existing server first
  VLLM_CONTAINER="${CONTAINER}" ./stop.sh >"${OUT_DIR}/stop_before_${tile}.log" 2>&1 || true

  # Set override inside container BEFORE starting vLLM
  # Note: FLASHINFER_DEBUG_TILE_OVERRIDE=1 must be set in start.sh for this to work
  docker exec "${CONTAINER}" bash -lc "echo '${tile}' > /tmp/flashinfer_moe_tile"

  # Per-tile vLLM log file inside the container (copied out after run).
  CONTAINER_LOG="/tmp/vllm_tile_${tile}.log"
  docker exec "${CONTAINER}" bash -lc "rm -f '${CONTAINER_LOG}' || true"

  # Per-tile Python logger output (dictConfig FileHandler) inside container.
  PY_LOG="/tmp/vllm_python_${tile}.log"
  LOG_CFG="/tmp/vllm_logging_${tile}.json"
  docker exec "${CONTAINER}" bash -lc "rm -f '${PY_LOG}' '${LOG_CFG}' || true"
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

  # Start vLLM detached (no TTY). Logs will show up in container process output.
  VLLM_CONTAINER="${CONTAINER}" \
  VLLM_PORT="${PORT}" \
  VLLM_LOG_FILE="${CONTAINER_LOG}" \
  VLLM_CONFIGURE_LOGGING="1" \
  VLLM_LOGGING_CONFIG_PATH="${LOG_CFG}" \
  VLLM_DOCKER_EXEC_FLAGS="-d" \
  FLASHINFER_DEBUG_TILE_OVERRIDE="1" \
    ./start.sh >"${OUT_DIR}/start_${tile}.log" 2>&1

  echo "Waiting for server readiness..."
  if ! wait_ready "${READY_TIMEOUT_S}" "${tile}" "${CONTAINER_LOG}" "${PY_LOG}"; then
    echo "Server did not become ready for tile ${tile}. Capturing recent status and continuing."
    docker exec "${CONTAINER}" bash -lc "ps aux | grep -E 'vllm serve|EngineCore|APIServer' | grep -v grep || true" \
      >"${OUT_DIR}/ps_${tile}.log" 2>&1 || true
    docker cp "${CONTAINER}:${CONTAINER_LOG}" "${OUT_DIR}/vllm_${tile}.log" >/dev/null 2>&1 || true
    docker cp "${CONTAINER}:${PY_LOG}" "${OUT_DIR}/vllm_python_${tile}.log" >/dev/null 2>&1 || true
    ./stop.sh >"${OUT_DIR}/stop_after_fail_${tile}.log" 2>&1 || true

    syslog_delta "${SYSLOG_L0}" "${OUT_DIR}/syslog_${tile}.log"
    if syslog_has_gpu_fault "${OUT_DIR}/syslog_${tile}.log"; then
      echo "Detected GPU fault (Xid/illegal instruction) during tile ${tile}. Aborting sweep."
      exit 3
    fi
    continue
  fi

  # Extra safety: confirm readiness from the same context as bench (inside container),
  # then give the server a brief moment to settle.
  echo "Server is up for tile ${tile}."
  docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
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
PY" >/dev/null
  sleep 2

  # Run benchmark and capture output
  if [[ "${SKIP_BENCH}" != "1" ]]; then
    VLLM_CONTAINER="${CONTAINER}" \
    VLLM_DOCKER_EXEC_FLAGS="-i" \
      ./bench.sh >"${OUT_DIR}/bench_${tile}.log" 2>&1 || true
  else
    echo "SWEEP_SKIP_BENCH=1 set; skipping bench for tile ${tile}."
  fi

  # Copy vLLM logs out for later inspection
  docker cp "${CONTAINER}:${CONTAINER_LOG}" "${OUT_DIR}/vllm_${tile}.log" >/dev/null 2>&1 || true
  docker cp "${CONTAINER}:${PY_LOG}" "${OUT_DIR}/vllm_python_${tile}.log" >/dev/null 2>&1 || true

  # Stop server between tiles
  VLLM_CONTAINER="${CONTAINER}" ./stop.sh >"${OUT_DIR}/stop_after_${tile}.log" 2>&1 || true

  # Capture syslog delta for this tile and abort on GPU faults.
  syslog_delta "${SYSLOG_L0}" "${OUT_DIR}/syslog_${tile}.log"
  if syslog_has_gpu_fault "${OUT_DIR}/syslog_${tile}.log"; then
    echo "Detected GPU fault (Xid/illegal instruction) during tile ${tile}. Aborting sweep."
    exit 3
  fi

  # Also abort if vLLM reported EngineCore death (even if syslog is missing).
  if grep -qi "Engine core proc .* died unexpectedly" "${OUT_DIR}/vllm_python_${tile}.log" 2>/dev/null; then
    echo "Detected EngineCore death during tile ${tile}. Aborting sweep."
    exit 4
  fi
done

echo "Done. Results in ${OUT_DIR}"
