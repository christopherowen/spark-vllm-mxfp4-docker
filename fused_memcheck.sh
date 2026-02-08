#!/bin/bash
# =============================================================================
# Run fused-kernel compute-sanitizer memcheck independently of profiling.
#
# Usage:
#   ./fused_memcheck.sh
#   ./fused_memcheck.sh /workspace/scripts/debug/repro_fused_illegal_instruction.py
#   ./fused_memcheck.sh scripts/debug/repro_fused_illegal_instruction.py
#   ./fused_memcheck.sh <repro_script> <output_dir>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
REPRO_SCRIPT="${1:-/workspace/scripts/debug/repro_fused_illegal_instruction.py}"
OUT_DIR="${2:-$SCRIPT_DIR/results/memcheck_$(date +%Y%m%d_%H%M%S)}"
HOST_LOG="$OUT_DIR/fused_memcheck.log"
CONTAINER_LOG="/tmp/fused_memcheck.log"
CONTAINER_WARMUP_LOG="/tmp/fused_memcheck_warmup.log"
MEMCHECK_PREWARM="${MEMCHECK_PREWARM:-1}"

mkdir -p "$OUT_DIR"

# Allow host-relative script paths and map them to container mount paths.
if [[ "$REPRO_SCRIPT" != /workspace/* ]]; then
    case "$REPRO_SCRIPT" in
        scripts/*)
            REPRO_SCRIPT="/workspace/${REPRO_SCRIPT}"
            ;;
        "$SCRIPT_DIR"/scripts/*)
            REPRO_SCRIPT="/workspace/scripts/${REPRO_SCRIPT#"$SCRIPT_DIR"/scripts/}"
            ;;
    esac
fi

echo "=============================================="
echo "Fused Kernel Memcheck"
echo "=============================================="
echo "Container:    $CONTAINER"
echo "Repro script: $REPRO_SCRIPT"
echo "Output dir:   $OUT_DIR"
echo "Prewarm JIT:  $MEMCHECK_PREWARM"
echo "=============================================="

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "${CONTAINER}"; then
    echo "ERROR: Container ${CONTAINER} is not running"
    echo "Start it with: docker compose -f docker-compose.dev.yml up -d"
    exit 1
fi

# Validate prerequisites inside the container before running memcheck.
if ! docker exec "$CONTAINER" bash -lc "command -v compute-sanitizer >/dev/null"; then
    echo "ERROR: compute-sanitizer not found in ${CONTAINER}"
    exit 1
fi

if ! docker exec "$CONTAINER" bash -lc "test -f \"$REPRO_SCRIPT\""; then
    echo "ERROR: Repro script not found in container: $REPRO_SCRIPT"
    exit 1
fi

# Optional prewarm: compile JIT artifacts outside compute-sanitizer so memcheck
# focuses on runtime faults rather than slow nvcc/ptxas paths.
if [[ "$MEMCHECK_PREWARM" == "1" ]]; then
    echo "Prewarming JIT outside memcheck..."
    set +e
    docker exec "$CONTAINER" bash -lc "
set -euo pipefail
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm:/workspace/scripts
rm -f \"$CONTAINER_WARMUP_LOG\"
python3 \"$REPRO_SCRIPT\" > \"$CONTAINER_WARMUP_LOG\" 2>&1
" >/dev/null 2>&1
    WARMUP_RC=$?
    set -e
    docker cp "$CONTAINER:${CONTAINER_WARMUP_LOG}" "$OUT_DIR/fused_memcheck_warmup.log" 2>/dev/null || true
    echo "Prewarm exit code: $WARMUP_RC (log: $OUT_DIR/fused_memcheck_warmup.log)"
fi

set +e
docker exec "$CONTAINER" bash -lc "
set -euo pipefail
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm:/workspace/scripts
rm -f \"$CONTAINER_LOG\"
set +e
compute-sanitizer --tool memcheck --target-processes all --error-exitcode=99 \
    python3 \"$REPRO_SCRIPT\" 2>&1 | tee \"$CONTAINER_LOG\"
status=\${PIPESTATUS[0]}
set -e
echo \"__MEMCHECK_EXIT_CODE__=\${status}\"
exit \"\${status}\"
" 2>&1 | tee "$HOST_LOG"
RC=${PIPESTATUS[0]}
set -e

docker cp "$CONTAINER:${CONTAINER_LOG}" "$OUT_DIR/fused_memcheck_container.log" 2>/dev/null || true

# Treat CUDA illegal instruction signatures as failure even if the repro script
# catches exceptions and exits 0.
if grep -Eqi "illegal instruction|cudaerrorillegalinstruction" "$HOST_LOG"; then
    RC=99
fi

if grep -q "ERROR SUMMARY: [1-9]" "$HOST_LOG"; then
    echo "Memcheck found errors. See: $HOST_LOG"
else
    echo "Memcheck completed. See: $HOST_LOG"
fi

echo "compute-sanitizer exit code: $RC"
exit "$RC"
