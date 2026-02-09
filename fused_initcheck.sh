#!/bin/bash
# =============================================================================
# Run fused-kernel compute-sanitizer initcheck to detect reads of
# uninitialised shared/global memory.
#
# This is the highest-priority diagnostic for the all-zero output bug:
# if consumer warps read SMEM before the TMA producer has written data,
# initcheck will report the exact location.
#
# Usage:
#   ./fused_initcheck.sh
#   ./fused_initcheck.sh /workspace/scripts/debug/repro_fused_illegal_instruction.py
#   ./fused_initcheck.sh scripts/debug/repro_fused_illegal_instruction.py
#   ./fused_initcheck.sh <repro_script> <output_dir>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"
REPRO_SCRIPT="${1:-/workspace/scripts/debug/repro_fused_illegal_instruction.py}"
OUT_DIR="${2:-$SCRIPT_DIR/results/initcheck_$(date +%Y%m%d_%H%M%S)}"
HOST_LOG="$OUT_DIR/fused_initcheck.log"
INITCHECK_PREWARM="${INITCHECK_PREWARM:-1}"

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
echo "Fused Kernel Initcheck"
echo "=============================================="
echo "Container:    $CONTAINER"
echo "Repro script: $REPRO_SCRIPT"
echo "Output dir:   $OUT_DIR"
echo "Prewarm JIT:  $INITCHECK_PREWARM"
echo "=============================================="

if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "${CONTAINER}"; then
    echo "ERROR: Container ${CONTAINER} is not running"
    echo "Start it with: docker compose -f docker-compose.dev.yml up -d"
    exit 1
fi

if ! docker exec "$CONTAINER" bash -c "command -v compute-sanitizer >/dev/null"; then
    echo "ERROR: compute-sanitizer not found in ${CONTAINER}"
    exit 1
fi

if ! docker exec "$CONTAINER" bash -c "test -f '$REPRO_SCRIPT'"; then
    echo "ERROR: Repro script not found in container: $REPRO_SCRIPT"
    exit 1
fi

# Optional prewarm: compile JIT artifacts outside compute-sanitizer so
# initcheck focuses on runtime reads rather than slow nvcc/ptxas paths.
if [[ "$INITCHECK_PREWARM" == "1" ]]; then
    echo "Prewarming JIT outside initcheck..."
    set +e
    docker exec \
        -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm:/workspace/scripts \
        "$CONTAINER" \
        python3 "$REPRO_SCRIPT" >"$OUT_DIR/fused_initcheck_warmup.log" 2>&1
    WARMUP_RC=$?
    set -e
    echo "Prewarm exit code: $WARMUP_RC (log: $OUT_DIR/fused_initcheck_warmup.log)"
fi

# Resolve the real python3 binary.  compute-sanitizer does not search PATH;
# it needs an absolute path to the target executable.
PYTHON3="$(docker exec "$CONTAINER" readlink -f /usr/bin/python3)"
if [[ -z "$PYTHON3" ]]; then
    echo "ERROR: Could not resolve python3 path in ${CONTAINER}"
    exit 1
fi
echo "Resolved python3: $PYTHON3"

# Run initcheck.  Note: --error-exitcode is not supported by all
# compute-sanitizer versions, so we parse the ERROR SUMMARY line instead.
set +e
docker exec \
    -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm:/workspace/scripts \
    -e CUDA_LAUNCH_BLOCKING=1 \
    "$CONTAINER" \
    compute-sanitizer --tool initcheck \
        --target-processes all \
        "$PYTHON3" "$REPRO_SCRIPT" 2>&1 | tee "$HOST_LOG"
RC=${PIPESTATUS[0]}
set -e

# Flag uninitialized-read signatures as failure even if the script exits 0.
if grep -Eqi "Uninitialized" "$HOST_LOG"; then
    echo ""
    echo "*** initcheck found uninitialized memory reads ***"
    echo "These likely explain the all-zero output: consumer warps are reading"
    echo "SMEM before the TMA producer has written data."
    echo ""
    echo "--- Summary of uninitialized reads ---"
    grep -c "Uninitialized" "$HOST_LOG" | xargs -I{} echo "  Total reports: {}"
    echo ""
    echo "--- First 10 reports ---"
    grep "Uninitialized" "$HOST_LOG" | head -10
    RC=99
fi

if grep -q "ERROR SUMMARY: [1-9]" "$HOST_LOG"; then
    NERR=$(grep "ERROR SUMMARY:" "$HOST_LOG" | tail -1 | grep -oP '\d+(?= error)')
    echo ""
    echo "Initcheck found $NERR error(s). See: $HOST_LOG"
else
    echo ""
    echo "Initcheck completed (0 errors). See: $HOST_LOG"
fi

echo "compute-sanitizer exit code: $RC"
exit "$RC"
