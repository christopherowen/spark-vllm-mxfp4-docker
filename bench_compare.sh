#!/bin/bash
# =============================================================================
# Three-Way Benchmark Comparison: golden vs current-unfused vs current-fused
#
# For each variant:
#   1. Clean benchmark (start.sh + bench.sh) — warms JIT cache, gets throughput
#   2. Profiled benchmark (start-profile.sh) — nsys kernel-level trace
#
# The clean run MUST come first: it warms the FlashInfer JIT cache. Without a
# warm cache, nsys's --trace-fork-before-exec gets confused by the nvcc/ptxas
# forks during JIT compilation and fails to capture CUDA kernel data.
#
# Golden SHAs come from Dockerfile (pinned, known-good configuration).
# Current code is mxfp4_v4 branch (all work committed).
#
# Usage:
#   ./bench_compare.sh              # Run all 3 variants
#   ./bench_compare.sh golden       # Run only golden
#   ./bench_compare.sh unfused      # Run only current unfused
#   ./bench_compare.sh fused        # Run only current fused
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Golden SHAs (from Dockerfile)
# ---------------------------------------------------------------------------
GOLDEN_FI_SHA="f349e52496a72a00d8c4ac02c7a1e38523ff7194"
GOLDEN_VLLM_SHA="045293d82b832229560ac4a13152a095af603b6e"
GOLDEN_CUTLASS_SHA="11af7f02ab52c9130e422eeb4b44042fbd60c083"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FI_REPO="$HOME/projects/flashinfer"
VLLM_REPO="$HOME/projects/vllm"
CONTAINER="${VLLM_CONTAINER:-vllm-dev}"

TS=$(date +%Y%m%d_%H%M)
RESULTS="$SCRIPT_DIR/results/compare_${TS}"

# Which variants to run (default: all)
VARIANTS="${1:-all}"

# Current branch for flashinfer/vllm/cutlass repos
CURRENT_BRANCH="mxfp4_v4"

# =============================================================================
# Helper functions
# =============================================================================

log() { echo "[$(date +%H:%M:%S)] $*"; }

# GPU temperature threshold (Celsius) — benchmark won't start until GPU is at or below this.
GPU_TEMP_THRESHOLD="${GPU_TEMP_THRESHOLD:-45}"

gpu_temp() {
    docker exec "$CONTAINER" nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' '
}

record_temp() {
    local label="$1"    # e.g., "pre_clean", "post_clean", "pre_profiled", "post_profiled"
    local dir="$2"
    local temp
    temp=$(gpu_temp)
    log "  GPU temp ($label): ${temp}C"
    echo "$label: ${temp}C  $(date -Iseconds)" >> "$dir/gpu_temps.txt"
}

wait_for_cooldown() {
    local temp
    temp=$(gpu_temp)
    if [ "$temp" -le "$GPU_TEMP_THRESHOLD" ] 2>/dev/null; then
        log "  GPU temp ${temp}C <= ${GPU_TEMP_THRESHOLD}C threshold, proceeding"
        return 0
    fi
    log "  GPU temp ${temp}C > ${GPU_TEMP_THRESHOLD}C, waiting for cooldown..."
    local waited=0
    while [ $waited -lt 600 ]; do
        sleep 10
        waited=$((waited + 10))
        temp=$(gpu_temp)
        if [ "$temp" -le "$GPU_TEMP_THRESHOLD" ] 2>/dev/null; then
            log "  GPU cooled to ${temp}C after ${waited}s, proceeding"
            return 0
        fi
        if [ $((waited % 60)) -eq 0 ]; then
            log "  Still cooling... ${temp}C (${waited}s)"
        fi
    done
    log "  WARNING: GPU still ${temp}C after 600s, proceeding anyway"
}

stop_server() {
    log "Stopping server..."
    "$SCRIPT_DIR/stop.sh" 2>&1 | tail -3

    # Wait until GPU processes are actually gone
    local attempts=0
    while [ $attempts -lt 12 ]; do
        local gpu_procs
        gpu_procs=$(docker exec "$CONTAINER" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        local mem_hogs
        mem_hogs=$(docker exec "$CONTAINER" bash -c 'ps aux --sort=-%mem | awk "NR>1 && \$6 > 1048576 { print \$2 }"' 2>/dev/null || true)
        if [[ -z "$gpu_procs" && -z "$mem_hogs" ]]; then
            break
        fi
        attempts=$((attempts + 1))
        if [ $attempts -eq 1 ]; then
            log "  Waiting for processes to fully exit..."
        fi
        if [ $attempts -ge 4 ]; then
            docker exec "$CONTAINER" bash -c 'pkill -9 -f "vllm|python.*vllm" 2>/dev/null; true'
        fi
        sleep 5
    done
    if [ $attempts -ge 12 ]; then
        log "  WARNING: Processes may still be running after 60s"
    fi
}

clear_jit_cache() {
    log "Clearing FlashInfer JIT cache..."
    docker exec "$CONTAINER" bash -c \
        'rm -rf /root/.cache/flashinfer/*/121f/cached_ops/* \
                /root/.cache/flashinfer/*/121a/cached_ops/* \
                2>/dev/null; echo "  cleared"'
}

clear_compile_caches() {
    log "Clearing vLLM/Torch/Triton compile caches..."
    docker exec "$CONTAINER" bash -c \
        'rm -rf /root/.cache/vllm/* \
                /root/.cache/torch/inductor/* \
                /root/.cache/torch_extensions/* \
                /root/.cache/triton/* \
                2>/dev/null; echo "  cleared"'
}

wait_for_server() {
    local max_wait=900
    local waited=0
    log "Waiting for server health..."
    while [ $waited -lt $max_wait ]; do
        if docker exec "$CONTAINER" curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log "Server ready after ${waited}s"
            return 0
        fi
        # Check if server process died (no vllm or nsys process running)
        if [ $waited -ge 30 ]; then
            local procs
            procs=$(docker exec "$CONTAINER" pgrep -f "vllm|nsys" 2>/dev/null || true)
            if [ -z "$procs" ]; then
                log "ERROR: Server process died after ${waited}s"
                log "Last 30 lines of server log:"
                docker exec "$CONTAINER" tail -30 /tmp/vllm_server.log 2>/dev/null || true
                return 1
            fi
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            log "  Still waiting... (${waited}s)"
        fi
    done
    log "ERROR: Server did not become ready within ${max_wait}s"
    log "Last 30 lines of server log:"
    docker exec "$CONTAINER" tail -30 /tmp/vllm_server.log 2>/dev/null || true
    return 1
}

save_metadata() {
    local variant="$1"
    local fuse_val="$2"
    local dir="$RESULTS/$variant"

    local fi_sha; fi_sha=$(cd "$FI_REPO" && git rev-parse HEAD)
    local fi_branch; fi_branch=$(cd "$FI_REPO" && git branch --show-current 2>/dev/null || echo "detached")
    local vllm_sha; vllm_sha=$(cd "$VLLM_REPO" && git rev-parse HEAD)
    local vllm_branch; vllm_branch=$(cd "$VLLM_REPO" && git branch --show-current 2>/dev/null || echo "detached")
    local cutlass_sha; cutlass_sha=$(cd "$FI_REPO/3rdparty/cutlass" && git rev-parse HEAD)

    cat > "$dir/metadata.txt" <<EOF
variant: $variant
timestamp: $(date -Iseconds)
fuse_activation: $fuse_val

git:
  flashinfer: $fi_sha ($fi_branch)
  vllm:       $vllm_sha ($vllm_branch)
  cutlass:    $cutlass_sha

server_config:
  model: openai/gpt-oss-120b
  quantization: mxfp4
  mxfp4_backend: CUTLASS
  mxfp4_layers: moe,qkv,o,lm_head
  attention_backend: FLASHINFER
  kv_cache_dtype: fp8
  gpu_memory_utilization: 0.70 (clean) / 0.60 (profiled)
  max_model_len: 131072
  max_num_seqs: 2
  max_num_batched_tokens: 8192
EOF
    log "Metadata saved to $dir/metadata.txt"
}

# =============================================================================
# Golden checkout / restore
# =============================================================================

checkout_golden() {
    log "=== Checking out golden SHAs ==="

    (cd "$FI_REPO" && git checkout -q "$GOLDEN_FI_SHA")
    (cd "$FI_REPO/3rdparty/cutlass" && git checkout -q "$GOLDEN_CUTLASS_SHA")
    (cd "$VLLM_REPO" && git checkout -q "$GOLDEN_VLLM_SHA")

    log "Golden checkout complete"
    log "  FlashInfer: $(cd "$FI_REPO" && git rev-parse --short HEAD)"
    log "  vLLM:       $(cd "$VLLM_REPO" && git rev-parse --short HEAD)"
    log "  CUTLASS:    $(cd "$FI_REPO/3rdparty/cutlass" && git rev-parse --short HEAD)"
}

restore_current() {
    log "=== Restoring $CURRENT_BRANCH ==="

    (cd "$FI_REPO" && git checkout -q "$CURRENT_BRANCH")
    (cd "$FI_REPO/3rdparty/cutlass" && git checkout -q "$CURRENT_BRANCH")
    (cd "$VLLM_REPO" && git checkout -q "$CURRENT_BRANCH")

    log "Restored to $CURRENT_BRANCH"
    log "  FlashInfer: $(cd "$FI_REPO" && git rev-parse --short HEAD)"
    log "  vLLM:       $(cd "$VLLM_REPO" && git rev-parse --short HEAD)"
    log "  CUTLASS:    $(cd "$FI_REPO/3rdparty/cutlass" && git rev-parse --short HEAD)"
}

# =============================================================================
# Run a single variant
#
# Order matters:
#   1. Clean benchmark first — warms JIT cache
#   2. Profiled benchmark second — JIT cache warm, nsys captures kernel data
# =============================================================================

run_variant() {
    local variant="$1"
    local fuse_val="$2"
    local dir="$RESULTS/$variant"
    mkdir -p "$dir"

    log ""
    log "============================================================"
    log "  VARIANT: $variant  (VLLM_MXFP4_FUSE_ACTIVATION=$fuse_val)"
    log "============================================================"
    log ""

    # ---- Phase 1: Clean benchmark (start.sh + bench.sh) ----
    # This warms the JIT cache for the profiled run.
    stop_server
    clear_jit_cache
    clear_compile_caches

    log "[$variant] Phase 1: Clean benchmark (start.sh + bench.sh)..."

    # Truncate server log to prevent stale data from previous variant's nsys
    # output leaking into this variant's log file.
    docker exec "$CONTAINER" truncate -s 0 /tmp/vllm_server.log 2>/dev/null || true

    # Start server in background (detached), log to /tmp/vllm_server.log
    VLLM_MXFP4_FUSE_ACTIVATION="$fuse_val" \
    VLLM_DOCKER_EXEC_FLAGS="-d" \
        "$SCRIPT_DIR/start.sh" 2>&1 | tail -5 || true

    if wait_for_server; then
        # Warmup request
        log "[$variant] Warmup request..."
        docker exec "$CONTAINER" curl -sf http://localhost:8000/v1/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"gpt-oss-120b","prompt":"Hello world","max_tokens":8,"temperature":0}' \
            > /dev/null 2>&1 || true
        sleep 2

        # Wait for GPU to cool down after warmup before measuring
        wait_for_cooldown

        # Record GPU temp before benchmark
        record_temp "pre_clean" "$dir"

        # Run clean benchmark
        log "[$variant] Running bench.sh..."
        VLLM_DOCKER_EXEC_FLAGS="-i" \
            "$SCRIPT_DIR/bench.sh" 2>&1 | tee "$dir/bench_clean.log" || true

        # Record GPU temp after benchmark
        record_temp "post_clean" "$dir"
    else
        log "[$variant] ERROR: Server failed to start for clean benchmark"
        echo "SERVER_FAILED_TO_START" > "$dir/bench_clean.log"
        docker cp "$CONTAINER:/tmp/vllm_server.log" "$dir/server_clean.log" 2>/dev/null || true
        stop_server
        save_metadata "$variant" "$fuse_val"
        log "[$variant] Aborting variant — server failed to start"
        return 1
    fi

    # Copy server log (captures startup errors even in detached mode)
    docker cp "$CONTAINER:/tmp/vllm_server.log" "$dir/server_clean.log" 2>/dev/null || true

    # ---- Phase 2: Profiled benchmark (start-profile.sh) ----
    # JIT cache is now warm from Phase 1.
    stop_server

    # Wait for GPU to cool down before profiled benchmark
    wait_for_cooldown

    log "[$variant] Phase 2: Profiled benchmark (start-profile.sh)..."
    # Remove stale nsys output so nsys uses the exact filename we request
    docker exec "$CONTAINER" bash -c "rm -f /tmp/nsys_profile/${variant}*" 2>/dev/null || true
    # Truncate server log before profiled run too
    docker exec "$CONTAINER" truncate -s 0 /tmp/vllm_server.log 2>/dev/null || true

    record_temp "pre_profiled" "$dir"

    VLLM_MXFP4_FUSE_ACTIVATION="$fuse_val" \
    PROFILE_OUTPUT="/tmp/nsys_profile/${variant}" \
        "$SCRIPT_DIR/start-profile.sh" 2>&1 | tee "$dir/bench_profiled.log" || true

    record_temp "post_profiled" "$dir"

    # Copy server log from profiled run
    docker cp "$CONTAINER:/tmp/vllm_server.log" "$dir/server_profiled.log" 2>/dev/null || true

    # Copy nsys profile and sqlite out of container.
    # nsys appends a numeric suffix (e.g., ".1") to output filenames, so try
    # both "${variant}.nsys-rep" and "${variant}.1.nsys-rep".
    if docker cp "$CONTAINER:/tmp/nsys_profile/${variant}.nsys-rep" "$dir/profile.nsys-rep" 2>/dev/null; then
        log "[$variant] Copied profile.nsys-rep ($(du -h "$dir/profile.nsys-rep" | cut -f1))"
    elif docker cp "$CONTAINER:/tmp/nsys_profile/${variant}.1.nsys-rep" "$dir/profile.nsys-rep" 2>/dev/null; then
        log "[$variant] Copied profile.nsys-rep from .1 suffix ($(du -h "$dir/profile.nsys-rep" | cut -f1))"
    else
        log "[$variant] WARNING: Could not copy nsys-rep"
    fi
    if docker cp "$CONTAINER:/tmp/nsys_profile/${variant}.sqlite" "$dir/profile.sqlite" 2>/dev/null; then
        log "[$variant] Copied profile.sqlite ($(du -h "$dir/profile.sqlite" | cut -f1))"
    elif docker cp "$CONTAINER:/tmp/nsys_profile/${variant}.1.sqlite" "$dir/profile.sqlite" 2>/dev/null; then
        log "[$variant] Copied profile.sqlite from .1 suffix ($(du -h "$dir/profile.sqlite" | cut -f1))"
    else
        log "[$variant] WARNING: Could not copy sqlite"
    fi

    stop_server

    # ---- Save metadata ----
    save_metadata "$variant" "$fuse_val"

    log "[$variant] Complete. Results in $dir/"
}

# =============================================================================
# Main
# =============================================================================

log "============================================================"
log "  Three-Way Benchmark Comparison"
log "  Results: $RESULTS"
log "============================================================"

# Pre-flight: verify container is running
if ! docker ps --filter "name=${CONTAINER}" --format "{{.Names}}" | grep -q "${CONTAINER}"; then
    log "ERROR: Container ${CONTAINER} is not running"
    log "Start it with: docker compose -f docker-compose.dev.yml up -d"
    exit 1
fi
log "Container $CONTAINER is running"

# Record starting git state
log "Starting state:"
log "  FlashInfer: $(cd "$FI_REPO" && git rev-parse --short HEAD) ($(cd "$FI_REPO" && git branch --show-current 2>/dev/null || echo detached))"
log "  vLLM:       $(cd "$VLLM_REPO" && git rev-parse --short HEAD) ($(cd "$VLLM_REPO" && git branch --show-current 2>/dev/null || echo detached))"
log "  CUTLASS:    $(cd "$FI_REPO/3rdparty/cutlass" && git rev-parse --short HEAD)"

mkdir -p "$RESULTS"

# Record system info for reproducibility
docker exec "$CONTAINER" bash -c '
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""
echo "=== CUDA ==="
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"
echo ""
echo "=== nsys ==="
nsys --version 2>&1
echo ""
echo "=== Python ==="
python3 --version
echo ""
echo "=== vLLM ==="
python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "not importable"
echo ""
echo "=== FlashInfer ==="
python3 -c "import flashinfer; print(flashinfer.__file__)" 2>/dev/null || echo "not importable"
' > "$RESULTS/system_info.txt" 2>&1
log "System info saved to $RESULTS/system_info.txt"

# ---- Run 1: Golden ----
#if [[ "$VARIANTS" == "all" || "$VARIANTS" == "golden" ]]; then
#    checkout_golden
#    run_variant golden 0
#    restore_current
#fi

# ---- Run 2: Current Fused ----
if [[ "$VARIANTS" == "all" || "$VARIANTS" == "fused" ]]; then
    if ! run_variant fused 1; then
        log "FATAL: 'fused' variant failed to start, aborting."
        exit 1
    fi
fi

# ---- Run 3: Current Unfused ----
if [[ "$VARIANTS" == "all" || "$VARIANTS" == "unfused" ]]; then
    if ! run_variant unfused 0; then
        log "FATAL: 'unfused' variant failed to start, aborting."
        exit 1
    fi
fi

log ""
log "============================================================"
log "  All benchmarks complete!"
log "  Results directory: $RESULTS"
log ""
ls -la "$RESULTS"/*/
log "============================================================"
