#!/bin/bash
# =============================================================================
# MXFP4 Benchmark Suite
# Produces 4 benchmark scenarios for validation
# =============================================================================

set -e

RESULTS_DIR="/home/swank/projects/ai/mxfp4/benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== MXFP4 Benchmark Suite ===" | tee "$RESULTS_DIR/summary.txt"
echo "Date: $(date)" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Common settings
export VLLM_NO_USAGE_STATS=1

run_benchmark() {
    local name="$1"
    local tp="$2"
    local max_seqs="$3"
    local max_model_len="$4"
    local max_batched_tokens="$5"
    
    echo "=== Benchmark: $name ===" | tee -a "$RESULTS_DIR/summary.txt"
    echo "TP=$tp, max_seqs=$max_seqs, max_model_len=$max_model_len" | tee -a "$RESULTS_DIR/summary.txt"
    
    # Run llama-benchy
    docker exec -i vllm-dev bash -c "
        llama-benchy \
            --base-url http://localhost:8000/v1 \
            --model gpt-oss-120b \
            --tokenizer openai/gpt-oss-120b \
            --pp 512 2048 \
            --tg 32 128 \
            --runs 3
    " 2>&1 | tee "$RESULTS_DIR/${name}.txt"
    
    echo "" | tee -a "$RESULTS_DIR/summary.txt"
}

echo "Benchmark results will be saved to: $RESULTS_DIR"
