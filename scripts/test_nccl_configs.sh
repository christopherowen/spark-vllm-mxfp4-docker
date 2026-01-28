#!/bin/bash
# Test different NCCL configurations for TP=2 over DAC
#
# Usage: 
#   1. Start vLLM server with TP=2 on the cluster
#   2. Run this script from a client machine
#
# The script will restart the server with different NCCL configs
# and benchmark each one.

set -e

# Configuration
BENCH_CMD="llama-benchy --base-url http://localhost:8000/v1 --model gpt-oss-120b --tokenizer openai/gpt-oss-120b --pp 2048 --tg 32 --runs 3"

echo "=============================================="
echo "NCCL Configuration Testing for TP=2 DAC"
echo "=============================================="
echo ""
echo "Make sure the vLLM server is NOT running."
echo "This script will set env vars and you'll need to restart the server."
echo ""

# Define configurations to test
declare -A configs
configs["baseline"]=""
configs["ll128"]="NCCL_PROTO=LL128"
configs["ll128_tree"]="NCCL_PROTO=LL128 NCCL_ALGO=Tree"
configs["ll128_ring_2ch"]="NCCL_PROTO=LL128 NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2"
configs["ll128_small_buf"]="NCCL_PROTO=LL128 NCCL_BUFFSIZE=1048576"
configs["overlap_comm"]="VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1"
configs["combined"]="NCCL_PROTO=LL128 NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2 NCCL_BUFFSIZE=1048576"

echo "Configurations to test:"
for name in "${!configs[@]}"; do
    echo "  - $name: ${configs[$name]:-'(default)'}"
done
echo ""

# Output results file
RESULTS_FILE="/tmp/nccl_benchmark_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

cat << 'EOF'
To test each configuration:

1. Stop the current vLLM server

2. For each config, export the variables and restart:

EOF

for name in "${!configs[@]}"; do
    echo "# === Test: $name ==="
    if [ -n "${configs[$name]}" ]; then
        echo "export ${configs[$name]}"
    else
        echo "# (no extra env vars - baseline)"
    fi
    echo "# Restart vLLM server, then run:"
    echo "# $BENCH_CMD"
    echo ""
done

cat << 'EOF'

Alternatively, run the benchmark interactively:

EOF

echo '# Quick benchmark command:'
echo "$BENCH_CMD"
echo ""

# Interactive mode
read -p "Would you like to run benchmarks interactively? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting interactive benchmark mode..."
    echo "Make sure the vLLM server is running with desired NCCL config."
    echo ""
    
    for name in "${!configs[@]}"; do
        echo "=============================================="
        echo "Ready to test: $name"
        echo "Config: ${configs[$name]:-'(baseline - no extra vars)'}"
        echo "=============================================="
        echo ""
        read -p "Press Enter when server is ready with this config (or 's' to skip): " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Ss]$ ]]; then
            echo "Running benchmark..."
            echo ""
            echo "=== $name ===" >> "$RESULTS_FILE"
            echo "Config: ${configs[$name]:-'baseline'}" >> "$RESULTS_FILE"
            
            # Run benchmark and capture output
            if $BENCH_CMD 2>&1 | tee -a "$RESULTS_FILE"; then
                echo "" >> "$RESULTS_FILE"
                echo "Benchmark completed for $name"
            else
                echo "Benchmark failed for $name" | tee -a "$RESULTS_FILE"
            fi
            echo ""
        else
            echo "Skipped $name"
        fi
    done
    
    echo ""
    echo "=============================================="
    echo "All benchmarks complete!"
    echo "Results saved to: $RESULTS_FILE"
    echo "=============================================="
fi
