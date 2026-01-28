# NCCL Tuning for TP=2 Inter-Node Performance

This document covers NCCL optimization options for improving TP=2 performance over ConnectX-7 (200Gbps) network.

## Current Baseline

| Config | Decode (tok/s) | Per-token latency |
|--------|----------------|-------------------|
| TP=1 | 60 | 16.7ms |
| TP=2 | 52 | 19.2ms |
| **Gap** | **8 tok/s** | **2.5ms** |

The 2.5ms overhead comes from **37 NCCL collective operations** per token (36 attention reduce_scatter + 1 lm_head all_gather).

## NCCL Environment Variables

### Low-Latency Protocol (for small messages)

```bash
# Use Low-Latency protocol - better for small messages
export NCCL_PROTO=LL128  # Options: Simple, LL, LL128

# LL128 is optimized for small messages over IB
# May improve latency for decode (small batch sizes)
```

### Algorithm Selection

```bash
# Tree algorithm often better for small collectives
export NCCL_ALGO=Tree  # Options: Ring, Tree, CollnetDirect, CollnetChain

# For all_gather specifically:
export NCCL_ALGO=allgather:Tree,reducescatter:Tree
```

### Channel Tuning

```bash
# More channels = more parallelism, but more overhead
# For small messages, fewer channels may be better
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4

# Default is often 8-16, which adds overhead for small ops
```

### Buffer Size

```bash
# Smaller buffer for lower latency
export NCCL_BUFFSIZE=2097152  # 2MB (default is 4MB)
```

### GPUDirect RDMA (should already be enabled)

```bash
# Ensure GPUDirect RDMA is enabled
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GDR_LEVEL=5

# Use specific HCA if multiple NICs
export NCCL_IB_HCA=mlx5_0
```

### Debug/Profiling

```bash
# Enable NCCL debug info
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Time collectives
export NCCL_DEBUG=TRACE
```

## vLLM-Specific Options

### Ray Compiled DAG with Communication Overlap (Experimental)

```bash
# Enable communication overlap with Ray
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1

# Use NCCL channel type
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl
```

### NCCL Symmetric Memory (for all_reduce)

```bash
# Enable symmetric memory for faster all_reduce
export VLLM_USE_NCCL_SYMM_MEM=1
```

## Recommended Configurations to Test

### Config A: Low-Latency Focus

```bash
export NCCL_PROTO=LL128
export NCCL_ALGO=Tree
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export NCCL_BUFFSIZE=2097152
```

### Config B: Overlap Communication

```bash
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl
```

### Config C: Combined

```bash
export NCCL_PROTO=LL128
export NCCL_ALGO=Tree
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1
```

## Benchmarking Script

```bash
#!/bin/bash
# Test different NCCL configurations

configs=(
    "baseline"
    "NCCL_PROTO=LL128"
    "NCCL_PROTO=LL128 NCCL_ALGO=Tree"
    "NCCL_PROTO=LL128 NCCL_ALGO=Tree NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=4"
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1"
)

for config in "${configs[@]}"; do
    echo "Testing: $config"
    export $config 2>/dev/null || true
    
    # Run benchmark
    llama-benchy --base-url http://localhost:8000/v1 \
        --model gpt-oss-120b \
        --tokenizer openai/gpt-oss-120b \
        --pp 2048 --tg 32 --runs 3
    
    # Reset
    unset NCCL_PROTO NCCL_ALGO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
    unset VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM
done
```

## Hardware Considerations

### ConnectX-7 (200Gbps)

- Theoretical bandwidth: 25 GB/s
- Typical latency: 1-2μs wire, 10-50μs with NCCL overhead
- For small messages (< 1MB), latency dominates

### SHARP (if available)

SHARP offloads collectives to the network switch, reducing latency:

```bash
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_SHARED_COMMS=1
```

Note: Requires SHARP-enabled InfiniBand switches.

## Architectural Alternatives

If NCCL tuning doesn't provide enough improvement:

1. **Pipeline Parallelism (PP)** instead of TP
   - Split layers across nodes instead of splitting within layers
   - Reduces collectives from 37 per token to ~1 per micro-batch

2. **Expert Parallelism only (EP=2, TP=1)**
   - MoE layers use EP, dense layers stay local
   - Requires model to fit on single GPU

3. **Speculative Decoding**
   - Generate multiple tokens per forward pass
   - Amortizes collective overhead

## Profiling Commands

### NCCL Timing

```bash
NCCL_DEBUG=TRACE vllm serve ... 2>&1 | grep -i "elapsed"
```

### Nsight Systems

```bash
nsys profile -o tp2_profile \
    --trace=cuda,nccl \
    --nccl-algo=all \
    python -m vllm.entrypoints.openai.api_server ...
```

### vLLM with NVTX

```bash
VLLM_TORCH_PROFILER_DIR=/tmp/profiles vllm serve ...
```
