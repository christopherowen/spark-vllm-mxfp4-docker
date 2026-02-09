#!/bin/bash
# Start vLLM with fused activation (single-kernel FC1 + SwiGLU)
exec env VLLM_MXFP4_FUSE_ACTIVATION=1 "$(dirname "$0")/start.sh" "$@"
