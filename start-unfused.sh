#!/bin/bash
# Start vLLM with unfused activation (standard two-kernel FC1 + SwiGLU)
exec env VLLM_MXFP4_FUSE_ACTIVATION=0 "$(dirname "$0")/start.sh" "$@"
