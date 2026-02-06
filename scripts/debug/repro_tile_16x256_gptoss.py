#!/usr/bin/env python3
"""
Repro: SM120/121 MoE tile_mn=(16,256) crash on gpt-oss-like dims.

Run inside vllm-dev:
  export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
  export FLASHINFER_FUSED_MOE_BUILD_PROFILE=mxfp4_minimal
  export CUDA_LAUNCH_BLOCKING=1
  export TLLM_LOG_LEVEL=DEBUG
  export TLLM_SM120_MOE_DUMP_TMA=1
  python3 /workspace/scripts/debug/repro_tile_16x256_gptoss.py
"""

import os
import sys

import torch


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def main() -> int:
    device = "cuda"

    # gpt-oss-like MoE params
    num_tokens = 16
    num_experts = 8
    topk = 2

    hidden_size = 6144
    inter_size = 24576

    tile_mn = (16, 256)

    # Activations (FP8)
    x_fp8 = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )

    # Routing
    token_selected_experts = torch.randint(
        0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32
    )
    token_final_scales = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).float()

    # Packed FP4 weights (int64 storage)
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )

    # MXFP4 quant scales
    FP8_PER_INT32 = 4
    SFVEC = 32
    MinNDimAlignment = 128
    MinKDimAlignment = 128

    hs_aligned_k = align_to(hidden_size, MinKDimAlignment)
    hs_aligned_n = align_to(hidden_size, MinNDimAlignment)
    inter_aligned_n = align_to(inter_size, MinNDimAlignment)
    inter_aligned_k = align_to(inter_size, MinKDimAlignment)

    fc1_weight_block = torch.zeros(
        (num_experts, inter_aligned_n * 2, hs_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc2_weight_block = torch.zeros(
        (num_experts, hs_aligned_n, inter_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc1_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    fc2_act_global = torch.ones((), device=device, dtype=torch.float32)
    fc2_global = torch.ones((num_experts,), device=device, dtype=torch.float32)

    quant_scales = [fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]

    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    print(f"device={torch.cuda.get_device_name()} cc={(major, minor)} backend={backend}")
    print(
        f"num_tokens={num_tokens} hidden={hidden_size} inter={inter_size} experts={num_experts} topk={topk}"
    )
    print(f"tile_mn={tile_mn}")

    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    mod = get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=tile_mn)

    y = mod.cutlass_fused_moe(
        out,
        x_fp8,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        None,
        fc2_expert_weights,
        None,
        torch.bfloat16,
        quant_scales,
        None,
        None,
        None,
        None,
        1,
        0,
        1,
        0,
        1,
        0,
        use_packed_weights=False,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        tune_max_num_tokens=256,
        enable_pdl=False,
        activation_type=3,  # SwiGLU
    )

    torch.cuda.synchronize()
    out_y = y[0] if isinstance(y, (list, tuple)) else y
    print("SUCCESS", out_y.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())

