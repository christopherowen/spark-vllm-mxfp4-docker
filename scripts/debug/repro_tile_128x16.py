#!/usr/bin/env python3
"""
Repro for SM120/121 tile_mn=(128,16) illegal-instruction.

Intended usage (inside container):
  export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
  export FLASHINFER_FUSED_MOE_BUILD_PROFILE=mxfp4_minimal
  export CUDA_LAUNCH_BLOCKING=1
  cuda-gdb --args python3 /tmp/repro_tile_128x16.py
"""

import torch


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def main():
    device = "cuda"
    assert torch.cuda.is_available()

    # Small shapes to reproduce quickly.
    num_tokens = 128
    hidden_size = 256
    inter_size = 256
    num_experts = 1
    topk = 1

    x_fp8 = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )

    token_selected_experts = torch.zeros(
        (num_tokens, topk), device=device, dtype=torch.int32
    )
    token_final_scales = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)

    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )

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

    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"

    print("backend", backend)
    print("Building module tile_mn=(128,16)")
    mod = get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=(128, 16))
    print("Module ready")

    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    print("Calling kernel")
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
        activation_type=3,
    )
    torch.cuda.synchronize()
    out_y = y[0] if isinstance(y, (list, tuple)) else y
    print("OK", out_y.shape)


if __name__ == "__main__":
    main()

