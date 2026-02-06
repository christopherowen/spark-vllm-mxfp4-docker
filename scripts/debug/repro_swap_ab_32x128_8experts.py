#!/usr/bin/env python3
"""
Repro harness to exercise SM120/121 swap_ab (M<64) with more "MoE-like" routing.
Designed to stay memory-light while increasing experts/topk.
"""

import torch


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def main():
    device = "cuda"
    assert torch.cuda.is_available()

    num_tokens = 32  # triggers swap_ab for tile_mn=(32,128)
    hidden_size = 1024
    inter_size = 4096
    num_experts = 8
    topk = 2

    x_fp8 = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )

    token_selected_experts = torch.zeros(
        (num_tokens, topk), device=device, dtype=torch.int32
    )
    for t in range(num_tokens):
        token_selected_experts[t, 0] = t % num_experts
        token_selected_experts[t, 1] = (t + 1) % num_experts

    token_final_scales = torch.full(
        (num_tokens, topk), 0.5, device=device, dtype=torch.float32
    )

    # FP4 packed weights (int64)
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )

    # MXFP4 scales
    FP8_PER_INT32 = 4
    SFVEC = 32
    hs_aligned_k = align_to(hidden_size, 128)
    hs_aligned_n = align_to(hidden_size, 128)
    inter_aligned_n = align_to(inter_size, 128)
    inter_aligned_k = align_to(inter_size, 128)

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

    tile_mn = (32, 128)
    print("Building module tile_mn=", tile_mn)
    mod = get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=tile_mn)
    print("module ready")

    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
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

