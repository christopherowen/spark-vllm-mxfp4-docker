#!/usr/bin/env python3
"""Unfused-only MoE GEMM for initcheck triage.

Runs ONLY the unfused (fuse_activation=False) CUTLASS SM120 MoE path so
compute-sanitizer initcheck reports can be attributed to a single kernel
without fused-path noise.

Usage:
    # Direct run (verify it works):
    python3 test_initcheck_unfused.py

    # Under initcheck with one report:
    compute-sanitizer --tool initcheck --print-limit 1 \
        python3 test_initcheck_unfused.py
"""
import os
import sys

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
sys.path.insert(0, '/workspace/flashinfer')
sys.path.insert(0, '/workspace/vllm')

import torch
from flashinfer import mxfp4_quantize, mxfp8_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"


def normalize_moe_output(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    raise RuntimeError(f"Unexpected return type: {type(out)}")


# gpt-oss-120b dimensions
hidden_size = 2944
intermediate_size = 7680
num_experts = 8
top_k = 2
num_tokens = 4

print(f"Config: hidden={hidden_size}, inter={intermediate_size}, "
      f"experts={num_experts}, top_k={top_k}, tokens={num_tokens}")

# Create inputs
x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)

# FC1: [experts, 2*inter, hidden] (gated)
w13_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                        dtype=torch.bfloat16, device=device) * 0.01
# FC2: [experts, hidden, inter]
w2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                       dtype=torch.bfloat16, device=device) * 0.01

# Quantize weights
w13_flat = w13_bf16.reshape(-1, hidden_size)
w2_flat = w2_bf16.reshape(-1, intermediate_size)
w13_fp4, w13_scale = mxfp4_quantize(w13_flat)
w2_fp4, w2_scale = mxfp4_quantize(w2_flat)

w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)
w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)

fc1_weights = w13_fp4.contiguous().view(torch.long)
fc2_weights = w2_fp4.contiguous().view(torch.long)
fc1_scale = w13_scale.contiguous().view(torch.int32)
fc2_scale = w2_scale.contiguous().view(torch.int32)

# Router
topk_ids = torch.zeros(num_tokens, top_k, device=device, dtype=torch.int32)
for t in range(num_tokens):
    topk_ids[t, 0] = t % num_experts
    topk_ids[t, 1] = (t + 1) % num_experts
topk_weights = torch.full((num_tokens, top_k), 1.0 / top_k,
                           device=device, dtype=torch.float32)

fake_scale = torch.ones(num_experts, device=device)
quant_scales = [fc1_scale, fake_scale, fc2_scale, fake_scale]

# Unfused only
print("\n--- Testing UNFUSED (fuse_activation=False) ---")
try:
    raw = cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True,
        input_sf=x_scale,
        quant_scales=quant_scales,
        fuse_activation=False,
        enable_pdl=False,
    )
    out = normalize_moe_output(raw)
    torch.cuda.synchronize()
    print(f"  OK: shape={out.shape}, "
          f"abs_max={out.float().abs().max().item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone.")
