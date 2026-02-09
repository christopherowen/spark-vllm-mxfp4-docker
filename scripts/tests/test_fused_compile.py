#!/usr/bin/env python3
"""Test fused activation kernel compilation and basic execution."""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType
from flashinfer import mxfp8_quantize

torch.manual_seed(42)
device = 'cuda'

x = torch.ones(4, 256, dtype=torch.bfloat16, device=device)
xq, xs = mxfp8_quantize(x, True, 32)
fc1 = torch.zeros(1, 1024, 128, dtype=torch.uint8, device=device)
fc1s = torch.ones(1, 1024, 8, dtype=torch.uint8, device=device) * 127
fc2 = torch.zeros(1, 256, 256, dtype=torch.uint8, device=device)
fc2s = torch.ones(1, 256, 16, dtype=torch.uint8, device=device) * 127
te = torch.zeros(4, 1, dtype=torch.int32, device=device)
tw = torch.ones(4, 1, dtype=torch.float32, device=device)
fs = torch.ones(1, dtype=torch.float32, device=device)

print('Building fused kernel (release mode)...')
out = cutlass_fused_moe(
    input=xq,
    token_selected_experts=te,
    token_final_scales=tw,
    fc1_expert_weights=fc1,
    fc2_expert_weights=fc2,
    output_dtype=torch.bfloat16,
    activation_type=ActivationType.Swiglu,
    quant_scales=[fc1s, fs, fc2s, fs],
    input_sf=xs,
    fuse_activation=True,
)
print(f'Shape: {out[0].shape} Zeros: {(out[0]==0).all()}')
print('SUCCESS')
