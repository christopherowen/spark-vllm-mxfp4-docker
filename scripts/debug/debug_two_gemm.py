#!/usr/bin/env python3
"""Debug the two-GEMM gated FC1 kernel output."""

import torch
import os
os.environ["FLASHINFER_LOGLEVEL"] = "3"

# Import FlashInfer MoE
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fp4_quantization import mxfp4_quantize, mxfp8_quantize

def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    
    # Minimal test sizes (simulating single token decode)
    num_tokens = 1
    hidden_size = 2048
    inter_size = 1024  # Must be divisible by 32
    num_experts = 8
    top_k = 2
    
    print(f"=== Two-GEMM Gated FC1 Debug ===")
    print(f"num_tokens={num_tokens}, hidden_size={hidden_size}, inter_size={inter_size}")
    print(f"num_experts={num_experts}, top_k={top_k}")
    
    # Create input
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    
    # Create routing (all tokens go to experts 0 and 1)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=dtype, device=device)
    
    # Create FC1 weights (gate + up packed): [num_experts, 2*inter_size, hidden_size]
    # In MXFP4, weights are packed with gate in first half, up in second half
    fc1_weights_bf16 = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device)
    fc1_weights_fp4, fc1_scales = mxfp4_quantize(fc1_weights_bf16.view(-1, hidden_size))
    fc1_weights_fp4 = fc1_weights_fp4.view(num_experts, 2 * inter_size, -1)
    fc1_scales = fc1_scales.view(num_experts, 2 * inter_size, -1)
    
    # Create FC2 weights: [num_experts, hidden_size, inter_size]
    fc2_weights_bf16 = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device)
    fc2_weights_fp4, fc2_scales = mxfp4_quantize(fc2_weights_bf16.view(-1, inter_size))
    fc2_weights_fp4 = fc2_weights_fp4.view(num_experts, hidden_size, -1)
    fc2_scales = fc2_scales.view(num_experts, hidden_size, -1)
    
    print(f"\nInput stats:")
    print(f"  hidden_states: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
    print(f"  min={hidden_states.min():.4f}, max={hidden_states.max():.4f}")
    
    print(f"\nFC1 weight stats (BF16):")
    print(f"  mean={fc1_weights_bf16.mean():.4f}, std={fc1_weights_bf16.std():.4f}")
    
    # Run the MoE
    try:
        output = cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_weights_fp4,
            fc2_expert_weights=fc2_weights_fp4,
            fc1_expert_scales=fc1_scales,
            fc2_expert_scales=fc2_scales,
            output_dtype=dtype,
            activation_type="swiglu",
        )
        
        print(f"\nOutput stats:")
        print(f"  shape: {output.shape}")
        print(f"  dtype: {output.dtype}")
        print(f"  mean={output.mean():.4f}, std={output.std():.4f}")
        print(f"  min={output.min():.4f}, max={output.max():.4f}")
        print(f"  has_nan: {torch.isnan(output).any()}")
        print(f"  has_inf: {torch.isinf(output).any()}")
        print(f"  all_zero: {(output == 0).all()}")
        
        # Check for garbage (very large values)
        if output.abs().max() > 1000:
            print(f"  WARNING: Very large values detected!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
