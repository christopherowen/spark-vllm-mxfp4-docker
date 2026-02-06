#!/usr/bin/env python3
"""Test gated FC1 kernel with 64x64x128 tile.

This script tests whether the 64x64x128 gated tile compiles and runs.
"""

import os
import sys

# Set environment for gated FC1 kernel
os.environ["FLASHINFER_GATED_FC1_LAUNCH"] = "1"
os.environ["FLASHINFER_LOGLEVEL"] = "3"
os.environ["FLASHINFER_JIT_VERBOSE"] = "1"

# Ensure we use local flashinfer
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch

def test_gated_fc1_compile():
    """Test if gated FC1 kernel compiles with 64x64x128 tile."""
    print("=" * 60)
    print("Testing Gated FC1 Kernel (64x64x128 tile)")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Import flashinfer after setting environment
    try:
        import flashinfer
        print(f"FlashInfer: {flashinfer.__file__}")
    except ImportError as e:
        print(f"ERROR: Could not import flashinfer: {e}")
        return False
    
    # Try to import the cutlass fused moe module to trigger JIT
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        print("Imported cutlass_fused_moe")
    except ImportError as e:
        print(f"ERROR: Could not import cutlass_fused_moe: {e}")
        return False
    
    # Create test tensors for a small MoE forward pass
    # This will trigger the JIT compilation of the gated kernel
    
    M = 64  # tokens (fits in 64x64 tile)
    K = 256  # hidden_size (must be multiple of 32)
    N = 128  # inter_size (must be multiple of 32)
    num_experts = 2
    topk = 1
    
    print(f"\nTest shape: M={M}, K={K}, N={N}, experts={num_experts}, topk={topk}")
    
    # Create input tensor
    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Create router (top-k selection)
    router_logits = torch.randn(M, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_indices = torch.topk(router_logits, topk, dim=-1)
    topk_weights = torch.softmax(topk_weights.float(), dim=-1).to(torch.float32)
    
    # Create FC1 weights (2*N for gated)
    # Shape: [experts, 2*inter_size, hidden_size] for column-major weights
    fc1_weights = torch.randn(num_experts, 2 * N, K, dtype=torch.bfloat16, device=device)
    
    # Create FC2 weights
    # Shape: [experts, hidden_size, inter_size]
    fc2_weights = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    
    # Quantize weights to MXFP4
    print("\nQuantizing weights to MXFP4...")
    try:
        from flashinfer import mxfp4_quantize
        
        fc1_quant, fc1_scale = mxfp4_quantize(fc1_weights.view(num_experts, -1))
        fc2_quant, fc2_scale = mxfp4_quantize(fc2_weights.view(num_experts, -1))
        
        # Reshape back
        fc1_quant = fc1_quant.view(num_experts, 2 * N, K // 2)  # FP4 packed
        fc2_quant = fc2_quant.view(num_experts, K, N // 2)
        
        print(f"FC1 quantized: {fc1_quant.shape}, scales: {fc1_scale.shape}")
        print(f"FC2 quantized: {fc2_quant.shape}, scales: {fc2_scale.shape}")
    except Exception as e:
        print(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run cutlass_fused_moe which should trigger gated FC1 path
    print("\nRunning cutlass_fused_moe (should trigger gated FC1 JIT)...")
    try:
        from flashinfer.fused_moe.core import ActivationType
        
        output = cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_indices,
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_quant,
            fc2_expert_weights=fc2_quant,
            output_dtype=torch.bfloat16,
            quant_scales=[fc1_scale, fc2_scale],
            activation_type=ActivationType.Swiglu,
        )
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output sample: {output[0, :5]}")
        
        # Check for NaN/Inf
        if torch.isnan(output).any():
            print("WARNING: Output contains NaN!")
        if torch.isinf(output).any():
            print("WARNING: Output contains Inf!")
            
        print("\n" + "=" * 60)
        print("SUCCESS: Gated FC1 kernel compiled and ran!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nERROR during cutlass_fused_moe: {e}")
        import traceback
        traceback.print_exc()
        
        # Check CUDA error state
        try:
            torch.cuda.synchronize()
        except RuntimeError as cuda_err:
            print(f"\nCUDA error after synchronize: {cuda_err}")
        
        return False

if __name__ == "__main__":
    success = test_gated_fc1_compile()
    sys.exit(0 if success else 1)
