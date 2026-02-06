#!/usr/bin/env python3
"""Analyze expected kernel launches per forward pass."""

print("=" * 70)
print("ANALYZING MoE KERNEL LAUNCH STRUCTURE")
print("=" * 70)
print()

num_layers = 24
experts_per_token = 8  # top-k

print("For gpt-oss-120b with 24 layers, top-8 experts:")
print()
print("MoE-RELATED KERNELS (per forward pass):")
print("-" * 50)
print()

# Check if grouped GEMM is truly batching all experts
print("If using CUTLASS Grouped GEMM (batches all experts in one launch):")
print(f"  - FC1 grouped GEMM:     {num_layers:3d} (one per layer)")
print(f"  - SwiGLU activation:    {num_layers:3d} (one per layer)")
print(f"  - FC2 grouped GEMM:     {num_layers:3d} (one per layer)")
print(f"  - Expert routing:       {num_layers:3d} (softmax + topk)")
print(f"  - Permute/scatter:      {num_layers:3d} (token reordering)")
print(f"  MoE Subtotal:           ~{5*num_layers} kernels")
print()

print("If launching per-expert (8 launches per GEMM):")
per_expert_gemm = num_layers * experts_per_token
print(f"  - FC1 per-expert GEMM:  {per_expert_gemm:3d} (8 per layer)")
print(f"  - SwiGLU activation:    {num_layers:3d} (one per layer)")
print(f"  - FC2 per-expert GEMM:  {per_expert_gemm:3d} (8 per layer)")
print(f"  - Expert routing:       {num_layers:3d}")
print(f"  - Permute/scatter:      {num_layers:3d}")
print(f"  MoE Subtotal:           ~{2*per_expert_gemm + 3*num_layers} kernels")
print()

print("OTHER PER-LAYER KERNELS:")
print("-" * 50)
print(f"  - Attention (FA2):      {num_layers:3d}")
print(f"  - QKV projection:       {num_layers:3d}")
print(f"  - O projection:         {num_layers:3d}")
print(f"  - RMSNorm:              {2*num_layers:3d} (2 per layer)")
print(f"  - Quantization:         {2*num_layers:3d} (~2 per layer)")
print(f"  Non-MoE Subtotal:       ~{7*num_layers} kernels")
print()

print("TOTALS:")
print("-" * 50)
grouped_total = 5*num_layers + 7*num_layers
per_expert_total = 2*per_expert_gemm + 3*num_layers + 7*num_layers
print(f"  With grouped GEMM:      ~{grouped_total} kernel launches")
print(f"  With per-expert GEMM:   ~{per_expert_total} kernel launches")
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("The '192 kernel launches' claim needs verification via nsys profiling.")
print()
print("If we see ~288 MoE GEMM kernels → per-expert launching (not grouped)")
print("If we see ~48 MoE GEMM kernels → grouped GEMM is working")
print()
print("Run: nsys profile --stats=true <inference command>")
print("Then look for kernel names containing 'moe_gemm' or 'cutlass'")
