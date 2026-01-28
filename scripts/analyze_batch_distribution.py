#!/usr/bin/env python3
"""
Analyze when batch size (M) allows switching from GEMV to GEMM.

Key factors:
1. Decode phase: M = concurrent requests being decoded
2. Prefill phase: M = prompt tokens (can be large)
3. Speculative decoding: M = num_speculative_tokens + 1
4. Tensor Core tile efficiency depends on M
"""

print("=" * 70)
print("Batch Size Analysis: GEMV vs GEMM")
print("=" * 70)

print("""
GEMV vs GEMM Efficiency Crossover
=================================

GEMV (DP4A on CUDA Cores):
- Optimal for M=1 (single token decode)
- Each thread computes one output element
- Memory-bound, ~80-90% bandwidth efficiency
- No wasted compute for small M

GEMM (Tensor Cores via CUTLASS):
- Minimum tile sizes: 64×64, 64×128, 128×128
- For M < tile_M, padding wastes compute
- For M >= tile_M, excellent efficiency
- Block-scaled MMA: FP8×FP4 on SM121

Crossover Point Analysis
========================

For gpt-oss-120b dense layers (K=2880):

M=1:  GEMV wins - no tile overhead
M=2:  GEMV wins - GEMM would waste 97% of 64-row tile
M=4:  GEMV wins - GEMM wastes 94%
M=8:  Borderline - GEMM wastes 88%, but has higher peak FLOPS
M=16: GEMM competitive - wastes 75% but Tensor Core throughput helps
M=32: GEMM wins - 50% tile utilization + Tensor Core advantage
M=64: GEMM clearly wins - 100% tile utilization

Rough crossover: M >= 8-16 for GEMM to be competitive
""")

# Workload analysis
print("\nWorkload Distribution for gpt-oss-120b on GB10")
print("=" * 50)

workloads = [
    ("Single-user chat", "1", "Decode: M=1"),
    ("2 concurrent users", "2", "Decode: M=2 (still GEMV)"),
    ("4 concurrent users", "4", "Decode: M=4 (GEMV)"),
    ("8 concurrent users", "8", "Decode: M=8 (borderline)"),
    ("Prefill (512 tokens)", "512", "GEMM clearly better"),
    ("Prefill (2K tokens)", "2048", "GEMM clearly better"),
    ("Prefill (8K tokens)", "8192", "GEMM optimal"),
    ("Eagle3 speculation (k=4)", "5", "5 tokens: GEMV"),
    ("Eagle3 speculation (k=8)", "9", "9 tokens: borderline"),
]

print(f"\n{'Workload':<30} {'M':>6} {'Recommendation':>20}")
print("-" * 60)
for name, m, rec in workloads:
    print(f"{name:<30} {m:>6} {rec:>20}")

# Memory constraint analysis
print("\n\nGB10 Memory Constraint Analysis")
print("=" * 50)
print("""
gpt-oss-120b model size: ~60GB (MXFP4 weights)
GB10 memory: 120GB
Available for KV cache: ~50GB

KV cache per token (FP8, 60 layers):
  K: 2880 * 60 * 1 byte = 173 KB
  V: 360 * 60 * 1 byte = 22 KB (GQA)
  Total: ~200 KB per token

Max context with 50GB KV cache:
  50GB / 200KB ≈ 250,000 tokens total

Practical concurrent users:
  - 8K context each: ~6 concurrent conversations
  - 4K context each: ~12 concurrent conversations
  - 2K context each: ~25 concurrent conversations

During decode, each user generates 1 token → M = num_users
""")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: When to Use GEMM vs GEMV")
print("=" * 70)
print("""
Phase          | Typical M | Kernel Choice | Notes
---------------|-----------|---------------|---------------------------
Prefill        | 512-8192  | GEMM          | Always use Tensor Cores
Decode (1 user)| 1         | GEMV          | DP4A optimal
Decode (2-4)   | 2-4       | GEMV          | Still GEMV territory
Decode (8+)    | 8+        | GEMM          | Consider switching
Speculation    | 4-9       | GEMV/GEMM     | Depends on k

Practical Reality for GB10:
- Most decode steps: M=1 to M=6 (few concurrent users)
- Prefill: Always GEMM
- Memory limits batch size more than compute

Recommendation:
- Keep GEMV for M < 8
- Switch to CUTLASS GEMM for M >= 8
- Always use GEMM for prefill
""")
