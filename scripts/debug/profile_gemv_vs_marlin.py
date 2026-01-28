"""Profile GEMV vs Marlin for MXFP4 dense layers."""
import torch
import time

# gpt-oss-120b dimensions (head_dim=64, not hidden_size/num_heads!)
K_HIDDEN = 2880  # hidden_dim (input to QKV, LM head)
K_ATTN = 4096  # attention output dim (64 heads * 64 head_dim) - input to O proj
N_QKV = 5120  # fused qkv projection: Q(4096) + K(512) + V(512)
N_O = 2880  # o_proj output (hidden_size)
N_LM_HEAD = 201088  # lm_head


def profile_gemv(M, N, K, warmup=10, iters=100):
    """Profile our DP4A GEMV kernel."""
    from flashinfer.gemv import gemv_mxfp4_dp4a

    # Create inputs
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")

    # Warmup
    for _ in range(warmup):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    for _ in range(iters):
        out = gemv_mxfp4_dp4a(input_bf16, weight_fp4, weight_scale)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms

    return elapsed


def profile_marlin(M, N, K, warmup=10, iters=100):
    """Profile Marlin dequant->GEMM using apply_fp4_marlin_linear."""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
    except ImportError as e:
        print(f"Marlin import failed: {e}")
        return None

    # Create inputs matching Marlin expectations
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Create a mock layer with FP4 weights
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            )
            self.weight_scale = torch.nn.Parameter(
                torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            )
            self.params_dtype = torch.bfloat16
            self.input_size_per_partition = K
            self.output_size_per_partition = N

    layer = MockLayer().cuda()

    try:
        prepare_fp4_layer_for_marlin(layer, input_dtype=torch.bfloat16)
    except Exception as e:
        print(f"Marlin prep failed: {e}")
        return None

    # Warmup (prepare_fp4_layer_for_marlin modifies layer.weight in place)
    try:
        for _ in range(warmup):
            out = apply_fp4_marlin_linear(
                input=input_bf16,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_scale_2=None,
                workspace=layer.workspace,
                size_n=N,
                size_k=K,
                bias=None,
            )
        torch.cuda.synchronize()
    except RuntimeError as e:
        # Marlin has minimum M requirements
        return None

    # Time
    start = time.perf_counter()
    for _ in range(iters):
        out = apply_fp4_marlin_linear(
            input=input_bf16,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_scale_2=None,
            workspace=layer.workspace,
            size_n=N,
            size_k=K,
            bias=None,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms

    return elapsed


def profile_bf16_baseline(M, N, K, warmup=10, iters=100):
    """Profile BF16 matmul as theoretical best case."""
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        out = torch.mm(input_bf16, weight_bf16.T)
    torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    for _ in range(iters):
        out = torch.mm(input_bf16, weight_bf16.T)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms

    return elapsed


if __name__ == "__main__":
    print("=" * 70)
    print("GEMV vs Marlin vs BF16 Baseline Comparison")
    print("gpt-oss-120b dimensions")
    print("=" * 70)
    print()

    # Test different M values and N values
    # (M, K, N, description)
    # K varies by layer: K_HIDDEN=2880 for QKV/LM, K_ATTN=4096 for O proj
    test_cases = [
        (1, K_HIDDEN, N_QKV, "M=1, QKV"),
        (1, K_ATTN, N_O, "M=1, O proj"),
        (1, K_HIDDEN, N_LM_HEAD, "M=1, LM Head"),
        (4, K_HIDDEN, N_QKV, "M=4, QKV"),
        (8, K_HIDDEN, N_QKV, "M=8, QKV"),
        (16, K_HIDDEN, N_QKV, "M=16, QKV"),
        (32, K_HIDDEN, N_QKV, "M=32, QKV"),
        (64, K_HIDDEN, N_QKV, "M=64, QKV"),
    ]

    header = f"{'Case':<20} {'GEMV (ms)':<12} {'Marlin (ms)':<14} {'BF16 (ms)':<12} {'Winner':<10}"
    print(header)
    print("-" * 70)

    for M, K, N, desc in test_cases:
        gemv_time = profile_gemv(M, N, K)
        marlin_time = profile_marlin(M, N, K)
        bf16_time = profile_bf16_baseline(M, N, K)

        if marlin_time is None:
            winner = "GEMV"
            marlin_str = "N/A"
        else:
            marlin_str = f"{marlin_time:.4f}"
            if gemv_time < marlin_time:
                winner = "GEMV"
            else:
                winner = "Marlin"

        print(f"{desc:<20} {gemv_time:.4f}       {marlin_str:<14} {bf16_time:.4f}       {winner}")

    print()
    print("Note: Lower is better. BF16 is theoretical best (no quantization overhead)")
