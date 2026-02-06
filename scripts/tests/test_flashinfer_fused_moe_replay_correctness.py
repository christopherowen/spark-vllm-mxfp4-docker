#!/usr/bin/env python3
"""
Regression test: replay a captured vLLM MoE call and validate FlashInfer "fused"
path numerics against a straightforward BF16 reference.

Why this exists
---------------
When we make heavy edits to FlashInfer fusion, vLLM "coherency" can fail for many
reasons. This test isolates the core fused MoE op by:
  - Replaying a real captured call (inputs, routing, packed weights, quant_scales)
  - Running FlashInfer's CUTLASS fused MoE kernel
  - Building a BF16 reference by dequantizing MXFP4 weights + (optionally)
    dequantizing MXFP8 activations, then running an unfused PyTorch MoE.

Usage
-----
  python3 scripts/tests/test_flashinfer_fused_moe_replay_correctness.py \
    --capture /tmp/moe_call_*.pt [--weights /tmp/moe_weights.pt] \
    [--max-tokens 16]

Notes
-----
- This test is designed to run inside the dev container with:
    export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
- If the capture does not contain BF16 activations and the activation scale
  format is unknown, we fall back to a best-effort MXFP8 dequantization.
  The test will print what it assumed.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Prefer local flashinfer in container.
sys.path.insert(0, "/workspace/flashinfer")

from flashinfer.fused_moe.core import ActivationType, cutlass_fused_moe  # noqa: E402
from flashinfer import mxfp4_dequantize  # noqa: E402


@dataclass
class Metrics:
    cosine: float
    pearson: float
    max_abs: float
    mean_abs: float
    max_rel: float
    mean_rel: float


def _load_capture(capture_path: str, weights_path: str | None) -> Dict[str, Any]:
    data = torch.load(capture_path, map_location="cpu")
    if "fc1_expert_weights" not in data:
        if weights_path is None:
            weights_path = "/tmp/moe_weights.pt"
        weights = torch.load(weights_path, map_location="cpu")
        data.update(weights)
    return data


def _infer_dims_from_packed_weights(data: Dict[str, Any]) -> Tuple[int, int, int]:
    fc1 = data["fc1_expert_weights"]
    fc2 = data["fc2_expert_weights"]
    if fc1.dtype != torch.int64 or fc2.dtype != torch.int64:
        raise ValueError(
            f"Expected packed weights as int64 views; got "
            f"fc1={fc1.dtype} fc2={fc2.dtype}"
        )
    num_experts = int(fc1.shape[0])
    fc1_rows = int(fc1.shape[1])  # 2*intermediate
    if fc1_rows % 2 != 0:
        raise ValueError(f"fc1 rows (2*intermediate) must be even, got {fc1_rows}")
    intermediate = fc1_rows // 2
    hidden_size = int(fc1.shape[2]) * 16  # int64 packs 8 bytes, fp4 packs 2 vals/byte

    # Cross-check against fc2 packed layout.
    hidden2 = int(fc2.shape[1])
    inter2 = int(fc2.shape[2]) * 16
    if hidden2 != hidden_size or inter2 != intermediate:
        raise ValueError(
            f"Inferred dims disagree: from fc1 hidden={hidden_size} inter={intermediate}, "
            f"from fc2 hidden={hidden2} inter={inter2}"
        )
    return num_experts, hidden_size, intermediate


def _infer_topk(data: Dict[str, Any]) -> int:
    topk_ids = data["topk_ids"]
    if topk_ids.ndim != 2:
        raise ValueError(f"Expected topk_ids [T, K], got shape {tuple(topk_ids.shape)}")
    return int(topk_ids.shape[1])


def _unpack_mxfp4_weights_uint8(
    packed_w_i64: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    # packed_w_i64: [..., rows, cols/16] int64 view
    if packed_w_i64.dtype != torch.int64:
        raise ValueError(f"Expected int64 packed weights, got {packed_w_i64.dtype}")
    # View underlying bytes then reshape to fp4-packed bytes: [..., rows, cols/2]
    w_u8 = packed_w_i64.contiguous().view(torch.uint8)
    leading = list(packed_w_i64.shape[:-2])
    expected_last_i64 = cols // 16
    if packed_w_i64.shape[-1] != expected_last_i64:
        raise ValueError(
            f"Packed weight last dim mismatch: got {packed_w_i64.shape[-1]}, "
            f"expected {expected_last_i64} for cols={cols}"
        )
    w_u8 = w_u8.reshape(*leading, rows, cols // 2)
    return w_u8


def _unpack_mxfp4_scales_uint8(
    packed_sf_i32: torch.Tensor, rows: int, cols: int, group_size: int = 32
) -> torch.Tensor:
    # packed_sf_i32: [..., rows, cols/128] int32 view (because cols/32 uint8 grouped into int32)
    if packed_sf_i32.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"Unexpected scale dtype: {packed_sf_i32.dtype}")
    if packed_sf_i32.dtype == torch.uint8:
        # Already unpacked.
        sf_u8 = packed_sf_i32.contiguous()
        # Best effort: ensure last dim matches cols/group_size if possible.
        return sf_u8

    sf_u8 = packed_sf_i32.contiguous().view(torch.uint8)
    leading = list(packed_sf_i32.shape[:-2])
    expected_last_i32 = cols // (group_size * 4)
    if packed_sf_i32.shape[-1] != expected_last_i32:
        raise ValueError(
            f"Packed scale last dim mismatch: got {packed_sf_i32.shape[-1]}, "
            f"expected {expected_last_i32} for cols={cols}, group={group_size}"
        )
    sf_u8 = sf_u8.reshape(*leading, rows, cols // group_size)
    return sf_u8


def _best_effort_mxfp8_dequant(
    x_fp8: torch.Tensor, x_sf: torch.Tensor | None, hidden_size: int
) -> Tuple[torch.Tensor, str]:
    """
    Best-effort MXFP8 dequantization.

    Assumptions:
    - x_fp8 is a float8 tensor shaped [T, H] (or [T, *, H])
    - x_sf is shaped [T, H/G] (or broadcastable), where G is group size (usually 32)
    - x_sf is either a float scale, or an E8M0/UE8M0 exponent-coded uint8/int tensor.
    """
    if x_fp8.dtype == torch.bfloat16:
        return x_fp8, "input already bf16"
    if x_sf is None:
        # Last resort: just cast (this ignores scaling, but still provides a signal).
        return x_fp8.to(torch.bfloat16), "no input_sf; used bf16(x_fp8) cast"

    # Flatten to [T, H] for dequantization, then reshape back.
    orig_shape = x_fp8.shape
    if orig_shape[-1] != hidden_size:
        raise ValueError(
            f"Expected input last dim {hidden_size}, got {orig_shape[-1]}"
        )
    x2 = x_fp8.reshape(-1, hidden_size)
    sf = x_sf
    if sf.ndim == 1:
        # vLLM MoE captures may store input_sf flattened as [T * (H/32)].
        expected = x2.shape[0] * (hidden_size // 32)
        if sf.numel() == expected:
            sf = sf.reshape(x2.shape[0], hidden_size // 32)
        else:
            # Some other 1D scale shape (e.g. per-expert fake scales); ignore.
            return (
                x_fp8.to(torch.bfloat16),
                f"input_sf is 1D (numel={sf.numel()}); treated as absent and casted",
            )
    sf2 = sf.reshape(x2.shape[0], -1)
    group = hidden_size // sf2.shape[1]
    if hidden_size % sf2.shape[1] != 0:
        return x_fp8.to(torch.bfloat16), f"input_sf shape {tuple(sf.shape)} not compatible; casted"

    # Decode scale values.
    if sf2.dtype.is_floating_point:
        scale = sf2.to(torch.float32)
        scale_desc = f"float scale dtype={sf2.dtype}"
    else:
        # Treat as E8M0-like exponent with bias 127.
        bias = 127.0
        scale = torch.pow(2.0, sf2.to(torch.float32) - bias)
        scale_desc = f"exp-coded scale dtype={sf2.dtype} bias={bias}"

    x_block = x2.to(torch.float32).reshape(x2.shape[0], sf2.shape[1], group)
    x_deq = (x_block * scale.unsqueeze(-1)).reshape(x2.shape[0], hidden_size)
    return x_deq.to(torch.bfloat16).reshape(orig_shape), f"mxfp8 dequant (group={group}, {scale_desc})"


def _bf16_reference_moe(
    x_bf16: torch.Tensor,
    fc1_deq_bf16: torch.Tensor,
    fc2_deq_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: str = "swiglu",
) -> torch.Tensor:
    """
    Unfused BF16 MoE for a small token slice.
    Shapes:
      x_bf16: [T, H]
      fc1_deq_bf16: [E_used, 2I, H]
      fc2_deq_bf16: [E_used, H, I]
      topk_ids: [T, K] (original expert ids)
      topk_weights: [T, K] float
    """
    if activation != "swiglu":
        raise NotImplementedError("Only SwiGLU is supported in this reference")
    T, H = x_bf16.shape
    E_used, twoI, H2 = fc1_deq_bf16.shape
    if H2 != H:
        raise ValueError("Hidden mismatch in reference weights")
    I = twoI // 2
    if twoI != 2 * I:
        raise ValueError("FC1 rows must be 2*I")
    if fc2_deq_bf16.shape != (E_used, H, I):
        raise ValueError(f"FC2 shape mismatch: {tuple(fc2_deq_bf16.shape)} vs {(E_used, H, I)}")

    # Build mapping original expert id -> compact index into dequant arrays.
    used_ids = torch.unique(topk_ids.flatten()).tolist()
    id_to_compact = {int(e): i for i, e in enumerate(used_ids)}

    out = torch.zeros((T, H), device=x_bf16.device, dtype=torch.bfloat16)
    # IMPORTANT: keep weights in BF16 to avoid massive float32 copies for
    # real-model captures (can OOM). Accumulate in float32 for stability.
    x_bf16 = x_bf16.to(torch.bfloat16)
    fc1_bf16 = fc1_deq_bf16.to(torch.bfloat16)
    fc2_bf16 = fc2_deq_bf16.to(torch.bfloat16)

    for t in range(T):
        acc = torch.zeros((H,), device=x_bf16.device, dtype=torch.float32)
        for k in range(int(topk_ids.shape[1])):
            e = int(topk_ids[t, k].item())
            w = float(topk_weights[t, k].item())
            e_c = id_to_compact[e]

            # FC1: [H] x [H,2I] -> [2I]
            fc1 = torch.matmul(x_bf16[t], fc1_bf16[e_c].transpose(0, 1))  # BF16
            gate = fc1[:I].float()
            up = fc1[I:].float()
            hid = F.silu(gate) * up
            # FC2: [I] x [I,H] -> [H]
            fc2 = torch.matmul(hid.to(torch.bfloat16), fc2_bf16[e_c].transpose(0, 1)).float()
            acc += w * fc2
        out[t] = acc.to(torch.bfloat16)
    return out


def _compute_metrics(y: torch.Tensor, ref: torch.Tensor) -> Metrics:
    y_f = y.float().flatten()
    r_f = ref.float().flatten()
    cosine = F.cosine_similarity(y_f.unsqueeze(0), r_f.unsqueeze(0)).item()
    pearson = torch.corrcoef(torch.stack([y_f, r_f]))[0, 1].item()

    diff = (y.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    return Metrics(
        cosine=float(cosine),
        pearson=float(pearson),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float(rel.max().item()),
        mean_rel=float(rel.mean().item()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="Path to /tmp/moe_call_*.pt")
    ap.add_argument("--weights", default=None, help="Optional /tmp/moe_weights.pt for split captures")
    ap.add_argument("--max-tokens", type=int, default=16, help="Token slice size for reference (keep small)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = _load_capture(args.capture, args.weights)
    device = args.device

    # Infer dimensions from packed weights early (needed for slicing flattened input_sf).
    num_experts, hidden_size, intermediate = _infer_dims_from_packed_weights(data)
    top_k = _infer_topk(data)

    # Core tensors (names follow scripts/utils/replay_moe_call.py capture contract).
    x_in_full = data["fi_input"]
    topk_ids = data["topk_ids"]
    topk_w = data["topk_weights"]
    qscales: List[torch.Tensor] = list(data.get("quant_scales", []))
    x_sf = data.get("input_sf", None)
    use_act_scaling = bool(data.get("use_mxfp8_act_scaling", False))

    # Move to device.
    x_in_full = x_in_full.to(device)
    topk_ids = topk_ids.to(device)
    topk_w = topk_w.to(device)
    qscales = [t.to(device) for t in qscales]
    if x_sf is not None:
        x_sf = x_sf.to(device)

    # Slice to keep reference fast.
    T_total = int(x_in_full.shape[0])
    T = min(int(args.max_tokens), T_total)
    x_in = x_in_full[:T]
    topk_ids = topk_ids[:T]
    topk_w = topk_w[:T]
    if x_sf is not None:
        # If flattened input_sf = [T_total*(H/32)], slice it to match token slice.
        if x_sf.ndim == 1:
            per_tok = hidden_size // 32
            if x_sf.numel() == T_total * per_tok:
                x_sf = x_sf[: T * per_tok].contiguous()
        elif x_sf.ndim >= 2:
            x_sf = x_sf[:T]

    print("=== Replay correctness test (FlashInfer fused MoE) ===")
    print(f"capture: {args.capture}")
    print(f"T slice: {T}/{T_total}  top_k={top_k}")
    print(f"dims: experts={num_experts} hidden={hidden_size} intermediate={intermediate}")
    print(f"input: shape={tuple(x_in.shape)} dtype={x_in.dtype} use_mxfp8_act_scaling={use_act_scaling}")

    # Run fused kernel.
    fc1_w = data["fc1_expert_weights"].to(device)
    fc2_w = data["fc2_expert_weights"].to(device)
    fc1_b = data.get("fc1_expert_biases", None)
    fc2_b = data.get("fc2_expert_biases", None)
    if fc1_b is not None:
        fc1_b = fc1_b.to(device)
    if fc2_b is not None:
        fc2_b = fc2_b.to(device)

    extra_kwargs: Dict[str, Any] = {}
    if use_act_scaling:
        extra_kwargs["use_mxfp8_act_scaling"] = True
        extra_kwargs["input_sf"] = x_sf

    out = torch.empty((T, hidden_size), device=device, dtype=torch.bfloat16)
    y = cutlass_fused_moe(
        input=x_in,
        token_selected_experts=topk_ids.to(torch.int).contiguous(),
        token_final_scales=topk_w,
        fc1_expert_weights=fc1_w,
        fc2_expert_weights=fc2_w,
        output_dtype=torch.bfloat16,
        output=out,
        quant_scales=qscales,
        fc1_expert_biases=fc1_b,
        fc2_expert_biases=fc2_b,
        activation_type=ActivationType.Swiglu,
        **extra_kwargs,
    )[0]

    torch.cuda.synchronize()
    print(f"fused out: shape={tuple(y.shape)} dtype={y.dtype} nan={bool(torch.isnan(y).any())} inf={bool(torch.isinf(y).any())}")

    # Build BF16 reference input.
    x_bf16, x_desc = _best_effort_mxfp8_dequant(x_in, x_sf if use_act_scaling else None, hidden_size)
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(torch.bfloat16)
    print(f"reference input: {x_desc} -> dtype={x_bf16.dtype}")

    # Dequantize only experts actually used in this slice.
    used_experts = torch.unique(topk_ids.flatten().to(torch.int64)).tolist()
    used_experts_sorted = sorted(int(e) for e in used_experts)
    if any(e < 0 or e >= num_experts for e in used_experts_sorted):
        raise ValueError(f"Invalid expert id in routing: {used_experts_sorted[:16]} ...")
    print(f"used experts in slice: {len(used_experts_sorted)} (min={used_experts_sorted[0]} max={used_experts_sorted[-1]})")

    # Expect quant_scales layout: [fc1_scale, (maybe input), fc2_scale, (maybe input)]
    if len(qscales) < 3:
        raise ValueError(f"Expected quant_scales length >= 3, got {len(qscales)}")
    fc1_sf_i32 = qscales[0].index_select(0, torch.tensor(used_experts_sorted, device=device))
    fc2_sf_i32 = qscales[2].index_select(0, torch.tensor(used_experts_sorted, device=device))

    fc1_w_i64 = fc1_w.index_select(0, torch.tensor(used_experts_sorted, device=device))
    fc2_w_i64 = fc2_w.index_select(0, torch.tensor(used_experts_sorted, device=device))

    fc1_u8 = _unpack_mxfp4_weights_uint8(fc1_w_i64, rows=2 * intermediate, cols=hidden_size)
    fc2_u8 = _unpack_mxfp4_weights_uint8(fc2_w_i64, rows=hidden_size, cols=intermediate)
    fc1_sf_u8 = _unpack_mxfp4_scales_uint8(fc1_sf_i32, rows=2 * intermediate, cols=hidden_size, group_size=32)
    fc2_sf_u8 = _unpack_mxfp4_scales_uint8(fc2_sf_i32, rows=hidden_size, cols=intermediate, group_size=32)

    # Dequantize (FlashInfer implementation, but unfused reference math).
    # FlashInfer dequant expects 2D inputs; flatten (E_used * rows, cols/2) then reshape back.
    e_used = fc1_u8.shape[0]
    fc1_deq_2d = mxfp4_dequantize(
        fc1_u8.reshape(e_used * (2 * intermediate), hidden_size // 2),
        fc1_sf_u8.reshape(e_used * (2 * intermediate), hidden_size // 32),
    )
    fc2_deq_2d = mxfp4_dequantize(
        fc2_u8.reshape(e_used * hidden_size, intermediate // 2),
        fc2_sf_u8.reshape(e_used * hidden_size, intermediate // 32),
    )
    # mxfp4_dequantize may return CPU tensors depending on build; move to `device`.
    fc1_deq = (
        fc1_deq_2d.reshape(e_used, 2 * intermediate, hidden_size)
        .to(device=device, dtype=torch.bfloat16)
    )
    fc2_deq = (
        fc2_deq_2d.reshape(e_used, hidden_size, intermediate)
        .to(device=device, dtype=torch.bfloat16)
    )
    print(
        f"dequant weights: fc1={tuple(fc1_deq.shape)} fc2={tuple(fc2_deq.shape)} "
        f"(E_used={e_used})"
    )

    ref = _bf16_reference_moe(
        x_bf16=x_bf16,
        fc1_deq_bf16=fc1_deq,
        fc2_deq_bf16=fc2_deq,
        topk_ids=topk_ids,
        topk_weights=topk_w,
    )

    m = _compute_metrics(y, ref)
    print("=== metrics (fused vs BF16 dequant reference) ===")
    print(f"cosine: {m.cosine:.4f}  pearson: {m.pearson:.4f}")
    print(f"abs err: max={m.max_abs:.4f} mean={m.mean_abs:.4f}")
    print(f"rel err: max={m.max_rel:.2%} mean={m.mean_rel:.2%}")

    # Thresholds: FP8×FP4 is noisy, but should still correlate.
    # If fusion is broken, cosine/pearson will typically collapse.
    ok = (
        not torch.isnan(y).any()
        and not torch.isinf(y).any()
        and m.cosine > 0.80
        and m.pearson > 0.80
    )
    if ok:
        print("PASS")
        raise SystemExit(0)
    print("FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

