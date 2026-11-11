"""Plain PyTorch references. These match Hugging Face Qwen2 numerics, which the kernels copy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm_ref(x, weight, eps=1e-6, residual=None):
    """Qwen2RMSNorm, optionally preceded by the residual add from the decoder layer.

    Returns `out`, or `(out, x + residual)` when `residual` is given.
    """
    if residual is not None:
        x = x + residual  # fp16 add, as in the HF decoder layer
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    out = weight * (h * torch.rsqrt(var + eps)).to(x.dtype)
    return out if residual is None else (out, x)


def swiglu_ref(gate, up):
    """Qwen2MLP activation: silu(gate) * up, computed the way HF does it in fp16."""
    return F.silu(gate) * up


def dequantize_ref(qweight, scales, zeros, group_size=128):
    """Unpack int4 weights to fp16 with layout [N, K] (same as nn.Linear.weight)."""
    from kernels.quant import unpack_int4

    q = unpack_int4(qweight)  # [K, N] int32 in 0..15
    K, N = q.shape
    s = scales.repeat_interleave(group_size, dim=0)[:K]
    z = zeros.repeat_interleave(group_size, dim=0)[:K]
    w = (q.to(scales.dtype) - z) * s  # one fp16 rounding, same as the kernels
    return w.t().contiguous()


def w4a16_ref(x, qweight, scales, zeros, group_size=128, bias=None):
    """fp32 reference for the W4A16 matmul: x @ dequant(W).T (+ bias)."""
    w = dequantize_ref(qweight, scales, zeros, group_size)
    out = x.float() @ w.float().t()
    if bias is not None:
        out = out + bias.float()
    return out
