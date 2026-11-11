"""Fused RMSNorm (+ residual add) in Triton.

One program normalizes one row. With a residual, the kernel also writes `x + residual`, which the
next decoder block needs as its residual stream. Unfused, that is 2 kernels and 2 extra trips to
memory; fused, every element is read once and written once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
    stride_x, stride_res, stride_out, stride_res_out,
    N, eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0)
        x = x + r  # fp16 add, same rounding as the unfused model
        tl.store(res_out_ptr + row * stride_res_out + cols, x, mask=mask)

    h = x.to(tl.float32)
    var = tl.sum(h * h, axis=0) / N
    normed = (h * tl.rsqrt(var + eps)).to(x.dtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride_out + cols, normed * w, mask=mask)


def _num_warps(block: int) -> int:
    return max(1, min(16, block // 256))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual=None):
    """RMSNorm over the last dim. Returns `out`, or `(out, x + residual)` with a residual."""
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)
    assert x2.stride(-1) == 1 and weight.is_contiguous()
    out = torch.empty_like(x2)
    if residual is not None:
        r2 = residual.reshape(-1, N)
        res_out = torch.empty_like(x2)
    else:
        r2, res_out = x2, out  # unused placeholders

    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, weight, out, res_out,
        x2.stride(0), r2.stride(0), out.stride(0), res_out.stride(0),
        N, eps,
        HAS_RESIDUAL=residual is not None,
        BLOCK=BLOCK,
        num_warps=_num_warps(BLOCK),
    )
    if residual is None:
        return out.view(shape)
    return out.view(shape), res_out.view(shape)


def rmsnorm_bytes(M: int, N: int, residual: bool, elem: int = 2) -> int:
    """Minimum DRAM traffic: read x (+ residual) and weight, write out (+ new residual)."""
    tensors = 4 if residual else 2
    return tensors * M * N * elem + N * elem
