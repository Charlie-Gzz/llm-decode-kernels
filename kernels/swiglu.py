"""Fused SwiGLU activation in Triton: out = silu(gate) * up, in one pass.

Accepts either two tensors or one fused [.., 2*I] tensor (gate first, then up), which is what a
merged gate/up projection produces. The fused form saves a split/copy in the model.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    g_ptr, u_ptr, out_ptr,
    stride_g, stride_u, stride_out,
    I,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(g_ptr + row * stride_g + cols, mask=mask, other=0.0)
    u = tl.load(u_ptr + row * stride_u + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    act = (gf * tl.sigmoid(gf)).to(g.dtype)  # silu in fp32, rounded like F.silu
    tl.store(out_ptr + row * stride_out + cols, act * u, mask=mask)


BLOCK = 1024


def _launch(gate2: torch.Tensor, up2: torch.Tensor) -> torch.Tensor:
    M, I = gate2.shape
    assert gate2.stride(-1) == 1 and up2.stride(-1) == 1
    out = torch.empty((M, I), dtype=gate2.dtype, device=gate2.device)
    grid = (M, triton.cdiv(I, BLOCK))
    _swiglu_kernel[grid](
        gate2, up2, out,
        gate2.stride(0), up2.stride(0), out.stride(0),
        I,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up for two same-shape tensors."""
    I = gate.shape[-1]
    out = _launch(gate.reshape(-1, I), up.reshape(-1, I))
    return out.view(gate.shape)


def swiglu_fused(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up where gate_up = cat([gate, up], dim=-1). Reads the halves in place."""
    I = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * I)
    out = _launch(gu[:, :I], gu[:, I:])
    return out.view(*gate_up.shape[:-1], I)


def swiglu_bytes(M: int, I: int, elem: int = 2) -> int:
    return 3 * M * I * elem
