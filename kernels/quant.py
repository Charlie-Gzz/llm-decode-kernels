"""4-bit weight format shared by every kernel in this repo.

For a linear layer with weight W of shape [N, K] (out_features, in_features):

- `qweight`: int32 [K // 8, N]. Each int32 holds 8 consecutive K values of one output column,
  value i in bits 4i..4i+3 (unsigned, 0..15). N is the contiguous dimension, so a warp reading
  one packed row loads consecutive output columns (coalesced).
- `scales`, `zeros`: fp16 [K // group_size, N]. Dequantized weight: (q - zero) * scale.

Quantization is plain round-to-nearest (RTN), asymmetric, group size 128 along K.
"""

from __future__ import annotations

import torch

PACK = 8  # int4 values per int32


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """[K, N] integers in 0..15 -> [K // 8, N] int32."""
    K, N = q.shape
    assert K % PACK == 0, "K must be a multiple of 8"
    q = q.to(torch.int64).reshape(K // PACK, PACK, N)
    shifts = (torch.arange(PACK, device=q.device) * 4).view(1, PACK, 1)
    packed = (q << shifts).sum(dim=1)  # < 2**32, fits in int64
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)  # reinterpret as int32
    return packed.to(torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """[K // 8, N] int32 -> [K, N] int32 in 0..15."""
    shifts = (torch.arange(PACK, device=packed.device, dtype=torch.int32) * 4).view(1, PACK, 1)
    q = (packed.unsqueeze(1) >> shifts) & 0xF
    return q.reshape(-1, packed.shape[1])


def quantize_rtn(weight: torch.Tensor, group_size: int = 128):
    """Round-to-nearest asymmetric int4 quantization of an [N, K] fp16 weight."""
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be a multiple of group_size={group_size}"
    w = weight.t().float().reshape(K // group_size, group_size, N)
    wmin = w.amin(dim=1)
    wmax = w.amax(dim=1)
    scale = ((wmax - wmin) / 15).clamp(min=1e-6)
    zero = (-wmin / scale).round().clamp(0, 15)
    q = (w / scale.unsqueeze(1) + zero.unsqueeze(1)).round().clamp(0, 15)
    qweight = pack_int4(q.reshape(K, N).to(torch.int32))
    return qweight, scale.to(weight.dtype), zero.to(weight.dtype)


def weight_bytes(N: int, K: int, group_size: int = 128) -> int:
    """Bytes a W4A16 kernel must read for the weight: packed int4 + fp16 scales and zeros."""
    return N * K // 2 + 2 * (K // group_size) * N * 2
