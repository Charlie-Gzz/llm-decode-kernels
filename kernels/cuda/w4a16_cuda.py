"""Builds and wraps the CUDA C++ W4A16 decode kernel (torch.utils.cpp_extension, JIT-compiled once
and cached in TORCH_EXTENSIONS_DIR, which lives on the Modal Volume)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

SRC = Path(__file__).with_name("w4a16_decode.cu")
SPLITS = (1, 2, 4, 8, 16)
_split_choice: dict[tuple, int] = {}


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    return load(
        name="w4a16_decode_ext",
        sources=[str(SRC)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _pick_split(x2, qweight, scales, zeros, bias, group_size, key) -> int:
    """Time each split-K factor once per shape and keep the fastest (same idea as Triton's
    autotuner, including the zero-fill and cast that split-K needs)."""
    import triton

    times = {}
    for split in SPLITS:
        def call(split=split):
            return _ext().w4a16_decode(x2, qweight, scales, zeros, bias, group_size, split)
        times[split] = triton.testing.do_bench(call, warmup=25, rep=100)
    _split_choice[key] = min(times, key=times.get)
    return _split_choice[key]


def w4a16_decode_cuda(x, qweight, scales, zeros, group_size=128, bias=None, split_k=None):
    """y = x @ dequant(W).T for M <= 16 rows. Same contract as kernels.w4a16.w4a16_matmul."""
    K = x.shape[-1]
    N = qweight.shape[1]
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    if split_k is None:
        key = (min(1 << (M - 1).bit_length(), 16), N, K, bias is not None)
        split_k = _split_choice.get(key) or _pick_split(x2, qweight, scales, zeros, bias,
                                                         group_size, key)
    out = _ext().w4a16_decode(x2, qweight, scales, zeros, bias, group_size, split_k)
    return out.view(*x.shape[:-1], N)


def split_choices() -> dict:
    return {str(k): v for k, v in _split_choice.items()}


def export_choices() -> dict:
    return {",".join(map(str, k)): v for k, v in _split_choice.items()}


def import_choices(choices: dict) -> None:
    for k, v in choices.items():
        m, n, kk, b = k.split(",")
        _split_choice[(int(m), int(n), int(kk), b == "True")] = v


if __name__ == "__main__":
    print(torch.__version__, _ext())
