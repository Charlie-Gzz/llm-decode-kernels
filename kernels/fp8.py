"""Stretch: FP8 weight-only matmul (W8A16) in Triton. y = x @ (Wq * s).T

Weights are float8 e4m3 with one fp16 scale per output channel, stored as [K, N] (N contiguous,
like the int4 format). The scale is constant along K, so it factors out of the dot product:
the inner loop only converts fp8 to fp16 and multiplies, and the scale is applied once at the end.
Half the bytes of fp16 (int4 is a quarter), with much smaller quality loss than 4-bit RTN.

Paths (all autotuned, chosen per decode shape like W4A16): tensor-core GEMM, split-K, and a
batch-1 GEMV without tensor cores.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_fp8(weight: torch.Tensor):
    """[N, K] fp16 -> (wq float8 [K, N], scale fp16 [N]) with per-output-channel absmax scaling."""
    w = weight.float()
    scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    wq = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return wq.t().contiguous(), scale.to(weight.dtype)


def dequantize_fp8(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """-> [N, K] in scale's dtype."""
    return (wq.t().float() * scale.float()[:, None]).to(scale.dtype)


def fp8_bytes(N: int, K: int) -> int:
    return N * K + N * 2


@triton.jit
def _w8a16_kernel(
    a_ptr, w_ptr, s_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
    M_KEY,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_m = group_id * GROUP_M
    group_rows = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_rows)
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_rows

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    t_start = pid_k * tiles_per_split
    t_end = min(t_start + tiles_per_split, k_tiles)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for t in range(t_start, t_end):
        offs_k = t * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w.to(tl.float16), acc)

    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc * s[None, :]
    if HAS_BIAS:
        if pid_k == 0:
            acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask, sem="relaxed")


@triton.jit
def _w8a16_gemv_kernel(
    a_ptr, w_ptr, s_ptr, bias_ptr, c_ptr,
    N, K, stride_wk, stride_wn,
    M_KEY,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """Batch 1: FMAs into a [BLOCK_K, BLOCK_N] fp32 accumulator, one reduction at the end."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    t_start = pid_k * tiles_per_split
    t_end = min(t_start + tiles_per_split, k_tiles)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for t in range(t_start, t_end):
        offs_k = t * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += a[:, None] * w.to(tl.float32)
    out = tl.sum(acc, axis=0) * tl.load(s_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if HAS_BIAS:
        if pid_k == 0:
            out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, out, mask=mask_n, sem="relaxed")


def _cfg(bm, bn, bk, split, w, s):
    return triton.Config(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, GROUP_M=8),
                         num_warps=w, num_stages=s)


def _gcfg(bn, bk, split, w, s):
    return triton.Config(dict(BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split), num_warps=w, num_stages=s)


DECODE = [_cfg(16, bn, bk, 1, w, s) for bn in (32, 64, 128) for bk in (64, 128)
          for w, s in ((2, 3), (4, 4))]
SPLIT = [_cfg(16, bn, bk, sp, w, s) for bn in (64, 128) for bk in (64, 128) for sp in (2, 4, 8)
         for w, s in ((2, 3), (4, 4))]
PREFILL = [_cfg(bm, bn, bk, 1, w, s) for bm, bn, bk, w, s in [
    (32, 64, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4), (64, 128, 128, 8, 3),
    (128, 64, 64, 4, 4), (128, 128, 64, 8, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3)]]
GEMV = [_gcfg(bn, bk, 1, w, s) for bn in (32, 64, 128) for bk in (64, 128)
        for w, s in ((2, 3), (4, 3), (4, 4))]
GEMV_SPLIT = [_gcfg(bn, bk, sp, w, s) for bn in (64, 128) for bk in (64, 128) for sp in (2, 4, 8)
              for w, s in ((2, 3), (4, 3))]
KEY = ["M_KEY", "N", "K", "HAS_BIAS"]
_t_decode = triton.autotune(DECODE, key=KEY, cache_results=True)(_w8a16_kernel)
_t_prefill = triton.autotune(PREFILL, key=KEY, cache_results=True)(_w8a16_kernel)
_t_split = triton.autotune(SPLIT, key=KEY, reset_to_zero=["c_ptr"], cache_results=True)(
    _w8a16_kernel)
_t_gemv = triton.autotune(GEMV, key=KEY, cache_results=True)(_w8a16_gemv_kernel)
_t_gemv_split = triton.autotune(GEMV_SPLIT, key=KEY, reset_to_zero=["c_ptr"],
                                cache_results=True)(_w8a16_gemv_kernel)
INTERPRET_CONFIG = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, SPLIT_K=1, GROUP_M=8)
_choice: dict[tuple, str] = {}


def _run(path, x2, wq, scale, bias, m_key, config=None):
    M, K = x2.shape
    N = wq.shape[1]
    split = path in ("split", "gemv_split") or bool(config and config["SPLIT_K"] > 1)
    c = (torch.zeros if split else torch.empty)(
        (M, N), dtype=torch.float32 if split else x2.dtype, device=x2.device)
    b = bias if bias is not None else scale
    if path.startswith("gemv"):
        kernel = _w8a16_gemv_kernel if config else (_t_gemv_split if split else _t_gemv)
        kernel[lambda m: (triton.cdiv(N, m["BLOCK_N"]), m["SPLIT_K"])](
            x2, wq, scale, b, c, N, K, wq.stride(0), wq.stride(1), m_key,
            HAS_BIAS=bias is not None, **(config or {}))
    else:
        if config:
            kernel = _w8a16_kernel
        else:
            kernel = {"split": _t_split, "direct": _t_decode if m_key <= 16 else _t_prefill}[path]
        kernel[lambda m: (triton.cdiv(M, m["BLOCK_M"]) * triton.cdiv(N, m["BLOCK_N"]),
                          m["SPLIT_K"])](
            x2, wq, scale, b, c, M, N, K, x2.stride(0), x2.stride(1), wq.stride(0), wq.stride(1),
            c.stride(0), c.stride(1), m_key, HAS_BIAS=bias is not None, **(config or {}))
    return c.to(x2.dtype) if split else c


def _pick(args, key) -> str:
    from kernels.bench.harness import time_rotating

    names = ["direct", "split"] + (["gemv", "gemv_split"] if args[0].shape[0] == 1 else [])
    times = {}
    for name in names:
        def fn(x2, wq, scale, name=name):
            return _run(name, x2, wq, scale, args[3], args[4])
        fn(*args[:3])
        times[name] = time_rotating(fn, args[:3], rotate=(0, 1, 2))
    _choice[key] = min(times, key=times.get)
    return _choice[key]


def w8a16_matmul(x, wq, scale, bias=None, config=None):
    """y = x @ (wq * scale).T (+ bias). x: [..., K] fp16, wq: float8 [K, N], scale: fp16 [N]."""
    K = x.shape[-1]
    N = wq.shape[1]
    x2 = x.reshape(-1, K)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    m_key = min(triton.next_power_of_2(M), 4096)
    if config is None and INTERPRET:
        config = INTERPRET_CONFIG
    if config is not None:
        path = "gemv" if config.get("GEMV") else "direct"
        cfg = {k: v for k, v in config.items() if k != "GEMV"}
        if path == "gemv":
            cfg = {k: cfg[k] for k in ("BLOCK_N", "BLOCK_K", "SPLIT_K")}
        out = _run(path, x2, wq, scale, bias, m_key, cfg)
    elif m_key <= 16:
        key = (m_key, N, K, bias is not None)
        path = _choice.get(key) or _pick((x2, wq, scale, bias, m_key), key)
        out = _run(path, x2, wq, scale, bias, m_key)
    else:
        out = _run("direct", x2, wq, scale, bias, m_key)
    return out.view(*x.shape[:-1], N)
