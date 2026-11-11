"""W4A16 matmul in Triton: y = x @ dequant(W).T, int4 weights, fp16 activations.

Weight format: see kernels/quant.py (int32 [K/8, N], fp16 scales/zeros [K/128, N]).

Each program computes a [BLOCK_M, BLOCK_N] output tile. For every K tile it loads the packed int32
tile [BLOCK_K/8, BLOCK_N], unpacks it to [BLOCK_K, BLOCK_N] in registers, applies that group's
scale and zero, and feeds tensor cores with tl.dot. The fp16 weight never touches DRAM, which is
the point: decode is memory-bound, and 4-bit weights are 4x fewer bytes than fp16.

Two launch paths:
- SPLIT_K == 1: plain tiled GEMM, writes fp16 directly. Used for prefill.
- SPLIT_K > 1: for decode (M <= 16) there are too few output tiles to fill the GPU, so K is split
  across programs that atomically add fp32 partials into a zeroed buffer, then cast to fp16.

Both paths are autotuned. For decode, the wrapper times the best config of each path once per
shape (including the zero-fill and cast that split-K needs) and keeps the faster one.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _w4a16_kernel(
    a_ptr, q_ptr, s_ptr, z_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    M_KEY,  # only used as an autotune key (M rounded up to a power of 2)
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Grouped tile order so neighbouring programs share A rows / B columns in L2 (prefill).
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
    offs_pk = tl.arange(0, BLOCK_K // 8)  # packed rows in one K tile
    shifts = (tl.arange(0, 8) * 4)[None, :, None]

    # This program's K range, aligned to BLOCK_K (BLOCK_K divides GROUP_SIZE divides K).
    k_tiles = K // BLOCK_K
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    t_start = pid_k * tiles_per_split
    t_end = min(t_start + tiles_per_split, k_tiles)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for t in range(t_start, t_end):
        k0 = t * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None], other=0.0,
        )
        packed = tl.load(
            q_ptr + (k0 // 8 + offs_pk)[:, None] * stride_qk + offs_n[None, :] * stride_qn,
            mask=mask_n[None, :], other=0,
        )
        # [BK/8, BN] int32 -> [BK/8, 8, BN] nibbles -> [BK, BN], row k = 8 * packed_row + i
        q = (packed[:, None, :] >> shifts) & 0xF
        q = tl.reshape(q, (BLOCK_K, BLOCK_N))
        g = k0 // GROUP_SIZE
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        b = (q.to(tl.float16) - z[None, :]) * s[None, :]
        acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        if pid_k == 0:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += bias.to(tl.float32)[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask, sem="relaxed")


def _cfg(bm, bn, bk, split, warps, stages, group_m=8):
    return triton.Config(
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, GROUP_M=group_m),
        num_warps=warps, num_stages=stages,
    )


# Decode (M <= 16): BLOCK_M = 16 is the smallest tl.dot tile. Small BLOCK_N gives more programs.
DECODE_CONFIGS = [
    _cfg(16, bn, bk, 1, w, s)
    for bn in (32, 64, 128) for bk in (64, 128) for w in (2, 4) for s in (3, 4)
]
SPLITK_CONFIGS = [
    _cfg(16, bn, bk, split, w, s)
    for bn in (32, 64, 128) for bk in (64, 128) for split in (2, 4, 8) for w, s in ((2, 3), (4, 4))
]
PREFILL_CONFIGS = [
    _cfg(bm, bn, bk, 1, w, s)
    for bm, bn, bk, w, s in [
        (16, 64, 128, 4, 4), (32, 64, 128, 4, 4), (32, 128, 64, 4, 4), (64, 64, 128, 4, 4),
        (64, 128, 64, 4, 4), (64, 128, 128, 8, 3), (128, 64, 64, 4, 4), (128, 128, 64, 8, 3),
        (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (64, 256, 64, 8, 3), (32, 256, 64, 4, 4),
    ]
]
# One fixed config per path for the CPU interpreter, which cannot autotune.
INTERPRET_CONFIG = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, SPLIT_K=1, GROUP_M=8)

KEY = ["M_KEY", "N", "K", "HAS_BIAS"]
_tuned_decode = triton.autotune(DECODE_CONFIGS, key=KEY)(_w4a16_kernel)
_tuned_prefill = triton.autotune(PREFILL_CONFIGS, key=KEY)(_w4a16_kernel)
_tuned_split = triton.autotune(SPLITK_CONFIGS, key=KEY, reset_to_zero=["c_ptr"])(_w4a16_kernel)
_decode_choice: dict[tuple, str] = {}  # (M_KEY, N, K, bias) -> "direct" | "split"


def _launch(kernel, x2, qweight, scales, zeros, bias, out, m_key, group_size, config=None):
    M, K = x2.shape
    N = qweight.shape[1]

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

    kernel[grid](
        x2, qweight, scales, zeros, bias if bias is not None else scales, out,
        M, N, K,
        x2.stride(0), x2.stride(1), qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1), out.stride(0), out.stride(1),
        m_key,
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        **(config or {}),
    )


def _run_direct(x2, qweight, scales, zeros, bias, m_key, group_size, config=None):
    out = torch.empty((x2.shape[0], qweight.shape[1]), dtype=x2.dtype, device=x2.device)
    kernel = _w4a16_kernel if config else (_tuned_decode if m_key <= 16 else _tuned_prefill)
    _launch(kernel, x2, qweight, scales, zeros, bias, out, m_key, group_size, config)
    return out


def _run_split(x2, qweight, scales, zeros, bias, m_key, group_size, config=None):
    acc = torch.zeros((x2.shape[0], qweight.shape[1]), dtype=torch.float32, device=x2.device)
    kernel = _w4a16_kernel if config else _tuned_split
    _launch(kernel, x2, qweight, scales, zeros, bias, acc, m_key, group_size, config)
    return acc.to(x2.dtype)


def _pick_decode_path(args, key) -> str:
    """Time the best direct and best split-K config once per shape, keep the faster."""
    times = {}
    for name, fn in (("direct", _run_direct), ("split", _run_split)):
        fn(*args)  # runs the autotuner for this path
        times[name] = triton.testing.do_bench(lambda fn=fn: fn(*args), warmup=10, rep=40)
    _decode_choice[key] = min(times, key=times.get)
    return _decode_choice[key]


def w4a16_matmul(x, qweight, scales, zeros, group_size=128, bias=None, config=None):
    """y = x @ dequant(W).T (+ bias). x: [..., K] fp16. Returns [..., N] in x's dtype.

    `config` (dict of BLOCK_M/BLOCK_N/BLOCK_K/SPLIT_K/GROUP_M) skips autotuning; tests use it to
    exercise both paths, and the CPU interpreter always needs it.
    """
    K = x.shape[-1]
    N = qweight.shape[1]
    assert qweight.shape[0] * 8 == K and K % group_size == 0
    x2 = x.reshape(-1, K)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    m_key = min(triton.next_power_of_2(M), 4096)
    args = (x2, qweight, scales, zeros, bias, m_key, group_size)

    if config is None and INTERPRET:
        config = INTERPRET_CONFIG
    if config is not None:
        run = _run_split if config["SPLIT_K"] > 1 else _run_direct
        out = run(*args, config)
    elif m_key <= 16:
        key = (m_key, N, K, bias is not None)
        path = _decode_choice.get(key) or _pick_decode_path(args, key)
        out = (_run_split if path == "split" else _run_direct)(*args)
    else:
        out = _run_direct(*args)
    return out.view(*x.shape[:-1], N)


def best_configs() -> dict:
    """Autotuner choices so far, for the results JSON."""
    out = {}
    tuners = (("decode", _tuned_decode), ("prefill", _tuned_prefill), ("split", _tuned_split))
    for name, tuner in tuners:
        for key, cfg in tuner.cache.items():
            out[f"{name}{list(key)[:4]}"] = {**cfg.kwargs, "warps": cfg.num_warps,
                                            "stages": cfg.num_stages}
    out["decode_path"] = {str(k): v for k, v in _decode_choice.items()}
    return out
