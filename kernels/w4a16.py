"""W4A16 matmul in Triton: y = x @ dequant(W).T, int4 weights, fp16 activations.

Weight format: see kernels/quant.py (int32 [K/8, N], fp16 scales/zeros [K/128, N]).

Each program computes a [BLOCK_M, BLOCK_N] output tile. For every K tile it loads the packed int32
tile [BLOCK_K/8, BLOCK_N], unpacks it to [BLOCK_K, BLOCK_N] in registers, applies that group's
scale and zero, and feeds tensor cores with tl.dot. The fp16 weight never touches DRAM, which is
the point: decode is memory-bound, and 4-bit weights are 4x fewer bytes than fp16.

Launch paths, all autotuned:
- direct (SPLIT_K == 1): plain tiled GEMM, writes fp16 directly. The only path for prefill.
- split: for decode (M <= 16) there are too few output tiles to fill the GPU, so K is split
  across programs that atomically add fp32 partials into a zeroed buffer, then cast to fp16.
- gemv / gemv_split: M == 1 only. No tensor cores, plain FMAs (see _w4a16_gemv_kernel).

For decode, the wrapper times the best config of each path once per shape (including the
zero-fill and cast that split-K needs) and keeps the fastest.
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


@triton.jit
def _w4a16_gemv_kernel(
    a_ptr, q_ptr, s_ptr, z_ptr, bias_ptr, c_ptr,
    N, K,
    stride_qk, stride_qn, stride_sg, stride_sn,
    M_KEY,  # autotune key only
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    FACTORED: tl.constexpr = False,
):
    """Batch-1 decode (M == 1) without tensor cores. tl.dot needs 16 rows, so the main kernel
    wastes 15/16 of its MMA work at M=1 and round-trips the dequantized tile through shared
    memory. Here each program does plain FMAs into a [BLOCK_K, BLOCK_N] fp32 accumulator and
    reduces over K once at the end.

    FACTORED: scale and zero are constant inside a group, so sum_k a*(q-z)*s equals
    s * (sum_k a*q - z * sum_k a). The inner loop is then shift, mask, convert, one FMA per
    weight (4 ops instead of 7), with the scale applied once per tile. Less ALU work matters on
    the power-limited L4. Numerics: fp32 throughout, no fp16 rounding of the weight."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_pk = tl.arange(0, BLOCK_K // 8)
    shifts = (tl.arange(0, 8) * 4)[None, :, None]

    k_tiles = K // BLOCK_K
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    t_start = pid_k * tiles_per_split
    t_end = min(t_start + tiles_per_split, k_tiles)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    out = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for t in range(t_start, t_end):
        k0 = t * BLOCK_K
        a = tl.load(a_ptr + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
        packed = tl.load(
            q_ptr + (k0 // 8 + offs_pk)[:, None] * stride_qk + offs_n[None, :] * stride_qn,
            mask=mask_n[None, :], other=0,
        )
        q = tl.reshape((packed[:, None, :] >> shifts) & 0xF, (BLOCK_K, BLOCK_N))
        g = k0 // GROUP_SIZE
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        if FACTORED:
            aq = tl.sum(a[:, None] * q.to(tl.float32), axis=0)
            out += s.to(tl.float32) * (aq - z.to(tl.float32) * tl.sum(a, axis=0))
        else:
            w = (q.to(tl.float16) - z[None, :]) * s[None, :]  # same fp16 rounding as reference
            acc += a[:, None] * w.to(tl.float32)
    if not FACTORED:
        out = tl.sum(acc, axis=0)

    if HAS_BIAS:
        if pid_k == 0:
            out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, out, mask=mask_n, sem="relaxed")


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
_tuned_decode = triton.autotune(DECODE_CONFIGS, key=KEY, cache_results=True)(_w4a16_kernel)
_tuned_prefill = triton.autotune(PREFILL_CONFIGS, key=KEY, cache_results=True)(_w4a16_kernel)
_tuned_split = triton.autotune(SPLITK_CONFIGS, key=KEY, reset_to_zero=["c_ptr"],
                               cache_results=True)(_w4a16_kernel)
_decode_choice: dict[tuple, str] = {}  # (M_KEY, N, K, bias) -> path name


def _gcfg(bn, bk, split, warps, stages, factored):
    return triton.Config(dict(BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, FACTORED=factored),
                         num_warps=warps, num_stages=stages)


GEMV_WS = ((2, 3), (4, 3), (4, 4))
GEMV_CONFIGS = [_gcfg(bn, bk, 1, w, s, f) for bn in (32, 64, 128) for bk in (64, 128)
                for w, s in GEMV_WS for f in (False, True)]
GEMV_SPLIT_CONFIGS = [_gcfg(bn, bk, sp, w, s, f) for bn in (64, 128) for bk in (64, 128)
                      for sp in (2, 4, 8) for w, s in ((2, 3), (4, 3)) for f in (False, True)]
GEMV_KEY = ["M_KEY", "N", "K", "HAS_BIAS"]
_tuned_gemv = triton.autotune(GEMV_CONFIGS, key=GEMV_KEY, cache_results=True)(_w4a16_gemv_kernel)
_tuned_gemv_split = triton.autotune(GEMV_SPLIT_CONFIGS, key=GEMV_KEY, reset_to_zero=["c_ptr"],
                                    cache_results=True)(_w4a16_gemv_kernel)


def _run_gemv(x2, qweight, scales, zeros, bias, m_key, group_size, config=None, split=False):
    assert x2.shape[0] == 1 and x2.stride(1) == 1
    K, N = x2.shape[1], qweight.shape[1]
    split = split or bool(config and config["SPLIT_K"] > 1)
    out = torch.zeros if split else torch.empty
    c = out((1, N), dtype=torch.float32 if split else x2.dtype, device=x2.device)
    if config:
        kernel = _w4a16_gemv_kernel
    else:
        kernel = _tuned_gemv_split if split else _tuned_gemv

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])

    kernel[grid](
        x2, qweight, scales, zeros, bias if bias is not None else scales, c,
        N, K, qweight.stride(0), qweight.stride(1), scales.stride(0), scales.stride(1),
        m_key, GROUP_SIZE=group_size, HAS_BIAS=bias is not None,
        **({k: config[k] for k in ("BLOCK_N", "BLOCK_K", "SPLIT_K", "FACTORED") if k in config}
           if config else {}),
    )
    return c.to(x2.dtype) if split else c


def _run_gemv_split(*args):
    return _run_gemv(*args, split=True)


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


PATHS = {"direct": _run_direct, "split": _run_split, "gemv": _run_gemv,
         "gemv_split": _run_gemv_split}


def _pick_decode_path(args, key) -> str:
    """Time the best config of each path once per shape (including the zero-fill and cast that
    split-K needs) and keep the fastest. GEMV paths only exist for M == 1.

    Timed like the benchmark: CUDA graph, rotating weight copies. The first version used eager
    do_bench, where Python launch overhead hid the kernel differences and biased the choice
    toward paths with fewer launches (8b_up at batch 1 got 67% instead of ~81% of peak)."""
    from kernels.bench.harness import time_rotating

    names = ["direct", "split"] + (["gemv", "gemv_split"] if args[0].shape[0] == 1 else [])
    times = {}
    for name in names:
        fn = PATHS[name]
        fn(*args)  # runs the autotuner for this path
        times[name] = time_rotating(fn, args, rotate=(0, 1, 2, 3))
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
        if config.get("GEMV"):
            out = _run_gemv(*args, config)
        else:
            out = (_run_split if config["SPLIT_K"] > 1 else _run_direct)(*args, config)
    elif m_key <= 16:
        key = (m_key, N, K, bias is not None)
        path = _decode_choice.get(key) or _pick_decode_path(args, key)
        out = PATHS[path](*args)
    else:
        out = _run_direct(*args)
    return out.view(*x.shape[:-1], N)


def export_choices() -> dict:
    return {json_key: path for json_key, path in
            ((",".join(map(str, k)), v) for k, v in _decode_choice.items())}


def import_choices(choices: dict) -> None:
    """Reuse decode-path choices made elsewhere (the profiler must not re-time under ncu)."""
    for k, v in choices.items():
        m, n, kk, b = k.split(",")
        _decode_choice[(int(m), int(n), int(kk), b == "True")] = v


def best_configs() -> dict:
    """Autotuner choices so far, for the results JSON."""
    out = {}
    tuners = (("decode", _tuned_decode), ("prefill", _tuned_prefill), ("split", _tuned_split),
              ("gemv", _tuned_gemv), ("gemv_split", _tuned_gemv_split))
    for name, tuner in tuners:
        for key, cfg in tuner.cache.items():
            out[f"{name}{list(key)[:4]}"] = {**cfg.kwargs, "warps": cfg.num_warps,
                                            "stages": cfg.num_stages}
    out["decode_path"] = {str(k): v for k, v in _decode_choice.items()}
    return out
