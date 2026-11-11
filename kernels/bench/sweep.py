"""Benchmark sweeps for every kernel and baseline. Runs inside the Modal container.

Resumable: each finished (sweep, kernel, impl, shape) is appended to a checkpoint file on the
Modal Volume, and a rerun with the same run_id skips it.

params:
  run_id   (required) checkpoint name, e.g. "m2-final"
  kernels  list from KERNELS, default all Triton kernels
  impls    optional list to filter implementations (e.g. ["eager", "compile"] = baselines only)
  sweeps   default 3
  quick    small shape set for development
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from kernels import config
from kernels.bench import harness

CKPT_DIR = Path("/cache/bench")


@dataclass
class Case:
    kernel: str
    impl: str
    shape: dict
    fn: object
    make: object  # () -> tuple of inputs, a fresh copy each call
    bytes: int
    flops: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)

    def key(self, sweep: int) -> str:
        shape = ",".join(f"{k}={v}" for k, v in self.shape.items())
        return f"{sweep}|{self.kernel}|{self.impl}|{shape}"


def _compile(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False)


def _setup_compile():
    import torch._dynamo

    # dynamic=False recompiles per shape; the default limit (8) would silently fall back to eager.
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 4096)


# ---------------------------------------------------------------- copy (practical ceiling)
def cases_copy(quick: bool):
    n = 64 * 2**20  # 64M fp16 = 128 MiB read + 128 MiB write
    yield Case("copy", "torch", {"elems": n},
               fn=lambda src, dst: dst.copy_(src),
               make=lambda: (torch.randn(n, device="cuda", dtype=torch.half),
                             torch.empty(n, device="cuda", dtype=torch.half)),
               bytes=2 * n * 2)


# ---------------------------------------------------------------- RMSNorm
def cases_rmsnorm(quick: bool):
    from kernels.reference import rmsnorm_ref
    from kernels.rmsnorm import rmsnorm, rmsnorm_bytes

    Ms = [1, 16, 4096] if quick else [1, 16, 128, 1024, 4096, 16384]
    Ns = [1536] if quick else [1536, 4096]
    compiled = _compile(rmsnorm_ref)
    impls = {"triton": rmsnorm, "eager": rmsnorm_ref, "compile": compiled}
    for residual in (True, False):
        for N in Ns:
            for M in Ms:
                def make(M=M, N=N, residual=residual):
                    x = torch.randn(M, N, device="cuda", dtype=torch.half)
                    w = torch.ones(N, device="cuda", dtype=torch.half)
                    r = torch.randn(M, N, device="cuda", dtype=torch.half) if residual else None
                    return (x, w, 1e-6, r)
                for impl, fn in impls.items():
                    yield Case("rmsnorm", impl, {"M": M, "N": N, "residual": residual},
                               fn=fn, make=make, bytes=rmsnorm_bytes(M, N, residual))


# ---------------------------------------------------------------- SwiGLU
def _silu_mul_halves(gu):
    i = gu.shape[-1] // 2
    return F.silu(gu[..., :i]) * gu[..., i:]


def cases_swiglu(quick: bool):
    from kernels.swiglu import swiglu_bytes, swiglu_fused

    Ms = [1, 16, 2048] if quick else [1, 16, 128, 1024, 4096]
    Is = [8960] if quick else [8960, 18944]
    impls = {"triton": swiglu_fused, "eager": _silu_mul_halves,
             "compile": _compile(_silu_mul_halves)}
    for I in Is:
        for M in Ms:
            def make(M=M, I=I):
                return (torch.randn(M, 2 * I, device="cuda", dtype=torch.half),)
            for impl, fn in impls.items():
                yield Case("swiglu", impl, {"M": M, "I": I}, fn=fn, make=make,
                           bytes=swiglu_bytes(M, I))


# ---------------------------------------------------------------- W4A16 matmul
W4_SHAPES = config.W4_SHAPES
DECODE_M = [1, 2, 4, 8, 16]
PREFILL_M = [128, 512, 2048]


def _naive(x, qw, s, z):
    from kernels.reference import dequantize_ref

    return F.linear(x, dequantize_ref(qw, s, z, config.GROUP_SIZE))


def _torchao_linear(w16: torch.Tensor):
    """torchao int4 weight-only (tinygemm). bf16 only, so it gets a bf16 copy of the weight."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    lin = torch.nn.Linear(w16.shape[1], w16.shape[0], bias=False, device="cuda",
                          dtype=torch.bfloat16)
    lin.weight.data.copy_(w16.to(torch.bfloat16))
    quantize_(lin, Int4WeightOnlyConfig(group_size=config.GROUP_SIZE,
                                        int4_packing_format="tile_packed_to_4d"))
    return lin


def cases_w4a16(quick: bool, phase: str = "all"):
    from kernels.quant import quantize_rtn, weight_bytes
    from kernels.w4a16 import w4a16_matmul

    shapes = W4_SHAPES[:2] if quick else W4_SHAPES
    Ms = {"decode": DECODE_M, "prefill": PREFILL_M, "all": DECODE_M + PREFILL_M}[phase]
    if quick:
        Ms = [m for m in Ms if m in (1, 16, 512)]
    impls = {
        "triton": lambda x, qw, s, z: w4a16_matmul(x, qw, s, z),
        "naive": _naive,
        "compile": _compile(_naive),
    }
    try:
        from kernels.cuda.w4a16_cuda import w4a16_decode_cuda
        impls["cuda"] = lambda x, qw, s, z: w4a16_decode_cuda(x, qw, s, z)
    except Exception as e:  # not built yet, or failed to build: reported, not hidden
        print(f"[sweep] CUDA kernel unavailable: {e!r}")

    for label, N, K in shapes:
        w16 = (torch.randn(N, K, device="cuda") * 0.02).half()
        qw, s, z = quantize_rtn(w16, config.GROUP_SIZE)
        ao = None
        for M in Ms:
            shape = {"layer": label, "M": M, "N": N, "K": K}
            flops = 2 * M * N * K
            act = M * K * 2 + M * N * 2
            w4_bytes = act + weight_bytes(N, K)

            def make_q(M=M, K=K, qw=qw, s=s, z=z):
                x = torch.randn(M, K, device="cuda", dtype=torch.half)
                return (x, qw.clone(), s.clone(), z.clone())

            for impl, fn in impls.items():
                if impl == "cuda" and M > 16:
                    continue  # CUDA kernel covers decode only (PLAN M4)
                yield Case("w4a16", impl, shape, fn=fn, make=make_q, bytes=w4_bytes, flops=flops)

            def make_fp16(M=M, K=K, w16=w16):
                return (torch.randn(M, K, device="cuda", dtype=torch.half), w16.clone())

            yield Case("w4a16", "fp16", shape, fn=F.linear, make=make_fp16,
                       bytes=act + N * K * 2, flops=flops, note="cuBLAS fp16, 4x the weight bytes")

            try:
                ao = ao or _torchao_linear(w16)
            except Exception as e:
                print(f"[sweep] torchao unavailable: {e!r}")
                ao = False
            if ao:
                def make_ao(M=M, K=K, ao=ao):
                    import copy
                    return (torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
                            copy.deepcopy(ao))

                yield Case("w4a16", "torchao", shape, fn=lambda x, lin: lin(x), make=make_ao,
                           bytes=w4_bytes, flops=flops, note="torchao int4 tinygemm, bf16",
                           extra={"set_bytes": w4_bytes})


KERNELS = {
    "copy": cases_copy,
    "rmsnorm": cases_rmsnorm,
    "swiglu": cases_swiglu,
    "w4a16": cases_w4a16,
    "w4a16_decode": lambda q: cases_w4a16(q, "decode"),
    "w4a16_prefill": lambda q: cases_w4a16(q, "prefill"),
}


def _commit():
    try:
        import modal

        modal.Volume.from_name("llm-decode-kernels-cache").commit()
    except Exception as e:
        print(f"[sweep] volume commit failed: {e!r}")


def aggregate(rows: list[dict], spec: dict) -> list[dict]:
    """Median over sweeps per (kernel, impl, shape); min/max and a >10% spread flag."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        groups.setdefault((r["kernel"], r["impl"], json.dumps(r["shape"])), []).append(r)
    out = []
    for (kernel, impl, shape), rs in groups.items():
        g = [r["graph_ms"] for r in rs]
        med = statistics.median(g)
        row = {
            "kernel": kernel, "impl": impl, "shape": json.loads(shape),
            "sweeps": len(rs), "ms": med, "ms_min": min(g), "ms_max": max(g),
            "spread_pct": 100 * (max(g) - min(g)) / med,
            "bytes": rs[0]["bytes"], "flops": rs[0]["flops"], "note": rs[0].get("note", ""),
            "copies": rs[0]["copies"],
        }
        eager = [r["eager_ms"] for r in rs if "eager_ms" in r]
        if eager:
            row["eager_ms"] = statistics.median(eager)
        row["flag_spread"] = row["spread_pct"] > 10
        row.update(harness.metrics(med, row["bytes"], row["flops"],
                                   spec["peak_gbps"], spec["peak_fp16_tflops"]))
        out.append(row)
    return out


def collect(params: dict) -> dict:
    """CPU-only: rebuild results from a checkpoint on the Volume (e.g. after a crashed run)."""
    gpu = params["gpu"]
    ckpt = CKPT_DIR / f"{params['run_id']}_{gpu}.jsonl"
    rows = [json.loads(line) for line in ckpt.read_text().splitlines()]
    gpu_name = next((r["gpu_name"] for r in rows if "gpu_name" in r), gpu)
    return {"ok": True, "task": "bench", "run_id": params["run_id"], "gpu": gpu,
            "spec": config.GPU_SPECS[gpu], "results": aggregate(rows, config.GPU_SPECS[gpu]),
            "raw": rows, "errors": [r for r in rows if "error" in r], "env": {"gpu_name": gpu_name},
            "collected_from_checkpoint": True,
            "summary": f"{len(rows)} rows from {ckpt.name}"}


def run(params: dict) -> dict:
    from kernels.remote import env_info

    _setup_compile()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # torch default
    env = env_info()
    gpu = config.gpu_key(env["gpu_name"])
    spec = config.GPU_SPECS[gpu]
    run_id = params["run_id"]
    kernels = params.get("kernels", ["copy", "rmsnorm", "swiglu", "w4a16"])
    impl_filter = params.get("impls")
    sweeps = int(params.get("sweeps", 3))
    quick = bool(params.get("quick", False))
    budget_s = float(params.get("budget_s", 1500))  # stop cleanly before the function timeout

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{run_id}_{gpu}.jsonl"
    rows = [json.loads(line) for line in ckpt.read_text().splitlines()] if ckpt.exists() else []
    done = {r["key"] for r in rows}
    print(f"[sweep] {run_id} on {gpu}: {len(done)} cases already done")

    t0 = last_commit = time.time()
    finished = True
    for sweep in range(sweeps):
        for kname in kernels:
            for case in KERNELS[kname](quick):
                if impl_filter and case.impl not in impl_filter:
                    continue
                key = case.key(sweep)
                if key in done:
                    continue
                if time.time() - t0 > budget_s:
                    finished = False
                    break
                row = {"key": key, "sweep": sweep, "kernel": case.kernel, "impl": case.impl,
                       "gpu_name": env["gpu_name"],
                       "shape": case.shape, "bytes": case.bytes, "flops": case.flops,
                       "note": case.note}
                try:
                    row.update(harness.measure(case.fn, case.make, spec["l2_mb"],
                                               set_bytes=case.extra.get("set_bytes")))
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
                    print(f"[sweep] ERROR {key}: {row['error']}")
                    torch.cuda.empty_cache()
                rows.append(row)
                with ckpt.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if time.time() - last_commit > 30:
                    _commit()
                    last_commit = time.time()
    _commit()

    agg = aggregate(rows, spec)
    extra = {}
    if any(k.startswith("w4a16") for k in kernels):
        from kernels.w4a16 import best_configs

        extra["triton_configs"] = best_configs()
    errors = [r for r in rows if "error" in r]
    lines = [f"{r['kernel']:8s} {r['impl']:8s} {json.dumps(r['shape']):60s} "
             f"{r['ms'] * 1e3:9.1f} us {r['gbps']:7.1f} GB/s {r['pct_peak_bw']:5.1f}%"
             f"{' SPREAD' if r['flag_spread'] else ''}" for r in agg]
    return {
        "ok": finished and not errors,
        "finished": finished,
        "run_id": run_id,
        "gpu": gpu,
        "spec": spec,
        "method": {"timing": "cuda_graph_rotating_buffers", "measurements": harness.MEASUREMENTS,
                   "min_ms": harness.MIN_MS, "sweeps": sweeps, "dtype": "float16"},
        "results": agg,
        "errors": errors,
        "raw": rows,
        **extra,
        "summary": "\n".join(lines[-80:]) + f"\nerrors: {len(errors)}, finished: {finished}",
    }
