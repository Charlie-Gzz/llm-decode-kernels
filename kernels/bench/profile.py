"""Profiling task: Nsight Compute on one launch of each kernel, PyTorch profiler as fallback.

`python -m kernels.bench.profile target` is the program ncu runs. It warms everything up (so
autotuning is not profiled), then calls each kernel once between cudaProfilerStart/Stop.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
]


def _cases():
    """(label, fn) pairs at decode and prefill shapes from Qwen2.5-1.5B."""
    import torch.nn.functional as F

    from kernels.quant import quantize_rtn
    from kernels.rmsnorm import rmsnorm
    from kernels.swiglu import swiglu_fused
    from kernels.w4a16 import w4a16_matmul

    cases = []
    for label, N, K in [("gate_up", 17920, 1536), ("down", 1536, 8960)]:
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        qw, s, z = quantize_rtn(w)
        for M in (1, 16):
            x = torch.randn(M, K, device="cuda", dtype=torch.half)
            cases.append((f"w4a16_triton_{label}_M{M}", lambda x=x, qw=qw, s=s, z=z:
                          w4a16_matmul(x, qw, s, z)))
            cases.append((f"fp16_cublas_{label}_M{M}", lambda x=x, w=w: F.linear(x, w)))
            try:
                from kernels.cuda.w4a16_cuda import w4a16_decode_cuda

                w4a16_decode_cuda(x, qw, s, z)
                cases.append((f"w4a16_cuda_{label}_M{M}", lambda x=x, qw=qw, s=s, z=z:
                              w4a16_decode_cuda(x, qw, s, z)))
            except Exception as e:
                print(f"[profile] cuda kernel unavailable: {e!r}", file=sys.stderr)
    x = torch.randn(4096, 1536, device="cuda", dtype=torch.half)
    r = torch.randn_like(x)
    wn = torch.ones(1536, device="cuda", dtype=torch.half)
    cases.append(("rmsnorm_res_M4096", lambda: rmsnorm(x, wn, 1e-6, r)))
    gu = torch.randn(4096, 2 * 8960, device="cuda", dtype=torch.half)
    cases.append(("swiglu_M4096", lambda: swiglu_fused(gu)))
    return cases


CHOICES = Path("/tmp/profile_choices.json")


def _warm(cases):
    for _, fn in cases:  # autotune, compile, pick paths / split-K
        for _ in range(3):
            fn()
    torch.cuda.synchronize()


def _save_choices():
    """Decode-path and split-K choices are timed in-process; under ncu every launch is slowed
    and serialized, so re-timing there picks badly (seen: split-K=1 with 12 blocks)."""
    from kernels import w4a16
    from kernels.cuda import w4a16_cuda

    CHOICES.write_text(json.dumps({"triton": w4a16.export_choices(),
                                   "cuda": w4a16_cuda.export_choices()}))


def _load_choices():
    from kernels import w4a16
    from kernels.cuda import w4a16_cuda

    if CHOICES.exists():
        c = json.loads(CHOICES.read_text())
        w4a16.import_choices(c["triton"])
        w4a16_cuda.import_choices(c["cuda"])


def target():
    _load_choices()
    cases = _cases()
    for _, fn in cases:  # warm up: autotune (cached on disk), compile
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for label, fn in cases:
        torch.cuda.nvtx.range_push(label)
        fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _ncu() -> dict:
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    # Modal containers cannot lock GPU clocks, so profile at the clocks the GPU is running.
    cmd = [ncu, "--profile-from-start", "off", "--clock-control", "none",
           "--nvtx", "--print-nvtx-rename", "kernel",
           "--csv", "--page", "raw", "--metrics", ",".join(METRICS),
           sys.executable, "-m", "kernels.bench.profile", "target"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/root")
    text = proc.stdout
    if proc.returncode != 0 or "ERR_NVGPUCTRPERM" in text + proc.stderr:
        return {"ok": False, "tool": "ncu", "returncode": proc.returncode,
                "log": (text + proc.stderr)[-4000:]}
    start = text.find('"ID"')
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    kernels = []
    for row in rows[1:]:  # first row after the header holds units
        kernels.append({
            "nvtx": row.get("NVTX Range", "") or row.get("thread Domain:Push/Pop_Range:PL_Type:"
                                                         "PL_Value:CLR_Type:Color:Msg_Type:Msg",
                                                         ""),
            "kernel": row.get("Kernel Name", ""),
            **{m: row.get(m, "") for m in METRICS},
        })
    units = {m: rows[0].get(m, "") for m in METRICS} if rows else {}
    return {"ok": True, "tool": "ncu", "units": units, "kernels": kernels}


def _torch_profiler() -> dict:
    from torch.profiler import ProfilerActivity, profile

    cases = _cases()
    for _, fn in cases:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    out = []
    for label, fn in cases:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        for ev in prof.key_averages():
            if ev.device_time_total > 0:
                out.append({"case": label, "kernel": ev.key, "us": ev.device_time_total})
    return {"ok": True, "tool": "torch.profiler", "kernels": out}


def run(params: dict) -> dict:
    _warm(_cases())  # make every timing-based choice here, outside ncu
    _save_choices()
    res = _ncu()
    if not res["ok"]:
        fallback = _torch_profiler()
        fallback["ncu_failure"] = res.get("log", "")[-1500:]
        res = fallback
    lines = [f"tool: {res['tool']}"]
    for k in res["kernels"]:
        lines.append(" | ".join(str(v)[:60] for v in k.values()))
    res["summary"] = "\n".join(lines[:60])
    return res


if __name__ == "__main__" and sys.argv[1:] == ["target"]:
    target()
