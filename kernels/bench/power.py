"""Power probe: is the L4 power-limited during sustained decode matmuls?

For each kernel, replay a CUDA graph (rotating weights, so DRAM-bound) for a few seconds while
nvidia-smi samples SM clock, memory clock, power and throttle reasons every 100 ms.
"""

from __future__ import annotations

import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

from kernels.bench import harness


def _query() -> str:
    """Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons."""
    help_text = subprocess.run(["nvidia-smi", "--help-query-gpu"], capture_output=True,
                               text=True).stdout
    reasons = ("clocks_event_reasons.active" if "clocks_event_reasons.active" in help_text
               else "clocks_throttle_reasons.active")
    return f"clocks.sm,clocks.mem,power.draw,power.limit,{reasons}"


def _cases(N=17920, K=1536):
    from kernels.cuda.w4a16_cuda import w4a16_decode_cuda
    from kernels.quant import quantize_rtn
    from kernels.w4a16 import w4a16_matmul

    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    qw, s, z = quantize_rtn(w)

    def q_inputs():
        return (torch.randn(1, K, device="cuda", dtype=torch.half), qw.clone(), s.clone(),
                z.clone())

    def f_inputs():
        return (torch.randn(1, K, device="cuda", dtype=torch.half), w.clone())

    return [
        ("triton", lambda x, a, b, c: w4a16_matmul(x, a, b, c), q_inputs),
        ("cuda", lambda x, a, b, c: w4a16_decode_cuda(x, a, b, c), q_inputs),
        ("fp16", F.linear, f_inputs),
    ]


def _probe(fn, make, seconds: float) -> dict:
    first = make()
    fn(*first)
    torch.cuda.synchronize()
    sets = [first] + [make() for _ in range(harness.n_copies(harness.tensor_bytes(first), 48) - 1)]
    for args in sets[:3]:
        fn(*args)
    torch.cuda.synchronize()
    graph, outs = harness.capture(lambda: [fn(*a) for a in sets])
    smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={_query()}",
                            "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=subprocess.PIPE, text=True)
    replays, t0 = 0, time.time()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    while time.time() - t0 < seconds:
        for _ in range(20):
            graph.replay()
        replays += 20
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    smi.terminate()
    out, _ = smi.communicate()
    rows = [line.split(", ") for line in out.strip().splitlines() if line.count(",") == 4]
    rows = rows[len(rows) // 4:]  # drop the ramp-up quarter
    us = start.elapsed_time(end) * 1e3 / (replays * len(sets))
    del graph, outs
    sets.clear()
    torch.cuda.empty_cache()
    return {
        "us_per_call": us,
        "sm_clock_mhz": statistics.median(float(r[0]) for r in rows),
        "mem_clock_mhz": statistics.median(float(r[1]) for r in rows),
        "power_w": statistics.median(float(r[2]) for r in rows),
        "power_limit_w": float(rows[0][3]),
        "throttle_reasons": sorted({r[4] for r in rows}),
        "samples": len(rows),
    }


def run(params: dict) -> dict:
    seconds = float(params.get("seconds", 4))
    out = {name: _probe(fn, make, seconds) for name, fn, make in _cases()}
    idle = subprocess.run(["nvidia-smi", f"--query-gpu={_query()}",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    lines = [f"{k:8s} {v['us_per_call']:7.1f} us  sm {v['sm_clock_mhz']:.0f} MHz  "
             f"mem {v['mem_clock_mhz']:.0f} MHz  {v['power_w']:.0f}/{v['power_limit_w']:.0f} W  "
             f"throttle {v['throttle_reasons']}" for k, v in out.items()]
    return {"ok": True, "shape": "gate_up 17920x1536, M=1", "kernels": out, "idle": idle,
            "summary": "\n".join(lines)}
