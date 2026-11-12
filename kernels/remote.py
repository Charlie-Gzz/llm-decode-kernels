"""Code that runs inside the Modal container. One entry point, a small task registry."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

CACHE = "/cache"


def env_info() -> dict:
    """GPU, driver, CUDA, and library versions. Recorded in every result file."""
    info = {"python": platform.python_version()}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        name, driver, mem, sm_clk, mem_clk = [s.strip() for s in out.splitlines()[0].split(",")]
        info.update(
            gpu_name=name,
            driver=driver,
            gpu_memory=mem,
            max_sm_clock=sm_clk,
            max_mem_clock=mem_clk,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        info["gpu_name"] = "none"
    try:
        import torch
        import triton

        info.update(
            torch=str(torch.__version__),  # TorchVersion would need torch to unpickle locally
            torch_cuda=str(torch.version.cuda),
            triton=str(triton.__version__),
        )
        if torch.cuda.is_available():
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    except ImportError:
        pass
    return info


def check_gpu(expected: str | None) -> dict:
    """Abort before doing any work if Modal gave us a different GPU than requested."""
    info = env_info()
    if expected and expected not in info.get("gpu_name", ""):
        raise RuntimeError(f"expected {expected}, got {info.get('gpu_name')}")
    return info


def task_smoke(params: dict) -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1 << 24
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    def launch():
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)

    t0 = time.perf_counter()
    launch()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    torch.testing.assert_close(out, x + y)
    ms = triton.testing.do_bench(launch)
    gbps = 3 * n * 4 / (ms * 1e-3) / 1e9
    return {"ok": True, "first_call_s": round(compile_s, 2), "vector_add_ms": ms, "gbps": gbps,
            "summary": f"vector add OK, {gbps:.0f} GB/s"}


def task_pytest(params: dict) -> dict:
    import shlex

    args = shlex.split(params.get("args", "tests"))
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *args]
    proc = subprocess.run(cmd, cwd="/root", capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "log": out,
            "summary": "\n".join(out.strip().splitlines()[-25:])}


def task_download(params: dict) -> dict:
    from kernels import config
    from kernels.e2e.data import download_all

    return {"ok": True, **download_all(params.get("model", config.MODEL_ID))}


def task_multi(params: dict) -> dict:
    """Several tasks in one container. L4s can queue for 10+ minutes, so batching saves wall time
    (and repeated startup cost). params: {"steps": [{"task": ..., **params}, ...],
    "deadline_s": seconds for all steps}. Bench steps get the time left before the deadline as
    their budget, so a long sweep stops cleanly (checkpointed) instead of hitting the timeout."""
    out, ok = {"steps": []}, True
    start = time.time()
    deadline = float(params.get("deadline_s", 1500))
    for step in params["steps"]:
        step = dict(step)
        name = step.pop("task")
        left = deadline - (time.time() - start)
        if left < 60:
            out["steps"].append({"task": name, "params": step, "ok": False, "step_s": 0,
                                 "error": "skipped: deadline reached"})
            ok = False
            continue
        if name == "bench":
            step["budget_s"] = min(float(step.get("budget_s", left)), left - 30)
        t0 = time.time()
        try:
            res = TASKS[name](step)
        except Exception:  # keep going: later steps may still be useful
            import traceback

            res = {"ok": False, "error": traceback.format_exc()[-3000:]}
        res.update(task=name, params=step, step_s=round(time.time() - t0, 1))
        ok = ok and bool(res.get("ok", True))
        out["steps"].append(res)
        print(f"[multi] {name} done in {res['step_s']}s ok={res.get('ok', True)}")
    out["ok"] = ok
    out["summary"] = "\n\n".join(
        f"== {s['task']} ({s['step_s']}s) ==\n{s.get('summary') or s.get('error', '')}"
        for s in out["steps"])
    return out


def _lazy(module: str, fn: str):
    def call(params: dict) -> dict:
        import importlib

        return getattr(importlib.import_module(module), fn)(params)

    return call


TASKS = {
    "smoke": task_smoke,
    "pytest": task_pytest,
    "download": task_download,
    "bench": _lazy("kernels.bench.sweep", "run"),
    "e2e": _lazy("kernels.e2e.run", "run"),
    "profile": _lazy("kernels.bench.profile", "run"),
    "collect": _lazy("kernels.bench.sweep", "collect"),
    "power": _lazy("kernels.bench.power", "run"),
    "multi": task_multi,
}


def run_task(task: str, params: dict) -> dict:
    expected = params.pop("expect_gpu", None)
    t0 = time.time()
    info = check_gpu(expected) if task not in ("download", "collect") else env_info()
    os.chdir("/root")
    result = TASKS[task](params)
    result["env"] = info
    result["task"] = task
    result["params"] = params
    result["container_s"] = round(time.time() - t0, 1)
    # Plain JSON types only: the local side has no torch to unpickle torch objects.
    return json.loads(json.dumps(result, default=str))
