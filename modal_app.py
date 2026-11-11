"""Modal app: every GPU run in this project goes through here.

Rules (PLAN.md 1.1): `modal run` only, one container at a time, timeouts on every function,
budget checked before and ledger written after every GPU call.

Usage (prefer the wrapper, which also checks for stray apps):
    python -m kernels.gpu smoke
    python -m kernels.gpu pytest --args "tests/test_rmsnorm.py"
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from kernels import config

BOOT = time.time()  # in a container: when Python started, used to estimate billed seconds
STARTUP_ALLOWANCE_S = 20  # image load before Python starts, which we cannot see from inside

app = modal.App("llm-decode-kernels")
cache = modal.Volume.from_name("llm-decode-kernels-cache", create_if_missing=True)
CACHE = "/cache"

image = (
    modal.Image.from_registry(config.CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install(*config.PIP_PACKAGES)
    .env(
        {
            "HF_HOME": f"{CACHE}/hf",
            "TRITON_CACHE_DIR": f"{CACHE}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor",
            "TORCH_EXTENSIONS_DIR": f"{CACHE}/torch_ext",
            "TORCH_CUDA_ARCH_LIST": "8.9;9.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir("tests", remote_path="/root/tests")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_python_source("kernels")
)

COMMON = dict(image=image, volumes={CACHE: cache}, max_containers=1)


def _run(task: str, params: dict) -> dict:
    from kernels import remote

    try:
        result = remote.run_task(task, params)
    finally:
        cache.commit()
    result["container_wall_s"] = round(time.time() - BOOT, 1)
    return result


# CPU-only: downloads go here, never on a GPU container.
@app.function(cpu=2, memory=8192, timeout=1200, **COMMON)
def cpu_task(task: str, params: dict) -> dict:
    return _run(task, params)


@app.function(gpu=config.MODAL_GPU["L4"], timeout=600, **COMMON)
def l4_task(task: str, params: dict) -> dict:
    return _run(task, params)


@app.function(gpu=config.MODAL_GPU["L4"], timeout=1800, **COMMON)
def l4_bench(task: str, params: dict) -> dict:
    return _run(task, params)


@app.function(gpu=config.MODAL_GPU["H100"], timeout=1800, **COMMON)
def h100_bench(task: str, params: dict) -> dict:
    return _run(task, params)


def _save(task: str, gpu: str, result: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("results") / "raw" / f"{task}_{gpu}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), newline="\n")
    return out


@app.local_entrypoint()
def main(
    task: str,
    gpu: str = "L4",
    long: bool = False,
    final: bool = False,
    params: str = "{}",
    expect_s: int = 600,
):
    from kernels import ledger

    kwargs = json.loads(params)
    if gpu == "CPU":
        result = cpu_task.remote(task, kwargs)
        print(json.dumps(result, indent=2)[:4000])
        return

    if gpu not in ("L4", "H100"):
        raise SystemExit(f"GPU {gpu!r} is not allowed (PLAN.md 1.1)")
    fn = h100_bench if gpu == "H100" else (l4_bench if long else l4_task)
    ledger.check_budget(gpu, final=final, expected_s=expect_s)

    purpose = f"{task} {kwargs}" if kwargs else task
    result, ok, gpu_name = {}, False, "?"
    t0 = time.monotonic()
    try:
        result = fn.remote(task, {**kwargs, "expect_gpu": gpu})
        ok = bool(result.get("ok", True))
        gpu_name = result.get("env", {}).get("gpu_name", "?")
    finally:
        # Modal bills container time, not queue time. Use the container's own clock plus a
        # startup allowance; if no result came back (crash, timeout), fall back to local wall
        # time, which over-estimates.
        local_wall = time.monotonic() - t0
        if "container_wall_s" in result:
            billed = result["container_wall_s"] + STARTUP_ALLOWANCE_S
            note = f"local wall {local_wall:.0f}s"
        else:
            billed, note = local_wall, "no result: local wall time used"
        row = ledger.record(purpose[:80], gpu, billed, gpu_name, final, ok, note)
        print(f"[ledger] {row}")
        print(f"[ledger] totals {ledger.totals()}")
    path = _save(task, gpu, result)
    print(f"[result] saved {path}")
    summary = result.get("summary")
    if summary:
        print(summary)
