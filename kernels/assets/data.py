"""Load benchmark and end-to-end results from results/raw/*.json into flat tables.

A result file is either a single task result or a `multi` result with `steps`. For each
(gpu, kernel, impl, shape) the newest measurement wins, with runs whose run_id starts with
"final" preferred over development runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernels.assets.style import ROOT

RAW = ROOT / "results" / "raw"


def _tasks(path: Path):
    d = json.loads(path.read_text())
    steps = d.get("steps") or [d]
    for s in steps:
        s.setdefault("env", d.get("env", {}))
        yield s


def _gpu(task: dict) -> str:
    name = task.get("env", {}).get("gpu_name", "")
    return "H100" if "H100" in name else "L4" if "L4" in name else name


def bench_rows() -> list[dict]:
    best: dict[tuple, tuple] = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if t.get("task") != "bench" or "results" not in t:
                continue
            run_id = t.get("run_id", "")
            rank = (run_id.startswith("final"), path.name)
            for r in t["results"]:
                row = {**r, "gpu": _gpu(t), "run_id": run_id, "env": t.get("env", {})}
                key = (row["gpu"], r["kernel"], r["impl"], json.dumps(r["shape"], sort_keys=True))
                if key not in best or rank >= best[key][0]:
                    best[key] = (rank, row)
    return [v[1] for v in best.values()]


def e2e_results() -> dict:
    """Newest end-to-end result per GPU."""
    out = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if t.get("task") == "e2e" and t.get("backends"):
                out[_gpu(t)] = {**t, "file": path.name}
    return out


def profile_results() -> dict:
    out = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if t.get("task") == "profile" and t.get("kernels"):
                out[_gpu(t)] = {**t, "file": path.name}
    return out


def power_results() -> dict:
    out = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if t.get("task") == "power" and t.get("kernels"):
                out[_gpu(t)] = {**t, "file": path.name}
    return out


def select(rows, **match):
    """Rows whose fields (or shape fields) equal every keyword."""
    out = []
    for r in rows:
        ok = True
        for k, v in match.items():
            have = r["shape"].get(k) if k in r["shape"] else r.get(k)
            if have != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def one(rows, **match):
    found = select(rows, **match)
    return found[0] if found else None
