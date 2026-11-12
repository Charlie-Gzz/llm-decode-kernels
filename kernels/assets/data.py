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
            # final runs beat dev runs; a later final run_id ("final2") beats "final"; then newest
            rank = (run_id.startswith("final"), run_id if run_id.startswith("final") else "",
                    path.name)
            for r in t["results"]:
                row = {**r, "gpu": _gpu(t), "run_id": run_id, "env": t.get("env", {})}
                key = (row["gpu"], r["kernel"], r["impl"], json.dumps(r["shape"], sort_keys=True))
                if key not in best or rank >= best[key][0]:
                    best[key] = (rank, row)
    return [v[1] for v in best.values()]


def e2e_results(model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> dict:
    """Newest end-to-end result per GPU for one model. Early results stored the snapshot path
    (".../models--Qwen--Qwen2.5-1.5B-Instruct/...") instead of the id, so match on the name."""
    name = model.split("/")[-1]
    out = {}
    stretch = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if t.get("task") != "e2e" or not t.get("backends") or name not in str(t.get("model")):
                continue
            if t.get("params", {}).get("tag") == "followup":
                continue  # reported on its own, see followup_results
            if t.get("params", {}).get("tag") == "stretch":
                # stretch runs only contribute their own backend (FP8), never replace the finals
                for b, r in t["backends"].items():
                    if b.startswith("ours_fp8"):
                        stretch.setdefault(_gpu(t), {})[b] = r
                continue
            out[_gpu(t)] = {**t, "file": path.name, "backends": dict(t["backends"])}
    for gpu, backends in stretch.items():
        if gpu in out:
            out[gpu]["backends"].update(backends)
    return out


def followup_results(model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> dict:
    """Newest follow-up e2e run per GPU (lm_head quantized too). All its backends ran in one
    container, so they are compared with each other, never mixed with the final runs."""
    name = model.split("/")[-1]
    out = {}
    for path in sorted(RAW.glob("*.json")):
        for t in _tasks(path):
            if (t.get("task") == "e2e" and t.get("backends") and name in str(t.get("model"))
                    and t.get("params", {}).get("tag") == "followup"):
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
