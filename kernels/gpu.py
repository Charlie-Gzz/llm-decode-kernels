"""One command per GPU run: clean up stray apps, `modal run` in the foreground, clean up again.

    python -m kernels.gpu smoke
    python -m kernels.gpu pytest --params '{"args": "tests/test_rmsnorm.py"}'
    python -m kernels.gpu bench --long --params '{"kernel": "rmsnorm"}'

Never detaches: if this process dies, Modal stops the app and the GPU with it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MODAL = [sys.executable, "-m", "modal"]
# Modal's CLI prints Unicode; without this, Windows' default codepage crashes it when redirected.
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
ACTIVE = {"ephemeral", "deployed", "running", "initializing", "stopping"}


def stop_stray_apps() -> list[str]:
    out = subprocess.run(MODAL + ["app", "list", "--json"], capture_output=True, text=True)
    apps = json.loads(out.stdout or "[]")
    stopped = []
    for a in apps:
        state = str(a.get("State", a.get("state", ""))).lower()
        app_id = a.get("App ID", a.get("app_id"))
        if app_id and any(s in state for s in ACTIVE) and "stopped" not in state:
            subprocess.run(MODAL + ["app", "stop", "-y", app_id], check=False)
            stopped.append(f"{app_id} ({state})")
    if stopped:
        print(f"[gpu] stopped stray apps: {stopped}  <- log this in the ledger notes")
    return stopped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--gpu", default="L4", choices=["L4", "H100", "CPU"])
    p.add_argument("--long", action="store_true", help="30 min timeout instead of 10")
    p.add_argument("--final", action="store_true", help="M6 final run (uses the reserve)")
    p.add_argument("--params", default="{}")
    p.add_argument("--expect-s", type=int, default=None, help="worst-case seconds for budget check")
    a = p.parse_args()

    expect = a.expect_s or (1800 if (a.long or a.gpu == "H100") else 600)
    stop_stray_apps()
    cmd = MODAL + ["run", "modal_app.py", "--task", a.task, "--gpu", a.gpu,
                   "--params", a.params, "--expect-s", str(expect)]
    if a.long:
        cmd.append("--long")
    if a.final:
        cmd.append("--final")
    rc = subprocess.run(cmd).returncode
    stop_stray_apps()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
