"""GPU spend ledger and budget guard.

docs/gpu_ledger.jsonl is the source of truth (one JSON object per GPU run).
docs/GPU_LEDGER.md is rendered from it after every run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kernels import config

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "docs" / "gpu_ledger.jsonl"
MARKDOWN = ROOT / "docs" / "GPU_LEDGER.md"


class BudgetError(RuntimeError):
    pass


def load() -> list[dict]:
    if not JSONL.exists():
        return []
    return [json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()]


def estimate_cost(gpu: str, seconds: float) -> float:
    rate = config.GPU_USD_PER_HOUR[gpu] / 3600
    return seconds * rate * (1 + config.CPU_MEM_ALLOWANCE)


def totals(rows: list[dict] | None = None) -> dict:
    rows = load() if rows is None else rows
    usd = sum(r["est_usd"] for r in rows)
    dev_usd = sum(r["est_usd"] for r in rows if not r.get("final"))
    seconds = sum(r["billed_s"] for r in rows)
    h100_s = sum(r["billed_s"] for r in rows if r["gpu"] == "H100")
    return {"usd": usd, "dev_usd": dev_usd, "gpu_hours": seconds / 3600, "h100_s": h100_s}


def check_budget(gpu: str, final: bool, expected_s: float) -> None:
    """Raise BudgetError if this run could push spend past a cap in PLAN.md 1.1."""
    t = totals()
    cost = estimate_cost(gpu, expected_s)
    if t["usd"] + cost > config.BUDGET_TOTAL_USD:
        raise BudgetError(f"total cap: spent ${t['usd']:.2f}, this run up to ${cost:.2f}")
    if t["gpu_hours"] + expected_s / 3600 > config.BUDGET_TOTAL_GPU_HOURS:
        raise BudgetError(f"GPU-hour cap: used {t['gpu_hours']:.2f} h")
    if not final and t["dev_usd"] + cost > config.BUDGET_DEV_USD:
        raise BudgetError(f"dev budget: spent ${t['dev_usd']:.2f} of ${config.BUDGET_DEV_USD}")
    if gpu == "H100":
        if not final:
            raise BudgetError("H100 is only allowed for the final (M6) session")
        if t["h100_s"] + expected_s > config.H100_MAX_SECONDS:
            raise BudgetError(f"H100 cap: used {t['h100_s']:.0f} s of {config.H100_MAX_SECONDS}")


def record(purpose: str, gpu: str, billed_s: float, gpu_name: str, final: bool, ok: bool) -> dict:
    row = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "purpose": purpose,
        "gpu": gpu,
        "gpu_name": gpu_name,
        "billed_s": round(billed_s, 1),
        "est_usd": round(estimate_cost(gpu, billed_s), 4),
        "final": final,
        "ok": ok,
    }
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a", newline="\n") as f:
        f.write(json.dumps(row) + "\n")
    render()
    return row


def render() -> None:
    rows = load()
    t = totals(rows)
    lines = [
        "# GPU ledger",
        "",
        "Generated from `docs/gpu_ledger.jsonl` by `kernels/ledger.py`. Do not edit by hand.",
        "",
        "Billed seconds are measured on the local side around each `modal run` call, so they",
        "include scheduling and container startup. Estimated cost = GPU rate x 1.15 (CPU and",
        f"memory allowance). Rates: L4 ${config.GPU_USD_PER_HOUR['L4']}/h, "
        f"H100 ${config.GPU_USD_PER_HOUR['H100']}/h.",
        "",
        f"**Totals:** {t['gpu_hours']:.3f} GPU-hours, est. ${t['usd']:.2f} of "
        f"${config.BUDGET_TOTAL_USD:.0f} (dev ${t['dev_usd']:.2f} of "
        f"${config.BUDGET_DEV_USD:.0f}, H100 {t['h100_s'] / 60:.1f} of "
        f"{config.H100_MAX_SECONDS / 60:.0f} min).",
        "",
        "| # | Date | Purpose | GPU | nvidia-smi name | Billed s | Est. $ | Running $ | OK |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    running = 0.0
    for i, r in enumerate(rows, 1):
        running += r["est_usd"]
        lines.append(
            f"| {i} | {r['date']} | {r['purpose']} | {r['gpu']} | {r['gpu_name']} | "
            f"{r['billed_s']:.0f} | {r['est_usd']:.3f} | {running:.3f} | "
            f"{'yes' if r['ok'] else 'no'} |"
        )
    recon = ROOT / "docs" / "ledger_reconciliation.md"
    if recon.exists():
        lines += ["", recon.read_text().strip()]
    MARKDOWN.write_text("\n".join(lines) + "\n", newline="\n")


if __name__ == "__main__":
    render()
    print(json.dumps(totals(), indent=2))
