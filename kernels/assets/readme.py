"""Fill the generated blocks of README.md from results JSON.

README.md holds hand-written prose plus blocks between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->`
that this module rewrites, so numbers in the README always match the charts.
"""

from __future__ import annotations

import re
import statistics

from kernels import ledger
from kernels.assets import data
from kernels.assets.style import ROOT

README = ROOT / "README.md"


def _fmt(v, spec=".1f", none="-"):
    return none if v is None else format(v, spec)


def headline(rows, e2e) -> str:
    gpu = "L4"
    r = data.one(rows, gpu=gpu, kernel="w4a16", impl="triton", layer="qwen_gate_up", M=1)
    b = e2e.get(gpu, {}).get("backends", {})
    if not r or "ours" not in b:
        return "_Headline pending: results not in yet._"
    sp = b["ours"]["decode"]["tok_s"] / b["fp16"]["decode"]["tok_s"]
    return (f"**4-bit decode matmul reaches {r['pct_peak_bw']:.0f}% of peak memory bandwidth on "
            f"L4, for {sp:.2f}x faster decoding than fp16 on Qwen2.5-1.5B.**")


def kernel_table(rows) -> str:
    lines = ["| Kernel | Shape | GPU | Ours (us) | PyTorch eager (us) | torch.compile (us) | "
             "Speedup vs eager | % peak BW |",
             "|---|---|---|---|---|---|---|---|"]
    picks = [("rmsnorm", {"residual": True, "N": 4096, "M": 16384}, "eager", "RMSNorm + residual"),
             ("rmsnorm", {"residual": True, "N": 1536, "M": 1}, "eager", "RMSNorm + residual"),
             ("swiglu", {"I": 8960, "M": 4096}, "eager", "SwiGLU"),
             ("swiglu", {"I": 8960, "M": 1}, "eager", "SwiGLU")]
    for gpu in ("L4", "H100"):
        for kernel, shape, base, label in picks:
            t = data.one(rows, gpu=gpu, kernel=kernel, impl="triton", **shape)
            e = data.one(rows, gpu=gpu, kernel=kernel, impl=base, **shape)
            c = data.one(rows, gpu=gpu, kernel=kernel, impl="compile", **shape)
            if not t:
                continue
            sh = ", ".join(f"{k}={v}" for k, v in shape.items() if k != "residual")
            lines.append(f"| {label} | {sh} | {gpu} | {t['ms'] * 1e3:.1f} | "
                         f"{_fmt(e and e['ms'] * 1e3)} | {_fmt(c and c['ms'] * 1e3)} | "
                         f"{_fmt(e and e['ms'] / t['ms'], '.2f')}x | {t['pct_peak_bw']:.0f}% |")
    return "\n".join(lines)


def w4_table(rows) -> str:
    lines = ["| Layer (N x K) | M | GPU | Triton (us) | CUDA (us) | fp16 cuBLAS (us) "
             "| torchao (us) "
             "| torch.compile (us) | Triton vs fp16 | Triton % peak BW |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for gpu in ("L4", "H100"):
        for layer in ("qwen_gate_up", "qwen_down", "8b_up"):
            for M in (1, 16, 2048):
                by = {i: data.one(rows, gpu=gpu, kernel="w4a16", impl=i, layer=layer, M=M)
                      for i in ("triton", "cuda", "fp16", "torchao", "compile")}
                t = by["triton"]
                if not t:
                    continue
                us = {k: (v["ms"] * 1e3 if v else None) for k, v in by.items()}
                f = by["fp16"]
                lines.append(
                    f"| {layer} ({t['shape']['N']}x{t['shape']['K']}) | {M} | {gpu} | "
                    f"{us['triton']:.1f} | {_fmt(us['cuda'])} | {_fmt(us['fp16'])} | "
                    f"{_fmt(us['torchao'])} | {_fmt(us['compile'])} | "
                    f"{_fmt(f and f['ms'] / t['ms'], '.2f')}x | {t['pct_peak_bw']:.0f}% |")
    return "\n".join(lines)


def e2e_table(e2e) -> str:
    lines = ["| GPU | Backend | Decode tok/s (batch 1) | Swapped layers, ms/token | "
             "Prefill tok/s (T=2048) | WikiText-2 ppl | Linear weights |",
             "|---|---|---|---|---|---|---|"]
    for gpu, res in e2e.items():
        for name, r in res["backends"].items():
            d = r.get("decode")
            pre = next((p["tok_s"] for p in r.get("prefill", []) if p["T"] == 2048), None)
            lines.append(f"| {gpu} | {name} | {_fmt(d and d['tok_s'])} | "
                         f"{_fmt(r.get('layers_only_ms'), '.2f')} | {_fmt(pre, '.0f')} | "
                         f"{r['ppl']:.3f} | {r['weights']['linear_layers'] / 1e9:.2f} GB |")
    return "\n".join(lines)


def cost() -> str:
    t = ledger.totals()
    return (f"Total GPU time: **{t['gpu_hours']:.2f} GPU-hours**, estimated **${t['usd']:.2f}** "
            f"(cap $30), of which H100: {t['h100_s'] / 60:.0f} min. Every run is in "
            "[docs/GPU_LEDGER.md](docs/GPU_LEDGER.md).")


def spread_note(rows) -> str:
    flagged = [r for r in rows if r.get("flag_spread")]
    spreads = [r["spread_pct"] for r in rows if "spread_pct" in r]
    if not spreads:
        return ""
    return (f"Median spread between the 3 sweeps: {statistics.median(spreads):.1f}%. "
            f"{len(flagged)} of {len(rows)} measurements spread more than 10% (marked in the "
            "site tables).")


def build() -> str:
    rows, e2e = data.bench_rows(), data.e2e_results()
    blocks = {"headline": headline(rows, e2e), "kernels": kernel_table(rows),
              "w4a16": w4_table(rows), "e2e": e2e_table(e2e), "cost": cost(),
              "spread": spread_note(rows)}
    text = README.read_text(encoding="utf-8")
    for name, body in blocks.items():
        text = re.sub(rf"(<!-- gen:{name} -->)(.*?)(<!-- /gen:{name} -->)",
                      lambda m, body=body: f"{m.group(1)}\n{body}\n{m.group(3)}", text, flags=re.S)
    README.write_text(text, encoding="utf-8", newline="\n")
    return text


if __name__ == "__main__":
    build()
