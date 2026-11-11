"""Static results site in site/: overview, one page per kernel, end-to-end page.

Plain HTML + Tailwind (CDN) + a little vanilla JS. Everything is generated from results JSON and
the chart files, so `python -m kernels.assets` rebuilds it with no GPU.
"""

from __future__ import annotations

import html
import json
import re
import shutil

from kernels import config, ledger
from kernels.assets import data
from kernels.assets.style import ACCENT, BG, BORDER, MUTED, PANEL, ROOT, TEXT

SITE = ROOT / "site"
CHARTS = ROOT / "assets" / "charts"
REPO = "https://github.com/Charlie-Gzz/llm-decode-kernels"

PAGES = [("index.html", "Overview"), ("rmsnorm.html", "RMSNorm"), ("swiglu.html", "SwiGLU"),
         ("w4a16.html", "W4A16 matmul"), ("e2e.html", "End to end"), ("cost.html", "Cost")]


def esc(s) -> str:
    return html.escape(str(s))


def layout(file: str, title: str, body: str) -> str:
    nav = "".join(
        f'<a href="{f}" class="px-3 py-1.5 rounded-md text-sm '
        f'{"bg-panel text-ink border border-line" if f == file else "text-muted hover:text-ink"}"'
        f'>{esc(t)}</a>' for f, t in PAGES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} | llm-decode-kernels</title>
<meta name="description" content="Triton and CUDA kernels for LLM decoding, benchmarked on L4 and H100.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com/3.4.16"></script>
<script>
tailwind.config = {{ theme: {{ extend: {{
  colors: {{ bg: "{BG}", panel: "{PANEL}", line: "{BORDER}", ink: "{TEXT}", muted: "{MUTED}",
            accent: "{ACCENT}" }},
  fontFamily: {{ sans: ["Inter", "sans-serif"], mono: ["JetBrains Mono", "monospace"] }} }} }} }};
</script>
<style>
  body {{ background: {BG}; color: {TEXT}; }}
  pre code {{ font-size: 12.5px; line-height: 1.55; }}
  table td, table th {{ padding: 6px 10px; border-bottom: 1px solid {BORDER}; text-align: right; }}
  table td:first-child, table th:first-child {{ text-align: left; }}
  .chart img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid {BORDER}; }}
</style>
</head>
<body class="font-sans antialiased">
<header class="border-b border-line">
  <div class="max-w-6xl mx-auto px-4 py-4 flex flex-wrap items-center gap-3 justify-between">
    <a href="index.html" class="flex items-center gap-2 font-semibold">
      <span class="inline-block w-3 h-3 rounded-sm bg-accent"></span>llm-decode-kernels</a>
    <nav class="flex flex-wrap gap-1">{nav}</nav>
  </div>
</header>
<main class="max-w-6xl mx-auto px-4 py-8 space-y-10">
{body}
</main>
<footer class="border-t border-line mt-10">
  <div class="max-w-6xl mx-auto px-4 py-6 text-sm text-muted">
    Generated from <code class="font-mono">results/raw/*.json</code> by
    <code class="font-mono">python -m kernels.assets</code>. Source on
    <a class="text-accent" href="{REPO}">GitHub</a>.
  </div>
</footer>
</body>
</html>
"""


def card(value: str, label: str, sub: str = "") -> str:
    return (f'<div class="bg-panel border border-line rounded-xl p-5">'
            f'<div class="font-mono text-3xl font-semibold text-accent">{esc(value)}</div>'
            f'<div class="mt-1 text-ink">{esc(label)}</div>'
            f'<div class="mt-1 text-xs text-muted">{esc(sub)}</div></div>')


def chart(name: str, alt: str) -> str:
    if not (SITE / "charts" / f"{name}.svg").exists():
        return f'<p class="text-muted text-sm">Chart {esc(name)} not generated yet (no results).</p>'
    return (f'<figure class="chart"><a href="charts/{name}.png">'
            f'<img src="charts/{name}.svg" alt="{esc(alt)}" loading="lazy"></a></figure>')


def section(title: str, inner: str, intro: str = "") -> str:
    p = f'<p class="text-muted max-w-3xl">{intro}</p>' if intro else ""
    return f'<section class="space-y-4"><h2 class="text-xl font-semibold">{esc(title)}</h2>{p}{inner}</section>'


def code_excerpt(path, func: str, max_lines: int = 60) -> str:
    """The source of one function, for the kernel pages."""
    src = (ROOT / path).read_text()
    m = re.search(rf"(@triton\.jit\n)?def {func}\(.*?(?=\n\n\n|\Z)", src, re.S)
    if not m:
        m = re.search(rf"[^\n]*{func}.*?(?=\n\}}\n)", src, re.S)
    text = m.group(0) if m else f"# {func} not found"
    lines = text.splitlines()[:max_lines]
    return (f'<div class="bg-panel border border-line rounded-xl overflow-x-auto">'
            f'<div class="px-4 py-2 text-xs text-muted border-b border-line font-mono">{esc(path)}</div>'
            f'<pre class="p-4"><code class="font-mono">{esc(chr(10).join(lines))}</code></pre></div>')


def table(headers, rows) -> str:
    th = "".join(f"<th class='text-muted font-medium'>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td class='font-mono'>{c}</td>" for c in r) + "</tr>"
                  for r in rows)
    return (f'<div class="overflow-x-auto bg-panel border border-line rounded-xl">'
            f'<table class="w-full text-sm"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')


def _us(r):
    return f"{r['ms'] * 1e3:.1f}" if r else "-"


def results_table(rows, kernel, shape_keys, impls, base):
    """One row per shape; time per impl (us), speedup of triton vs base, % peak bw of triton."""
    sel = data.select(rows, kernel=kernel)
    shapes = []
    for r in sel:
        key = tuple(r["shape"].get(k) for k in shape_keys)
        if key not in shapes:
            shapes.append(key)
    out = []
    for key in shapes:
        match = dict(zip(shape_keys, key))
        cells = [", ".join(f"{k}={v}" for k, v in match.items())]
        by = {i: data.one(sel, impl=i, **match) for i in impls}
        cells += [_us(by[i]) for i in impls]
        t, b = by.get("triton"), by.get(base)
        cells.append(f"{b['ms'] / t['ms']:.2f}x" if t and b else "-")
        flag = " *" if t and t.get("flag_spread") else ""
        cells.append(f"{t['pct_peak_bw']:.0f}%{flag}" if t else "-")
        out.append(cells)
    return table(["shape", *[f"{i} (us)" for i in impls], f"speedup vs {base}", "triton % peak bw"],
                 out)


def profile_notes(prof: dict, prefix: str) -> str:
    if not prof:
        return '<p class="text-muted text-sm">No profile yet.</p>'
    ks = [k for k in prof["kernels"] if prefix in json.dumps(k)]
    if not ks:
        return '<p class="text-muted text-sm">No profiled launches for this kernel.</p>'
    if prof["tool"] == "ncu":
        rows = [[esc(k.get("nvtx") or k["kernel"][:40]),
                 esc(k.get("gpu__time_duration.sum", "")),
                 esc(k.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "")),
                 esc(k.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "")),
                 esc(k.get("sm__warps_active.avg.pct_of_peak_sustained_active", "")),
                 esc(k.get("launch__registers_per_thread", ""))] for k in ks]
        return table(["launch", f"time ({prof.get('units', {}).get('gpu__time_duration.sum', '')})",
                      "DRAM % peak", "SM % peak", "occupancy %", "regs/thread"], rows)
    rows = [[esc(k["case"]), esc(k["kernel"][:60]), f"{k['us']:.1f}"] for k in ks]
    return table(["case", "kernel", "GPU time (us)"], rows)


# ----------------------------------------------------------------------------- pages
def page_index(rows, e2e, gpus):
    cards = []
    l4 = "L4"
    r = data.one(rows, gpu=l4, kernel="w4a16", impl="triton", layer="8b_up", M=1)
    if r:
        cards.append(card(f"{r['pct_peak_bw']:.0f}%", "of L4 peak bandwidth",
                          "W4A16 Triton, batch 1, 14336x4096"))
    if l4 in e2e and "ours" in e2e[l4]["backends"]:
        b = e2e[l4]["backends"]
        sp = b["ours"]["decode"]["tok_s"] / b["fp16"]["decode"]["tok_s"]
        cards.append(card(f"{sp:.2f}x", "decode tokens/s vs fp16",
                          f"Qwen2.5-1.5B, L4, {b['ours']['decode']['tok_s']:.0f} vs "
                          f"{b['fp16']['decode']['tok_s']:.0f} tok/s"))
        wq, wf = b["ours"]["weights"]["linear_layers"], b["fp16"]["weights"]["linear_layers"]
        cards.append(card(f"{wf / wq:.1f}x", "smaller linear-layer weights",
                          f"{wf / 1e9:.2f} GB fp16 -> {wq / 1e9:.2f} GB 4-bit"))
        cards.append(card(f"{b['ours']['ppl'] - b['fp16']['ppl']:+.2f}", "WikiText-2 perplexity",
                          f"{b['fp16']['ppl']:.2f} fp16 -> {b['ours']['ppl']:.2f} RTN 4-bit"))
    t = ledger.totals()
    cards.append(card(f"${t['usd']:.2f}", "total GPU spend (est.)",
                      f"{t['gpu_hours']:.2f} GPU-hours on Modal"))
    hero = f"""
<section class="space-y-5">
  <h1 class="text-3xl md:text-4xl font-bold max-w-3xl">Custom GPU kernels for the slow parts of LLM decoding</h1>
  <p class="text-muted max-w-3xl text-lg">Fused RMSNorm, fused SwiGLU, and a 4-bit weight matmul in
  Triton, plus a CUDA C++ version of the decode matmul. Each one is tested against PyTorch,
  measured against the GPU's limits, and swapped into Qwen2.5-1.5B.</p>
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-{min(len(cards), 5)} gap-3">{''.join(cards)}</div>
</section>"""
    explainer = section("How the 4-bit matmul works",
                        '<div class="bg-panel border border-line rounded-xl overflow-hidden">'
                        '<iframe src="explainer.html?embed" title="W4A16 explainer" '
                        'class="w-full" style="aspect-ratio: 16/9.6; border:0"></iframe></div>')
    charts = "".join(
        section(f"Results on {g}", '<div class="grid md:grid-cols-2 gap-4">'
                + chart(f"speedup_w4a16_{g.lower()}", "W4A16 speedup")
                + chart(f"roofline_{g.lower()}", "Roofline")
                + chart(f"bandwidth_{g.lower()}", "Bandwidth")
                + chart(f"triton_vs_cuda_{g.lower()}", "Triton vs CUDA") + "</div>")
        for g in gpus)
    e2e_sec = section("End to end", chart("decode_tokens", "Decode tokens per second"))
    return hero + explainer + e2e_sec + charts


def page_elementwise(rows, gpus, prof, kernel):
    if kernel == "rmsnorm":
        title = "Fused RMSNorm + residual add"
        intro = ("Every decoder block adds its output to the residual stream and then normalizes. "
                 "Unfused, that is an add kernel, a norm kernel, and two extra trips through "
                 "memory. The fused kernel reads x and the residual once, writes the new residual "
                 "and the normalized output, and does the reduction in fp32. Numerics match "
                 "Hugging Face's Qwen2RMSNorm step for step.")
        code = code_excerpt("kernels/rmsnorm.py", "_rmsnorm_kernel")
        keys = ["M", "N", "residual"]
    else:
        title = "Fused SwiGLU"
        intro = ("The MLP activation is silu(gate) * up. With a merged gate/up projection the "
                 "kernel reads both halves of one tensor in place and writes the product in one "
                 "pass: 3 bytes moved per 2 read, the minimum possible.")
        code = code_excerpt("kernels/swiglu.py", "_swiglu_kernel")
        keys = ["M", "I"]
    body = f'<h1 class="text-3xl font-bold">{title}</h1><p class="text-muted max-w-3xl">{intro}</p>'
    body += section("Kernel", code)
    for g in gpus:
        r = data.select(rows, gpu=g)
        body += section(f"{g}", chart(f"speedup_{kernel}_{g.lower()}", f"{kernel} speedup")
                        + results_table(r, kernel, keys, ["triton", "eager", "compile"], "eager"))
        body += section(f"Profile on {g}", profile_notes(prof.get(g), kernel))
    body += ('<p class="text-xs text-muted">Times are CUDA-graph medians over 3 sweeps with '
             'rotating buffers (working set at least 4x L2). * = more than 10% spread between '
             'sweeps.</p>')
    return body


def page_w4a16(rows, gpus, prof):
    intro = ("Decode multiplies a few activation rows by every weight matrix in the model, so it "
             "is limited by how fast the weights stream from DRAM. Storing weights as 4-bit "
             "integers (8 per int32, fp16 scale and zero per group of 128) cuts those bytes ~3.6x. "
             "The kernel dequantizes in registers and feeds tensor cores. For decode, split-K "
             "spreads one small matmul over the whole GPU. A CUDA C++ version of the decode case "
             "is compared side by side.")
    body = f'<h1 class="text-3xl font-bold">W4A16 matmul</h1><p class="text-muted max-w-3xl">{intro}</p>'
    body += section("Triton kernel", code_excerpt("kernels/w4a16.py", "_w4a16_kernel", 80))
    body += section("CUDA C++ decode kernel (inner loop)",
                    code_excerpt("kernels/cuda/w4a16_decode.cu", "w4a16_decode_kernel", 70))
    note = ROOT / "docs" / "TRITON_VS_CUDA.md"
    if note.exists():
        body += section("Triton or CUDA?", '<div class="text-muted max-w-3xl space-y-2">'
                        + "".join(f"<p>{esc(p)}</p>" for p in note.read_text().split("\n\n")[1:])
                        + "</div>")
    for g in gpus:
        r = data.select(rows, gpu=g)
        body += section(g, chart(f"speedup_w4a16_{g.lower()}", "speedup")
                        + '<div class="grid md:grid-cols-2 gap-4">'
                        + chart(f"roofline_{g.lower()}", "roofline")
                        + chart(f"triton_vs_cuda_{g.lower()}", "triton vs cuda") + "</div>"
                        + results_table(r, "w4a16", ["layer", "M"],
                                        ["triton", "cuda", "fp16", "torchao", "compile", "naive"],
                                        "fp16"))
        body += section(f"Profile on {g}", profile_notes(prof.get(g), "w4a16"))
    return body


def page_e2e(e2e):
    body = '<h1 class="text-3xl font-bold">End to end: Qwen2.5-1.5B-Instruct</h1>'
    body += ('<p class="text-muted max-w-3xl">All attention and MLP projections use round-to-'
             'nearest 4-bit weights (group 128) and our kernels. Embeddings, lm_head, attention '
             '(SDPA) and RoPE stay in fp16 on both sides. Both sides run the same small runner '
             'under CUDA graphs with a static KV cache, so the only difference is the kernels.</p>')
    body += section("Decode speed", chart("decode_tokens", "decode tokens/s"))
    for g, res in e2e.items():
        b = res["backends"]
        rows = []
        for name, r in b.items():
            d = r.get("decode", {})
            pre = ", ".join(f"{p['T']}: {p['tok_s']:.0f}" for p in r.get("prefill", []))
            rows.append([esc(name), f"{d.get('tok_s', 0):.1f}" if d else "-",
                         f"{r.get('layers_only_ms', 0):.2f}" if "layers_only_ms" in r else "-",
                         esc(pre or "-"), f"{r['ppl']:.3f}",
                         f"{r['weights']['linear_layers'] / 1e9:.2f}"])
        body += section(f"{g}", table(["backend", "decode tok/s", "swapped layers ms/token",
                                       "prefill tok/s (T: tok/s)", "WikiText-2 ppl",
                                       "linear weights GB"], rows))
        f16 = b.get("fp16", {})
        if "parity_max_abs_logit_diff" in f16:
            greedy = ""
            if "parity_first_divergence" in f16:
                d, n = f16["parity_first_divergence"], f16["parity_compared_tokens"]
                greedy = (f" Greedy tokens are identical for all {n} steps." if d == n else
                          f" Greedy tokens are identical for the first {d} of {n} steps, then a "
                          "near-tie flips and the texts drift apart (expected from fp16 "
                          "rounding order).")
            body += (f'<p class="text-sm text-muted">Runner check against Hugging Face: max abs '
                     f'logit difference {f16["parity_max_abs_logit_diff"]:.3f} on the prompt; '
                     f'perplexity {f16["ppl"]:.3f} vs HF {res["hf_ppl"]:.3f}.{greedy}</p>')
        if "fp16" in b and "ours" in b:
            outs = ""
            for a, o in zip(b["fp16"]["greedy"], b["ours"]["greedy"]):
                outs += (f'<div class="bg-panel border border-line rounded-xl p-4 space-y-3">'
                         f'<div class="text-sm text-muted">{esc(a["prompt"])}</div>'
                         f'<div class="grid md:grid-cols-2 gap-3 text-sm">'
                         f'<div><div class="text-xs text-muted mb-1">fp16</div>{esc(a["text"])}</div>'
                         f'<div><div class="text-xs text-accent mb-1">4-bit, our kernels</div>'
                         f'{esc(o["text"])}</div></div></div>')
            body += section(f"Greedy outputs ({g})", f'<div class="space-y-3">{outs}</div>')
    return body


def page_cost():
    t = ledger.totals()
    rows = [[esc(r["date"]), esc(r["purpose"]), esc(r["gpu"]), f"{r['billed_s']:.0f}",
             f"{r['est_usd']:.3f}", "yes" if r["ok"] else "no"] for r in ledger.load()]
    body = '<h1 class="text-3xl font-bold">What this project cost</h1>'
    body += ('<div class="grid sm:grid-cols-3 gap-3">'
             + card(f"${t['usd']:.2f}", "estimated GPU spend", f"cap ${config.BUDGET_TOTAL_USD:.0f}")
             + card(f"{t['gpu_hours']:.2f} h", "GPU time", f"cap {config.BUDGET_TOTAL_GPU_HOURS:.0f} h")
             + card(f"{t['h100_s'] / 60:.0f} min", "H100 time", "one final session") + "</div>")
    body += section("Every GPU run", table(["date", "purpose", "GPU", "billed s", "est. $", "ok"],
                                           rows),
                    "Billed seconds are container time plus a 20 s startup allowance; estimates add "
                    "15% for CPU and memory. Reconciled against Modal's billing report.")
    return body


def build() -> list:
    SITE.mkdir(exist_ok=True)
    (SITE / "charts").mkdir(exist_ok=True)
    for f in CHARTS.glob("*"):
        shutil.copy2(f, SITE / "charts" / f.name)
    gif = ROOT / "assets" / "explainer.gif"
    if gif.exists():
        shutil.copy2(gif, SITE / "explainer.gif")
    rows = data.bench_rows()
    e2e = data.e2e_results()
    prof = data.profile_results()
    gpus = [g for g in ("L4", "H100") if any(r["gpu"] == g for r in rows)]
    pages = {
        "index.html": ("Overview", page_index(rows, e2e, gpus)),
        "rmsnorm.html": ("RMSNorm", page_elementwise(rows, gpus, prof, "rmsnorm")),
        "swiglu.html": ("SwiGLU", page_elementwise(rows, gpus, prof, "swiglu")),
        "w4a16.html": ("W4A16 matmul", page_w4a16(rows, gpus, prof)),
        "e2e.html": ("End to end", page_e2e(e2e)),
        "cost.html": ("Cost", page_cost()),
    }
    out = []
    for f, (title, body) in pages.items():
        (SITE / f).write_text(layout(f, title, body), encoding="utf-8", newline="\n")
        out.append(SITE / f)
    return out


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(ROOT))
