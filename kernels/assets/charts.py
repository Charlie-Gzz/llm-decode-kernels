"""Every chart, generated from results JSON (no GPU needed). PNG (1600 px wide) + SVG.

Titles are computed from the data so they always state the current takeaway.
"""

from __future__ import annotations

import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kernels import config  # noqa: E402
from kernels.assets import data  # noqa: E402
from kernels.assets.style import (  # noqa: E402
    ACCENT,
    MUTED,
    ROOT,
    SERIES,
    TEXT,
    footnote,
    setup_matplotlib,
)

OUT = ROOT / "assets" / "charts"
DPI = 200  # 8 in x 200 dpi = 1600 px

# Approximate FLOPs per element for the memory-bound kernels (roofline x-position only).
ELEMWISE_FLOPS = {"rmsnorm": 5, "swiglu": 5}


def _save(fig, name: str) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{name}.png", OUT / f"{name}.svg"]
    fig.savefig(paths[0], dpi=DPI)
    fig.savefig(paths[1])
    paths[1].write_bytes(paths[1].read_bytes().replace(b"\r\n", b"\n"))  # LF on Windows too
    plt.close(fig)
    return paths


def _fmt_x(n: int) -> str:
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def _series_line(ax, xs, ys, impl, label_end=True):
    s = SERIES[impl]
    ax.plot(xs, ys, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"],
            markeredgecolor=ax.get_facecolor(), markeredgewidth=1.5, markersize=8)
    if label_end and ys:
        ax.annotate(s["label"], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    color=TEXT, fontsize=9, va="center")


def _gpu_note(gpu: str, extra: str) -> str:
    return f"{gpu} | fp16 | CUDA-graph timing, median of 3 sweeps | {extra}"


# ----------------------------------------------------------------------------- speedups
def speedup_elementwise(rows, gpu: str, kernel: str, fixed: dict, x_key: str, base="eager"):
    """Speedup of triton and torch.compile over eager, one panel per value of the other dim."""
    panel_key = "N" if kernel == "rmsnorm" else "I"
    panels = sorted({r["shape"][panel_key] for r in data.select(rows, gpu=gpu, kernel=kernel)})
    if not panels:
        return []
    fig, axes = plt.subplots(1, len(panels), figsize=(8, 4.6), sharey=True, squeeze=False)
    best = (0, None)
    vs_compile = []
    for ax, pv in zip(axes[0], panels):
        sel = data.select(rows, gpu=gpu, kernel=kernel, **fixed, **{panel_key: pv})
        xs = sorted({r["shape"][x_key] for r in sel})
        for x in xs:
            t, c = data.one(sel, impl="triton", **{x_key: x}), data.one(sel, impl="compile",
                                                                          **{x_key: x})
            if t and c:
                vs_compile.append(c["ms"] / t["ms"])
        for impl in ("triton", "compile"):
            pts = []
            for x in xs:
                b = data.one(sel, impl=base, **{x_key: x})
                r = data.one(sel, impl=impl, **{x_key: x})
                if b and r:
                    pts.append((x, b["ms"] / r["ms"]))
                    if impl == "triton" and b["ms"] / r["ms"] > best[0]:
                        best = (b["ms"] / r["ms"], x)
            if pts:
                _series_line(ax, [p[0] for p in pts], [p[1] for p in pts], impl, label_end=False)
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [_fmt_x(x) for x in xs])
        ax.set_xlabel(f"rows {x_key} (tokens)")
        ax.set_title(f"{panel_key} = {pv}", fontsize=11, color=MUTED, loc="left")
    axes[0][0].set_ylabel(f"speedup vs {SERIES[base]['label']}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, bbox_to_anchor=(0.06, 0.885))
    name = "Fused RMSNorm + residual" if kernel == "rmsnorm" else "Fused SwiGLU"
    rel = statistics.median(vs_compile) if vs_compile else 1
    if 0.95 <= rel <= 1.05:
        vs = "on par with torch.compile"
    else:
        vs = f"{rel:.2f}x vs torch.compile" if rel > 1 else f"{1 / rel:.2f}x slower than compile"
    fig.suptitle(f"{name}: up to {best[0]:.1f}x faster than eager, {vs}",
                 x=0.012, ha="left", color=TEXT, fontsize=14, fontweight="semibold")
    footnote(fig, _gpu_note(gpu, "higher is better"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    return _save(fig, f"speedup_{kernel}_{gpu.lower()}")


def speedup_w4a16(rows, gpu: str):
    sel = data.select(rows, gpu=gpu, kernel="w4a16")
    layers = []
    for r in sel:
        if r["shape"]["layer"] not in layers:
            layers.append(r["shape"]["layer"])
    if not layers:
        return []
    order = [lab for lab, _, _ in _w4_shapes() if lab in layers]
    ncol = 3
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 2.6 * nrow + 0.9), sharey=True,
                             squeeze=False)
    decode_speedups, prefill_speedups = [], []
    for i, layer in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        ls = data.select(sel, layer=layer)
        xs = sorted({r["shape"]["M"] for r in ls})
        for impl in ("triton", "cuda", "torchao", "compile", "naive"):
            pts = []
            for x in xs:
                b, r = data.one(ls, impl="fp16", M=x), data.one(ls, impl=impl, M=x)
                if b and r:
                    pts.append((x, b["ms"] / r["ms"]))
                    if impl == "triton":
                        (decode_speedups if x <= 16 else prefill_speedups).append(pts[-1][1])
            if pts:
                _series_line(ax, [p[0] for p in pts], [p[1] for p in pts], impl, label_end=False)
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [_fmt_x(x) for x in xs], fontsize=8)
        N, K = ls[0]["shape"]["N"], ls[0]["shape"]["K"]
        ax.set_title(f"{layer}  {N}x{K}", fontsize=10, color=MUTED, loc="left")
        if i % ncol == 0:
            ax.set_ylabel("speedup vs fp16")
        if i // ncol == nrow - 1:
            ax.set_xlabel("batch rows M")
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.875),
               fontsize=9)
    dmed = statistics.median(decode_speedups) if decode_speedups else 0
    title = f"4-bit Triton matmul is {dmed:.1f}x faster than fp16 cuBLAS at decode (median)"
    if prefill_speedups:
        pmed = statistics.median(prefill_speedups)
        verdict = "loses to cuBLAS" if pmed < 1 else "still wins"
        title += f"\nand {verdict} at prefill ({pmed:.2f}x median)"
    fig.suptitle(title, x=0.012, ha="left", color=TEXT, fontsize=13, fontweight="semibold")
    footnote(fig, _gpu_note(gpu, "group size 128 | higher is better"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    return _save(fig, f"speedup_w4a16_{gpu.lower()}")


def _w4_shapes():
    return config.W4_SHAPES


# ----------------------------------------------------------------------------- roofline
def _points(rows, gpu):
    """Representative point per kernel: (label, impl, row, flops)."""
    pts = []

    def add(label, impl, row, flops=None):
        if row:
            pts.append((label, impl, row, flops if flops is not None else row["flops"]))

    big_n = max((r["shape"]["N"] for r in data.select(rows, gpu=gpu, kernel="rmsnorm")),
                default=None)
    if big_n:
        r = max(data.select(rows, gpu=gpu, kernel="rmsnorm", impl="triton", residual=True,
                            N=big_n), key=lambda r: r["shape"]["M"], default=None)
        if r:
            add("RMSNorm+res", "triton", r,
                ELEMWISE_FLOPS["rmsnorm"] * r["shape"]["M"] * r["shape"]["N"])
    sw = max(data.select(rows, gpu=gpu, kernel="swiglu", impl="triton"),
             key=lambda r: r["shape"]["M"] * r["shape"]["I"], default=None)
    if sw:
        add("SwiGLU", "triton", sw, ELEMWISE_FLOPS["swiglu"] * sw["shape"]["M"] * sw["shape"]["I"])
    for M in (1, 16, 2048):
        add(f"W4A16 M={M}", "triton", data.one(rows, gpu=gpu, kernel="w4a16", impl="triton",
                                               layer="qwen_gate_up", M=M))
        add(f"fp16 M={M}", "fp16", data.one(rows, gpu=gpu, kernel="w4a16", impl="fp16",
                                            layer="qwen_gate_up", M=M))
    add("W4A16 CUDA M=1", "cuda", data.one(rows, gpu=gpu, kernel="w4a16", impl="cuda",
                                          layer="qwen_gate_up", M=1))
    return pts


def roofline(rows, gpu: str):
    import numpy as np

    spec = config.GPU_SPECS[gpu]
    pts = _points(rows, gpu)
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    bw, peak = spec["peak_gbps"] * 1e9, spec["peak_fp16_tflops"] * 1e12
    xs = np.logspace(-1, 4, 200)
    roof = np.minimum(bw * xs, peak) / 1e12
    ax.plot(xs, roof, color=MUTED, lw=1.5)
    ridge = peak / bw
    ax.annotate(f"memory roof {spec['peak_gbps']:.0f} GB/s", (ridge / 4, bw * ridge / 4e12),
                xytext=(-8, 8), textcoords="offset points", color=MUTED, fontsize=9, ha="right")
    ax.annotate(f"fp16 tensor roof {spec['peak_fp16_tflops']:.0f} TFLOPS", (ridge * 1.5,
                peak / 1e12), xytext=(0, 6), textcoords="offset points", color=MUTED, fontsize=9)
    best = None
    for label, impl, r, flops in pts:
        if impl == "cuda":  # sits on top of the Triton point; compared in its own chart
            continue
        intensity = flops / r["bytes"]
        tflops = flops / (r["ms"] * 1e-3) / 1e12
        s = SERIES[impl]
        hollow = impl == "fp16"
        ax.scatter([intensity], [tflops], s=70, marker=s["marker"],
                   facecolors="none" if hollow else s["color"], edgecolors=s["color"],
                   linewidths=1.6, zorder=3)
        # fp16 reference points get their label on the left so they do not collide with ours
        offset, ha = ((-8, -12), "right") if hollow else ((7, -3), "left")
        if label == "fp16 M=1":
            offset = (-8, 6)
        ax.annotate(label, (intensity, tflops), xytext=offset, textcoords="offset points",
                    color=TEXT if not hollow else MUTED, fontsize=9, ha=ha)
        if label == "W4A16 M=1":
            best = 100 * (r["bytes"] / (r["ms"] * 1e-3)) / bw
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_xlim(0.1, 1e4)
    if best and best >= 60:
        title = (f"Decode kernels run close to the {gpu}'s memory roof\n"
                 f"W4A16 at batch 1 reaches {best:.0f}% of peak bandwidth")
    elif best:
        title = (f"On the {gpu}, the 4-bit matmul falls well short of the memory roof\n"
                 f"W4A16 at batch 1: {best:.0f}% of peak (unpacking, not memory, limits it)")
    else:
        title = f"Decode kernels against the {gpu}'s roofline"
    ax.set_title(title, fontsize=13)
    footnote(fig, f"{gpu} | fp16 | filled = ours, hollow = fp16 cuBLAS | matmul 17920x1536 "
                  "(Qwen gate_up)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, f"roofline_{gpu.lower()}")


# ----------------------------------------------------------------------------- % of peak bw
def bandwidth_bars(rows, gpu: str):
    spec = config.GPU_SPECS[gpu]
    bars = []
    copy = data.one(rows, gpu=gpu, kernel="copy")
    for label, impl, r, _ in _points(rows, gpu):
        if impl != "fp16" and "2048" not in label:
            name = label.replace("W4A16", "W4A16 gate_up").replace(" CUDA", "")
            bars.append((name + (" (CUDA)" if impl == "cuda" else ""), impl, r["pct_peak_bw"]))
    for layer in ("qwen_down", "8b_up"):
        r = data.one(rows, gpu=gpu, kernel="w4a16", impl="triton", layer=layer, M=1)
        if r:
            bars.append((f"W4A16 {layer.replace('qwen_', '')} M=1", "triton", r["pct_peak_bw"]))
    if not bars:
        return []
    bars.sort(key=lambda b: b[2])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(bars) + 1.6))
    ys = range(len(bars))
    ax.barh(list(ys), [b[2] for b in bars], color=[SERIES[b[1]]["color"] for b in bars],
            height=0.62, edgecolor=ax.get_facecolor(), linewidth=2)
    for y, b in zip(ys, bars):
        ax.text(b[2] + 1, y, f"{b[2]:.0f}%", va="center", color=TEXT, fontsize=10,
                family="JetBrains Mono")
    ax.set_yticks(list(ys), [b[0] for b in bars])
    if copy:
        ax.axvline(copy["pct_peak_bw"], color=MUTED, ls="--", lw=1.2)
        ax.text(copy["pct_peak_bw"], len(bars) - 0.4, f" device copy {copy['pct_peak_bw']:.0f}%",
                color=MUTED, fontsize=9, va="bottom")
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"% of {spec['peak_gbps']:.0f} GB/s spec-sheet bandwidth")
    ax.grid(axis="y", visible=False)
    top = bars[-1]
    ax.set_title(f"{top[0]}: {top[2]:.0f}% of the {gpu}'s peak bandwidth",
                 fontsize=13)
    footnote(fig, f"{gpu} | fp16 | minimum bytes / time | dashed = plain device-to-device copy")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, f"bandwidth_{gpu.lower()}")


# ----------------------------------------------------------------------------- triton vs cuda
def triton_vs_cuda(rows, gpu: str):
    """% of peak bandwidth per layer, batch 1 and batch 16 side by side."""
    import numpy as np

    layers = [lab for lab, _, _ in _w4_shapes()]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), sharey=True)
    med = {}
    for ax, M in zip(axes, (1, 16)):
        pairs = []
        for layer in layers:
            t = data.one(rows, gpu=gpu, kernel="w4a16", impl="triton", layer=layer, M=M)
            c = data.one(rows, gpu=gpu, kernel="w4a16", impl="cuda", layer=layer, M=M)
            if t and c:
                pairs.append((layer, t["pct_peak_bw"], c["pct_peak_bw"]))
        if not pairs:
            return []
        y = np.arange(len(pairs))
        h = 0.38
        for off, idx, impl in ((-h / 2, 1, "triton"), (h / 2, 2, "cuda")):
            vals = [p[idx] for p in pairs]
            ax.barh(y + off, vals, height=h, color=SERIES[impl]["color"],
                    label=SERIES[impl]["label"], edgecolor=ax.get_facecolor(), linewidth=1.5)
            for yy, v in zip(y + off, vals):
                ax.text(v + 1, yy, f"{v:.0f}", va="center", color=TEXT, fontsize=8,
                        family="JetBrains Mono")
        ax.set_yticks(y, [p[0] for p in pairs])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"batch M = {M}", fontsize=11, color=MUTED, loc="left")
        ax.set_xlabel("% of peak bandwidth")
        med[M] = (statistics.median(p[1] for p in pairs), statistics.median(p[2] for p in pairs))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.86))
    (t1, c1), (t16, c16) = med[1], med[16]
    first = (f"Batch 1: Triton {t1:.0f}% vs CUDA C++ {c1:.0f}%" if t1 >= c1 else
             f"Batch 1: CUDA C++ {c1:.0f}% vs Triton {t1:.0f}%")
    fig.suptitle(f"{first} of peak bandwidth.\n"
                 f"Batch 16: Triton holds {t16:.0f}%, CUDA drops to {c16:.0f}% (medians)",
                 x=0.012, ha="left", color=TEXT, fontsize=13, fontweight="semibold")
    footnote(fig, _gpu_note(gpu, "int4 weights, group 128"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    return _save(fig, f"triton_vs_cuda_{gpu.lower()}")


# ----------------------------------------------------------------------------- end to end
def decode_tokens(e2e: dict):
    names = [("fp16", "fp16", "fp16"), ("ours", "triton", "Ours (Triton)"),
             ("ours_cuda", "cuda", "Ours (CUDA)")]
    gpus = [g for g in ("L4", "H100") if g in e2e]
    if not gpus:
        return []
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.4))
    present = [n for n in names if any(n[0] in e2e[g]["backends"] and
                                       "decode" in e2e[g]["backends"][n[0]] for g in gpus)]
    w = 0.8 / len(present)
    x = np.arange(len(gpus))
    speedups = []
    for i, (key, impl, label) in enumerate(present):
        vals = [e2e[g]["backends"].get(key, {}).get("decode", {}).get("tok_s", 0) for g in gpus]
        color = SERIES["fp16"]["color"] if key == "fp16" else SERIES[impl]["color"]
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w, color=color, label=label,
                      edgecolor=ax.get_facecolor(), linewidth=2)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                        color=TEXT, fontsize=10, family="JetBrains Mono")
    for g in gpus:
        b = e2e[g]["backends"]
        if "fp16" in b and "ours" in b:
            speedups.append((g, b["ours"]["decode"]["tok_s"] / b["fp16"]["decode"]["tok_s"]))
    ax.set_xticks(x, gpus)
    ax.set_ylabel("decode tokens / s (batch 1)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    s = ", ".join(f"{sp:.2f}x on {g}" for g, sp in speedups)
    ax.set_title(f"Qwen2.5-1.5B decodes faster with 4-bit kernels: {s}", fontsize=13)
    footnote(fig, "batch 1 | CUDA graphs both sides | embeddings and lm_head stay fp16")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, "decode_tokens")


def decode_breakdown(e2e: dict):
    """Per-token decode time split into the swapped layers and everything else."""
    gpus = [g for g in ("L4", "H100") if g in e2e]
    rows = []
    for g in gpus:
        for key, label in (("fp16", "fp16"), ("ours", "4-bit, Triton")):
            r = e2e[g]["backends"].get(key, {})
            if "decode" in r and "layers_only_ms" in r:
                total = r["decode"]["ms_per_token"]
                rows.append((f"{g}  {label}", r["layers_only_ms"], total - r["layers_only_ms"]))
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(8, 1.9 + 0.62 * len(rows)))
    ys = list(range(len(rows)))
    layers = [r[1] for r in rows]
    rest = [r[2] for r in rows]
    ax.barh(ys, layers, color=ACCENT, height=0.6, label="swapped layers (projections, norms, act)",
            edgecolor=ax.get_facecolor(), linewidth=2)
    ax.barh(ys, rest, left=layers, color=MUTED, height=0.6,
            label="everything else (fp16 lm_head, attention, small ops)",
            edgecolor=ax.get_facecolor(), linewidth=2)
    for y, (_, a, b) in zip(ys, rows):
        ax.text(a + b + 0.15, y, f"{a + b:.1f} ms", va="center", color=TEXT, fontsize=10,
                family="JetBrains Mono")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ms per decode token (batch 1)")
    ax.set_xlim(0, max(a + b for _, a, b in rows) * 1.15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=1, fontsize=9,
               bbox_to_anchor=(0.012, 0.86))
    f, o = rows[0], rows[1] if len(rows) > 1 else rows[0]
    fig.suptitle(f"Swapped layers got {f[1] / o[1]:.1f}x faster; the untouched {o[2]:.1f} ms "
                 f"is now {100 * o[2] / (o[1] + o[2]):.0f}% of a token", x=0.012, ha="left",
                 color=TEXT, fontsize=13, fontweight="semibold")
    footnote(fig, "Qwen2.5-1.5B | CUDA graphs | layers timed alone, rest = total minus layers")
    fig.tight_layout(rect=(0, 0.05, 1, 0.72))
    return _save(fig, "decode_breakdown")


def build_all() -> list:
    setup_matplotlib()
    rows = data.bench_rows()
    e2e = data.e2e_results()
    made = []
    for gpu in sorted({r["gpu"] for r in rows}):
        made += speedup_elementwise(rows, gpu, "rmsnorm", {"residual": True}, "M")
        made += speedup_elementwise(rows, gpu, "swiglu", {}, "M")
        made += speedup_w4a16(rows, gpu)
        made += roofline(rows, gpu)
        made += bandwidth_bars(rows, gpu)
        made += triton_vs_cuda(rows, gpu)
    made += decode_tokens(e2e)
    made += decode_breakdown(e2e)
    return made


if __name__ == "__main__":
    for p in build_all():
        print(p.relative_to(ROOT))
