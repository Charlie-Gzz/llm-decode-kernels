"""One look for every chart, the site and the animation (PLAN.md section 6)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "assets" / "fonts"

BG = "#0B0F17"
PANEL = "#131A26"
BORDER = "#1F2937"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#A3E635"

# Series identity. The plan's palette fails the colorblind check for sky vs violet and gray vs
# violet (validated with the dataviz validator), so every series also has its own marker and
# line style, and charts carry direct labels. Color is never the only cue.
SERIES = {
    "triton": {"color": "#A3E635", "label": "Ours (Triton)", "marker": "o", "ls": "-"},
    "cuda": {"color": "#38BDF8", "label": "Ours (CUDA C++)", "marker": "s", "ls": "-"},
    "compile": {"color": "#A78BFA", "label": "torch.compile", "marker": "^", "ls": "--"},
    "eager": {"color": "#9CA3AF", "label": "PyTorch eager", "marker": "D", "ls": ":"},
    "fp16": {"color": "#9CA3AF", "label": "fp16 cuBLAS", "marker": "D", "ls": ":"},
    "torchao": {"color": "#F59E0B", "label": "torchao int4", "marker": "v", "ls": "-."},
    "naive": {"color": "#6B7280", "label": "dequant + matmul", "marker": "x", "ls": ":"},
}


def setup_matplotlib():
    import matplotlib as mpl
    from matplotlib import font_manager

    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "semibold",
        "axes.titlesize": 15,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": BORDER,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": TEXT,
        "font.family": "Inter",
        "font.size": 11,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "svg.fonttype": "path",  # SVGs in <img> cannot use page web fonts
    })


def footnote(fig, text: str):
    """Small muted line at the bottom: which GPU, dtype and shapes a chart covers."""
    fig.text(0.012, 0.012, text, color=MUTED, fontsize=9, family="JetBrains Mono",
             ha="left", va="bottom")
