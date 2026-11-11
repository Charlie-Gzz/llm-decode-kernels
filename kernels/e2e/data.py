"""Model and dataset download. Runs on a CPU-only Modal container into the cache Volume."""

from __future__ import annotations

import os
from pathlib import Path

from kernels import config

CACHE = Path(os.environ.get("HF_HOME", "/cache/hf")).parent
WIKITEXT = CACHE / "data" / "wikitext2_test.txt"


def model_dir() -> str:
    """The cached snapshot folder. Read straight from the HF cache layout: GPU containers never
    touch the network, and newer huggingface_hub refuses `local_files_only` on a snapshot that
    was downloaded with allow_patterns (it calls it incomplete)."""
    repo = CACHE / "hf" / "hub" / ("models--" + config.MODEL_ID.replace("/", "--"))
    snaps = sorted((repo / "snapshots").glob("*/config.json"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(f"no cached snapshot under {repo}; run the download task first")
    return str(snaps[-1].parent)


def download_all() -> dict:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    path = snapshot_download(config.MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    WIKITEXT.parent.mkdir(parents=True, exist_ok=True)
    WIKITEXT.write_text("\n\n".join(ds["text"]))
    return {
        "model_path": path,
        "model_bytes": size,
        "wikitext_chars": WIKITEXT.stat().st_size,
        "summary": f"model {size / 1e9:.2f} GB, wikitext {WIKITEXT.stat().st_size / 1e6:.1f} MB",
    }
