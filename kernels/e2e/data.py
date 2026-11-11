"""Model and dataset download. Runs on a CPU-only Modal container into the cache Volume."""

from __future__ import annotations

import os
from pathlib import Path

from kernels import config

CACHE = Path(os.environ.get("HF_HOME", "/cache/hf")).parent
WIKITEXT = CACHE / "data" / "wikitext2_test.txt"


def model_dir() -> str:
    from huggingface_hub import snapshot_download

    # local_files_only after the first download, so GPU containers never hit the network.
    return snapshot_download(config.MODEL_ID, local_files_only=True)


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
