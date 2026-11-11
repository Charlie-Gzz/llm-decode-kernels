"""Version-sensitive facts in one place: GPU peaks, Modal prices, budget caps, pinned versions.

Sources (checked 2026-11-11):
- L4 peaks: NVIDIA L4 product page (300 GB/s, FP16 tensor 242 TFLOPS with sparsity -> 121 dense).
- H100 SXM peaks: NVIDIA H100 product page / datasheet (3.35 TB/s, FP16 tensor 1979 TFLOPS with
  sparsity -> 989 dense). Modal's pricing page lists its H100 as "H100 SXM5".
- Prices: `modal billing rates` and modal.com/pricing.
"""

from __future__ import annotations

# Peak numbers from NVIDIA spec sheets. Dense (no 2:4 sparsity) tensor-core throughput.
GPU_SPECS = {
    "L4": {
        "peak_gbps": 300.0,
        "peak_fp16_tflops": 121.0,
        "l2_mb": 48,
        "sms": 58,
        "source": "https://www.nvidia.com/en-us/data-center/l4/",
    },
    "H100": {
        "peak_gbps": 3350.0,
        "peak_fp16_tflops": 989.0,
        "l2_mb": 50,
        "sms": 132,
        "source": "https://www.nvidia.com/en-us/data-center/h100/",
    },
}

# Modal GPU strings. "H100!" stops Modal from silently upgrading the request to an H200.
MODAL_GPU = {"L4": "L4", "H100": "H100!"}

# USD per hour (modal billing rates, 2026-11-11). Ledger adds 15% for CPU and memory.
GPU_USD_PER_HOUR = {"L4": 0.80, "H100": 3.95}
CPU_MEM_ALLOWANCE = 0.15

# Budget caps from PLAN.md section 1.1.
BUDGET_TOTAL_USD = 30.0
BUDGET_RESERVE_USD = 8.0  # kept for M6 final runs
BUDGET_DEV_USD = BUDGET_TOTAL_USD - BUDGET_RESERVE_USD
BUDGET_TOTAL_GPU_HOURS = 25.0
H100_MAX_SECONDS = 1.5 * 3600

# Pinned versions for the Modal image (PyPI, 2026-11-11).
CUDA_IMAGE = "nvidia/cuda:13.2.1-devel-ubuntu24.04"
PIP_PACKAGES = [
    "torch==2.15.0",
    "triton==3.9.0",
    "torchao==0.18.0",
    "transformers==5.22.2",
    "datasets==5.2.0",
    "accelerate==1.15.0",
    "numpy",
    "pytest",
    "ninja",
]

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GROUP_SIZE = 128


def gpu_key(name: str) -> str:
    """Map an nvidia-smi name like 'NVIDIA H100 80GB HBM3' to a GPU_SPECS key."""
    for key in GPU_SPECS:
        if key in name:
            return key
    raise ValueError(f"unknown GPU {name!r}")
