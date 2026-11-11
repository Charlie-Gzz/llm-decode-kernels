"""Diagnostic: does a tiny slim-image L4 function schedule faster than the full CUDA image?

    python scripts/l4_probe.py      (records itself in the GPU ledger)
"""

import subprocess
import time

import modal

app = modal.App("llm-decode-kernels-probe")


@app.function(gpu="L4", timeout=120, max_containers=1, image=modal.Image.debian_slim())
def probe() -> dict:
    t0 = time.time()
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu": out, "in_container_s": time.time() - t0}


if __name__ == "__main__":
    from kernels import ledger

    ledger.check_budget("L4", final=False, expected_s=120)
    t0 = time.monotonic()
    res = {}
    with modal.enable_output(), app.run():
        try:
            res = probe.remote()
        finally:
            wall = time.monotonic() - t0
            billed = 25.0 if res else wall  # container is ~instant; 20 s startup allowance + 5
            ledger.record("l4 probe (slim image)", "L4", billed, res.get("gpu", "?").split(",")[0],
                          False, bool(res), f"queue+run wall {wall:.0f}s")
    print(res, f"wall {wall:.0f}s")
