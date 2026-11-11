# GPU ledger

Generated from `docs/gpu_ledger.jsonl` by `kernels/ledger.py`. Do not edit by hand.

Billed seconds are measured on the local side around each `modal run` call, so they
include scheduling and container startup. Estimated cost = GPU rate x 1.15 (CPU and
memory allowance). Rates: L4 $0.8/h, H100 $3.95/h.

**Totals:** 0.000 GPU-hours, est. $0.00 of $30 (dev $0.00 of $22, H100 0.0 of 90 min).

| # | Date | Purpose | GPU | nvidia-smi name | Billed s | Est. $ | Running $ | OK |
|---|---|---|---|---|---|---|---|---|
