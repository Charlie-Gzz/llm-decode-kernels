# GPU ledger

Generated from `docs/gpu_ledger.jsonl` by `kernels/ledger.py`. Do not edit by hand.

Billed seconds = the container's own wall clock (from Python start) + 20 s for image
load before Python starts. Queue time is not billed by Modal and not counted. If a run
returns no result, local wall time is used instead (an over-estimate).
Estimated cost = GPU rate x 1.15 (CPU and
memory allowance). Rates: L4 $0.8/h, H100 $3.95/h.

**Totals:** 1.506 GPU-hours, est. $1.39 of $30 (dev $1.09 of $22, H100 0.0 of 90 min).

| # | Date | Purpose | GPU | nvidia-smi name | Billed s | Est. $ | Running $ | OK | Note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-11 17:44 UTC | smoke | L4 | NVIDIA L4 | 30 | 0.008 | 0.008 | yes | corrected: local wall 929s included ~15 min queue wait; container ran 10s + 20s allowance |
| 2 | 2026-11-11 20:18 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests'}, {'task | L4 | NVIDIA L4 | 1496 | 0.382 | 0.390 | no | local wall 3851s |
| 3 | 2026-11-11 20:53 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 1497 | 0.383 | 0.772 | no | local wall 1892s |
| 4 | 2026-11-11 21:11 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'e2e', 'backends': ['fp16', 'ours' | L4 | NVIDIA L4 | 709 | 0.181 | 0.954 | no | local wall 966s |
| 5 | 2026-11-11 21:18 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'e2e', 'backends': ['fp16', 'ours' | L4 | NVIDIA L4 | 300 | 0.077 | 1.030 | yes | local wall 287s |
| 6 | 2026-11-11 22:42 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 218 | 0.056 | 1.086 | yes | local wall 4868s |
| 7 | 2026-11-11 23:18 UTC | multi {'deadline_s': 1560, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 1172 | 0.299 | 1.385 | yes | local wall 1927s |

## Reconciliation with Modal billing

Last checked 2026-11-11 23:20 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$1.2798** (CPU $0.0676, L4 $1.1813, Memory $0.0302, Network Egress $0.0006)
- Ledger estimate: **$1.3855**
