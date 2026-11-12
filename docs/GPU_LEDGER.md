# GPU ledger

Generated from `docs/gpu_ledger.jsonl` by `kernels/ledger.py`. Do not edit by hand.

Billed seconds = the container's own wall clock (from Python start) + 20 s for image
load before Python starts. Queue time is not billed by Modal and not counted. If a run
returns no result, local wall time is used instead (an over-estimate).
Estimated cost = GPU rate x 1.15 (CPU and
memory allowance). Rates: L4 $0.8/h, H100 $3.95/h.

**Totals:** 3.123 GPU-hours, est. $5.90 of $30 (dev $1.53 of $22, H100 50.2 of 90 min).

| # | Date | Purpose | GPU | nvidia-smi name | Billed s | Est. $ | Running $ | OK | Note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-11 17:44 UTC | smoke | L4 | NVIDIA L4 | 30 | 0.008 | 0.008 | yes | corrected: local wall 929s included ~15 min queue wait; container ran 10s + 20s allowance |
| 2 | 2026-11-11 20:18 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests'}, {'task | L4 | NVIDIA L4 | 1496 | 0.382 | 0.390 | no | local wall 3851s |
| 3 | 2026-11-11 20:53 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 1497 | 0.383 | 0.772 | no | local wall 1892s |
| 4 | 2026-11-11 21:11 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'e2e', 'backends': ['fp16', 'ours' | L4 | NVIDIA L4 | 709 | 0.181 | 0.954 | no | local wall 966s |
| 5 | 2026-11-11 21:18 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'e2e', 'backends': ['fp16', 'ours' | L4 | NVIDIA L4 | 300 | 0.077 | 1.030 | yes | local wall 287s |
| 6 | 2026-11-11 22:42 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 218 | 0.056 | 1.086 | yes | local wall 4868s |
| 7 | 2026-11-11 23:18 UTC | multi {'deadline_s': 1560, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 1172 | 0.299 | 1.385 | yes | local wall 1927s |
| 8 | 2026-11-11 23:58 UTC | multi {'deadline_s': 1560, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | L4 | NVIDIA L4 | 1070 | 0.274 | 1.659 | yes | local wall 2307s |
| 9 | 2026-11-12 00:38 UTC | multi {'deadline_s': 1560, 'steps': [{'task': 'pytest', 'args': 'tests/test_w4a1 | H100 | NVIDIA H100 80GB HBM3 | 1405 | 1.773 | 3.432 | yes | local wall 2374s |
| 10 | 2026-11-12 00:38 UTC | h100_a preempted first container (estimated) | H100 | NVIDIA H100 80GB HBM3 | 1009 | 1.273 | 4.705 | no | Modal preempted the container mid-sweep; Modal re-ran the input. Estimated from local wall 2374s minus second container 1385s, +20s allowance. Reconcile with billing. |
| 11 | 2026-11-12 00:48 UTC | multi {'deadline_s': 1560, 'steps': [{'task': 'e2e', 'backends': ['fp16', 'ours' | H100 | NVIDIA H100 80GB HBM3 | 598 | 0.754 | 5.459 | yes | local wall 582s |
| 12 | 2026-11-12 01:23 UTC | e2e {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'backends': ['fp16', 'ours', 'ours_c | L4 | NVIDIA L4 | 512 | 0.131 | 5.590 | yes | local wall 1769s |
| 13 | 2026-11-12 02:43 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_fp8. | L4 | NVIDIA L4 | 807 | 0.206 | 5.796 | yes | local wall 4542s |
| 14 | 2026-11-12 05:24 UTC | multi {'deadline_s': 1500, 'steps': [{'task': 'pytest', 'args': 'tests/test_runn | L4 | NVIDIA L4 | 422 | 0.108 | 5.904 | yes | local wall 599s |

## Reconciliation with Modal billing

Last checked 2026-11-12 02:45 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$5.1191** (CPU $0.1254, H100 $3.2346, L4 $1.7018, Memory $0.0567, Network Egress $0.0007)
- Ledger estimate: **$5.7965**
- L4 seconds: Modal billed **7658 s**, ledger has 7810 s
- H100 seconds: Modal billed **2948 s**, ledger has 3012 s
