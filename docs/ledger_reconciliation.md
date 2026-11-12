## Reconciliation with Modal billing

Last checked 2026-11-12 01:15 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$4.8163** (CPU $0.1126, H100 $3.2346, L4 $1.4160, Memory $0.0524, Network Egress $0.0007)
- Ledger estimate: **$5.4595**
- L4 seconds: Modal billed **6372 s**, ledger has 6492 s
- H100 seconds: Modal billed **2948 s**, ledger has 3012 s
