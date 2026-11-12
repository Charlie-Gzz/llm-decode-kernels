## Reconciliation with Modal billing

Last checked 2026-11-12 02:45 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$5.1191** (CPU $0.1254, H100 $3.2346, L4 $1.7018, Memory $0.0567, Network Egress $0.0007)
- Ledger estimate: **$5.7965**
- L4 seconds: Modal billed **7658 s**, ledger has 7810 s
- H100 seconds: Modal billed **2948 s**, ledger has 3012 s
