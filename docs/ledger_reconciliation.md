## Reconciliation with Modal billing

Last checked 2026-11-11 23:20 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$1.2798** (CPU $0.0676, L4 $1.1813, Memory $0.0302, Network Egress $0.0006)
- Ledger estimate: **$1.3855**
