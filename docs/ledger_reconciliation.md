## Reconciliation with Modal billing

Last checked 2026-11-11 18:00 UTC with `modal billing report --for 'this month'`. Modal's data can lag by about an hour, so the newest runs may be missing. The actual figure also includes CPU-only containers (downloads), which the GPU ledger does not track.

- Modal actual, this app, all resources (GPU, CPU, memory, egress): **$0.0097** (CPU $0.0044, L4 $0.0036, Memory $0.0011, Network Egress $0.0006)
- Ledger estimate: **$0.0077**
