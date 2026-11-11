# Open questions

Each question has my recommendation and what I did in the meantime. None of these block the work.

| # | Question | Context | Recommendation | Meanwhile |
|---|---|---|---|---|
| 1 | Allow a fallback GPU when L4s are scarce? | On 2026-11-11 every L4 request queued 15 to 50+ minutes before a container started. Queue time is free (confirmed in Modal's billing report), it only costs wall-clock time. The plan allows L4 only. | Keep L4 only. The cost stays predictable and the results stay on one GPU. | Batched several steps into each run (`multi` task) so each queue wait buys more work. |
| 2 | Does Modal's $30/month Starter credit change the budget? | Modal's pricing page says the Starter plan includes $30 of compute per month. All spend in this project so far fits inside it. | No change: keep the plan's own caps ($30 total, $22 dev, $8 reserve). Check the Modal Usage page against `docs/GPU_LEDGER.md` at the end. | Ledger reconciles against `modal billing report` after each milestone. |
