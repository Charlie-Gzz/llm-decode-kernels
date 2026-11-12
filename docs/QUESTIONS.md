# Open questions

Each question has my recommendation and what I did in the meantime. None of these block the work.

| # | Question | Context | Recommendation | Meanwhile |
|---|---|---|---|---|
| 1 | Make the repo public and turn on GitHub Pages for the site? | The repo is private. `site/` is ready and `.github/workflows/pages.yml` is prepared but only runs by hand, and Pages is not enabled. | Yes, once you have read the README and site. Then set the description and topics (Section 8 of PLAN.md) and run the Pages workflow once. | Left private; nothing published. |
| 2 | Next optimization to try? | On the L4 the untouched fp16 `lm_head` (0.47 GB per token) plus attention is now 61% of a decode token. On the H100 the 4-bit matmul is limited by unpacking, not memory. | Quantize `lm_head` with the same W4A16 kernel (a small change, ~$0.50 of L4 time to measure; perplexity must be rechecked). A Hopper-specific path is a much bigger project. | Not started; listed in README limitations. |
| 3 | Better quantization (AWQ/GPTQ)? | RTN 4-bit costs +1.25 WikiText-2 perplexity on this 1.5B model. The plan put AWQ/GPTQ out of scope. | Keep it out of scope for this repo; the kernels accept any int4 weights with group scales/zeros, so a GPTQ checkpoint could be packed into the same format later. | Nothing. |
| 4 | Allow a fallback GPU when L4s are scarce? | On 2026-11-11 L4 requests often queued 15 to 70 minutes. Queue time is free. | Keep L4 only (as the plan says). All results are on one GPU type per chart. | Batched several steps into each run. |
| 5 | Does Modal's $30/month Starter credit change the budget? | Modal's Starter plan includes $30 of compute per month; this project used about $5.50. | No change. Check the Modal Usage page against `docs/GPU_LEDGER.md`. | Ledger reconciled against `modal billing report`. |
