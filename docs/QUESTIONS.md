# Open questions

All five were answered on 2026-11-12: the user went with the recommendations.

| # | Question | Recommendation | Decision |
|---|---|---|---|
| 1 | Make the repo public and turn on GitHub Pages for the site? | Yes, after review; set description and topics, run the Pages workflow once. | **Done.** Description and topics set; repo made public; Pages enabled from the Actions workflow. The user confirmed that the commit author email being public is fine. |
| 2 | Next optimization to try? | Quantize `lm_head` with the same W4A16 kernel (~$0.50 of L4 time). | **Done** (follow-up run, L4, $0.11): 2.36x decode vs fp16 (140.5 tok/s), perplexity 11.90 to 12.59. Reported next to the headline config, not in place of it. |
| 3 | Better quantization (AWQ/GPTQ)? | Keep out of scope; the kernel format can take GPTQ weights later. | Kept out of scope. |
| 4 | Allow a fallback GPU when L4s are scarce? | Keep L4 only. | Kept L4 only. |
| 5 | Does Modal's $30/month Starter credit change the budget? | No change. | No change. The Modal Volume (~5 GB, about $0.45/month) is kept for quick reruns. |
