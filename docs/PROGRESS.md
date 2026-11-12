# Progress

## Status
- **M0-M8 done.** Final results on L4 and H100 are in `results/raw/`, assets regenerated,
  README, site and handoff written.
- **GPU spend:** est. $5.46 (2.64 GPU-hours; H100 50 of 90 min). Dev $1.09 of $22, final runs
  $4.37 of the $8 reserve. See `docs/GPU_LEDGER.md`.

## Done
- **M0** setup: private repo, Modal app + ledger + budget guard, CPU test image, nvcc image,
  model and WikiText-2 in the Volume, L4 smoke test.
- **M1** harness + frozen method; baseline-only sweep `m1-baselines`.
- **M2** RMSNorm + SwiGLU: correct, 79-89% of peak at large shapes on L4/H100, up to 8x faster
  than eager, on par with torch.compile.
- **M3** W4A16 Triton: direct, split-K and batch-1 GEMV paths (factored dequant), all autotuned.
  L4 batch 1: 82-84% of peak, 2.8x faster than fp16 cuBLAS (median). Prefill: 0.96x of cuBLAS.
- **M4** CUDA C++ decode kernel: correct, benchmarked; note in `docs/TRITON_VS_CUDA.md`.
- **M5** Qwen2.5-1.5B end to end: decode 1.99x (L4), 1.41x (H100); ppl 10.654 -> 11.900, same as
  the dequant reference.
- **M6** final runs: `final` + `final2` on L4, `final` on H100 (one preemption, resumed).
- **M7** charts (`python -m kernels.assets`), site in `site/`, explainer GIF, screenshots,
  DEMO_SCRIPT, SLIDES_OUTLINE, PORTFOLIO_CARD, Pages workflow (manual, not enabled).
- **M8** README, HANDOFF.
- Stretch 1 (residual add fused into the next norm) is part of the runner from the start.

## Next step
- Stretch 2: a second model size (Qwen2.5-3B) end to end on L4, if it fits the budget (~$0.30).
- Then: final pass (lint, CPU tests, fresh-clone check), ledger reconciliation, cleanup.

## Restart checklist
1. `python -m modal app list` and stop anything running (`modal app stop -y <id>`).
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
