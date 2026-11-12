# Progress

## Status
- **M0-M8 done.** Final results on L4 and H100 are in `results/raw/`, assets regenerated,
  README, site and handoff written.
- **GPU spend:** est. $5.80 (3.0 GPU-hours; H100 50 of 90 min). Dev $1.42 of $22, final runs
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
- Stretch 2: Qwen2.5-0.5B end to end on L4 (1.39x decode).
- Stretch 3: FP8 weight-only matmul, e2e and sweep on L4 (1.54x decode, ppl +0.07).

## Next step
- Waiting on the user's answers in `docs/QUESTIONS.md` (publish, next optimization).
- GitHub description and 15 topics set (2026-11-11, at the user's request), from PLAN.md
  section 8 plus the L4 headline number. Repo is still private.

## Restart checklist
1. `python -m modal app list` and stop anything running (`modal app stop -y <id>`).
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
