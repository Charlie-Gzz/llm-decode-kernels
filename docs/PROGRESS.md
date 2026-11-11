# Progress

## Status
- **Current milestone:** M0 setup (L4 smoke test pending)
- **GPU spend:** see `docs/GPU_LEDGER.md`

## Done
- M0.1 environment: Modal profile connected, no stray apps, Docker running, no ANTHROPIC_API_KEY,
  gh logged in. Modal workspace had $0 usage this month at start.
- M0.2 private repo github.com/Charlie-Gzz/llm-decode-kernels, plan renamed to PLAN.md, LF endings.
- M0.3 CPU test image (`docker compose run --rm cpu`) works. Modal image builds.
- M0.4 model (3.1 GB) and WikiText-2 test split downloaded into the Modal Volume on a CPU container.
- Kernels written and passing CPU interpreter tests: RMSNorm(+residual), SwiGLU, W4A16 Triton
  (direct + split-K). CUDA C++ decode kernel written, not yet compiled.
- Harness, sweep runner and e2e runner written, not yet run on GPU.

## Notes
- First L4 smoke call (12:29) sat queued with 0 containers for 15+ minutes (nothing billed).

## Next step
- Get the L4 smoke test through; then `python -m kernels.gpu pytest` for Triton kernels on L4.

## Restart checklist
1. `python -m modal app list` and stop anything running.
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
