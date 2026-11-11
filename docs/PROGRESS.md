# Progress

## Status
- **Current milestone:** M1/M2 (first batched GPU run in the L4 queue)
- **GPU spend:** see `docs/GPU_LEDGER.md`

## Done
- M0 setup: private repo, Modal app + ledger + budget guard, CPU test image, nvcc image, model
  and WikiText-2 in the Volume (CPU download), L4 smoke test passed (vector add 229 GB/s).
- Kernels pass CPU interpreter tests: RMSNorm(+residual), SwiGLU, W4A16 Triton (direct +
  split-K). CUDA C++ decode kernel compiles locally with nvcc (sm_89, sm_90).
- E2E runner verified on CPU against HF Qwen2 (tiny random model): logits match, KV-cache decode
  matches, our kernels match the 4-bit reference.
- Harness, sweep runner, profile task (ncu with torch.profiler fallback), charts, site generator,
  explainer animation + GIF, CI (CPU only), Pages workflow (manual, not enabled).

## Notes
- L4 capacity is scarce: queue waits of 15-50+ min with no container. Queue time is not billed.
  Two queued calls were cancelled before any container started (nothing billed).
- Bash tool moves commands over 10 min to background tracking; the `modal run` process is still
  a non-detached child of this session, so the GPU stops if the session dies.

## Next step
- Batched run `.scratch_run3.json`: pytest (all, incl. CUDA), bench m1-baselines, bench m2,
  profile, bench m3-decode (resumable). When it lands: commit results + ledger immediately,
  then fix any failures; then w4a16 prefill sweep and e2e.

## Restart checklist
1. `python -m modal app list` and stop anything running (`modal app stop -y <id>`).
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
