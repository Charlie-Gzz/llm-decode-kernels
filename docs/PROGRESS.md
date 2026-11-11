# Progress

## Status
- **Current milestone:** M3-M5 dev results in; one optimization experiment (run 7) before M6 finals
- **GPU spend:** ~$1.03 dev of $22 (see `docs/GPU_LEDGER.md`)

## Done
- **M0** setup: private repo, Modal app + ledger + budget guard, CPU test image, nvcc image,
  model and WikiText-2 in the Volume, L4 smoke test.
- **M1** harness + frozen method (DECISIONS). Baseline-only sweep `m1-baselines` on L4.
- **M2** RMSNorm and SwiGLU: correct on L4, `m2-v2` sweep (3 sweeps). 78-85% of peak at large
  shapes, up to 8x / 2.4x faster than eager, on par with torch.compile. First `m2` run was wrong
  (output buffers stayed in L2, up to 171% of peak); fixed and logged in DECISIONS.
- **M3** W4A16 Triton: correct across decode and prefill shapes (68+ GPU tests). `m3-decode-v2`
  and `m3-prefill` sweeps with all baselines (fp16 cuBLAS, torch.compile, naive, torchao).
  GEMV path for M=1 added after the first sweep.
- **M4** CUDA C++ decode kernel: correct on L4 (20 tests). Batch 1: CUDA 69% vs Triton 63% of
  peak (medians); batch 16: Triton 61%, CUDA 28% (CUDA cores, compute-bound). LOC 72 vs 58.
- **M5** e2e Qwen2.5-1.5B on L4: decode 57.9 -> 105.5 tok/s (Triton, 1.82x), 109.0 (CUDA, 1.88x);
  swapped layers 11.5 -> 3.9 ms/token; ppl 10.654 -> 11.900 (same as dequant reference 11.901);
  runner matches HF ppl exactly.
- Profiling: Nsight Compute works on Modal with `--clock-control none`.
- Assets: charts, site, README generator, explainer GIF, CI, Pages workflow (manual).

## In flight
- Run 7: factored-dequant GEMV variant (autotuned against the original) + power probe to test
  whether the 72 W L4 throttles our ALU-heavier kernels.

## Next step
1. Read run 7. If the factored GEMV wins, keep it (autotuner picks it anyway).
2. M6 final L4: `final-l4` sweeps (copy, rmsnorm, swiglu, w4a16 decode+prefill, 3 sweeps) +
   final e2e + profile, `--final`.
3. M6 H100 session (`--gpu H100 --final`), same steps, cap 1.5 h.
4. M7/M8: TRITON_VS_CUDA note, DEMO_SCRIPT, SLIDES_OUTLINE, PORTFOLIO_CARD, README, HANDOFF.

## Restart checklist
1. `python -m modal app list` and stop anything running (`modal app stop -y <id>`).
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
