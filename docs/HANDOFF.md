# Handoff

## What was built
- Three Triton kernels for LLM decoding: fused RMSNorm + residual add, fused SwiGLU, and a W4A16
  matmul (int4 weights, fp16 activations, group size 128) with a tensor-core GEMM path, a split-K
  path and a batch-1 GEMV path with factored dequantization. A CUDA C++ version of the decode
  matmul for comparison.
- A benchmark harness (CUDA graphs, rotating input and output buffers, 3 sweeps), baselines
  (PyTorch eager, `torch.compile`, cuBLAS fp16, dequantize-then-matmul, torchao int4), profiling
  (Nsight Compute on Modal), and a power probe.
- An end-to-end runner for Qwen2.5 on the Hugging Face weights, captured in CUDA graphs, checked
  against Hugging Face's model.
- A cost-capped Modal workflow: budget guard before every GPU run, ledger after, stray-app cleanup,
  resumable sweeps.
- Charts, a static results site, an explainer animation and GIF, README, and talk material.

## How to run
```bash
pip install -e ".[local]" && python -m modal setup    # once
docker compose run --rm cpu                           # kernel tests on CPU (free)
python -m kernels.gpu pytest --long                   # GPU tests on an L4 (~$0.15)
python -m kernels.assets                              # rebuild charts, site, README tables (no GPU)
```
Full benchmark reproduction: `runs/README.md`.

## Headline results
| | L4 | H100 |
|---|---|---|
| Qwen2.5-1.5B decode, fp16 -> ours (batch 1) | 59.5 -> 118.2 tok/s (**1.99x**) | 283 -> 398 tok/s (**1.41x**) |
| W4A16 batch 1, % of peak bandwidth (Qwen gate_up) | **82%** (84% on the 8B-sized layer) | 34% (45%) |
| W4A16 vs fp16 cuBLAS, decode sizes (median) | 2.8x | 1.1x |
| W4A16 vs fp16 cuBLAS, prefill sizes (median) | 0.96x | 0.41x |
| RMSNorm / SwiGLU, % of peak at large sizes | 79% / 80% | 89% / 86% |
| Triton vs CUDA, batch 1 / batch 16 (% of peak, median) | 77/70, 62/29 | 25/17, 19/5 |

WikiText-2 perplexity 10.654 (fp16) -> 11.900 (4-bit RTN); the dequantize-then-cuBLAS reference
gives 11.901, so the kernels add no error.

**GPU time and spend:** 3.0 GPU-hours, estimated $5.80 of the $30 cap (dev $1.42 of $22; final
runs $4.37 of the $8 reserve; H100 50 of 90 minutes). No cap was hit. Modal's own billing
report is lower than the ledger (the ledger rounds up). Ledger:
`docs/GPU_LEDGER.md`, reconciled against `modal billing report`.

## Where things are
- Results: `results/raw/*.json` (every GPU run, with GPU, driver, CUDA, Triton and torch versions).
- Charts: `assets/charts/*.png|svg`. Screenshots: `assets/screens/`. GIF: `assets/explainer.gif`.
- Site: `site/index.html` (open locally; Pages workflow prepared, not enabled).
- Docs: `docs/DECISIONS.md` (every judgment call, including method changes after seeing data),
  `docs/TRITON_VS_CUDA.md`, `docs/DEMO_SCRIPT.md`, `docs/SLIDES_OUTLINE.md`,
  `docs/PORTFOLIO_CARD.md`, `docs/QUESTIONS.md`.

## Top questions (details and the rest in `docs/QUESTIONS.md`)
1. **Make the repo public and enable Pages?** Recommendation: yes, after you read the README and
   site; then set the description and topics from PLAN.md section 8.
2. **Next optimization?** Recommendation: quantize `lm_head` with the same kernel (~$0.50 to
   measure); it is now the biggest single cost per token on the L4.
3. **AWQ/GPTQ?** Recommendation: keep out of scope; the kernel format can take such weights later.

## Skipped, changed or estimated, and why
- **Benchmark method changes after seeing data** (all logged with the earlier numbers in
  DECISIONS): output buffers now rotate (first M2 run showed 171% of peak); torchao working set
  sized by real bytes; the decode path picker now times in CUDA graphs (final L4 decode sweep
  for Triton and CUDA rerun as `final2`).
- **H100 preemption:** Modal preempted the first H100 container; the sweep resumed from its
  checkpoint. The preempted container's time (about 990 s) is in the ledger as an estimate.
- **Nsight Compute** runs with `--clock-control none` (Modal cannot lock clocks), so its numbers
  are at boost clocks and higher than sustained ones; both are reported.
- **Stretch items** (all L4 only, the single H100 session was already used):
  (1) residual fused into the next norm: done in the runner.
  (2) second model size: Qwen2.5-0.5B, 1.39x decode (vs 1.99x on 1.5B), ppl 15.79 to 19.09.
  7B did not fit the L4 with this runner and 3B has a non-commercial license.
  (3) FP8 weight-only matmul: 1.54x decode on 1.5B with ppl 10.654 to 10.719; 78-84% of peak
  from batch 1 to 16.
- **Fresh-clone check** covered the CPU path (install, lint, CPU tests, asset rebuild); the GPU
  quickstart step was not repeated because it costs money and the same commands ran many times.
- **L4 capacity:** many L4 requests queued 15 to 70 minutes (not billed). Work was batched to cope.
