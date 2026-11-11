# Slides outline (7 slides)

Audience: engineers and hiring managers who know what an LLM is but not GPU internals.

1. **Decoding an LLM is a memory problem.** At batch 1, every new token reads every weight once.
   On an L4 (300 GB/s), fp16 Qwen2.5-1.5B moves ~3.1 GB per token, so ~60 tokens/s is the ceiling
   no matter how fast the math is. Visual: roofline with fp16 at batch 1 pinned to the memory roof.
2. **Fewer bytes, fused passes.** Three kernels: 4-bit weight matmul (3.8x smaller weights),
   fused RMSNorm + residual add, fused SwiGLU. Visual: the explainer GIF.
3. **How the 4-bit matmul works.** Packed int32 tile, unpack in registers, scale and zero per
   group, tensor cores for batch > 1, a plain FMA path and split-K for batch 1. Visual: kernel
   excerpt next to the animation stills.
4. **Measured against the hardware.** CUDA-graph timing with rotating buffers (and the bug where
   outputs stayed in L2 and showed 171% of peak). Visual: bandwidth bars with the device-copy line.
5. **Triton vs CUDA C++.** CUDA wins batch 1 by ~10%, Triton wins batch 16 by 2x and covers
   prefill too. Visual: the two-panel % of peak chart.
6. **In the real model.** Qwen2.5-1.5B on L4 and H100, decode tokens/s fp16 vs ours, perplexity
   cost of round-to-nearest 4-bit, and why the end-to-end gain is smaller than the kernel gain
   (lm_head and attention stay fp16). Visual: decode tokens/s bars.
7. **What it cost and what's next.** GPU-hours and dollars from the ledger; next steps: AWQ/GPTQ
   for quality, a tensor-core CUDA path, quantized lm_head. Visual: cost card.
