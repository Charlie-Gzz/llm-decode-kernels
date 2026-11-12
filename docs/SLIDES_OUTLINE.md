# Slides outline (7 slides)

Audience: engineers and hiring managers who know what an LLM is but not GPU internals.

1. **Decoding an LLM is a memory problem.** At batch 1, every new token reads every weight once.
   On an L4 (300 GB/s), fp16 Qwen2.5-1.5B reads ~3.1 GB per token: even at full spec bandwidth
   that caps it near 100 tokens/s, and we measure 60, no matter how fast the math is. Visual: roofline with fp16 at batch 1 pinned to the memory roof.
2. **Fewer bytes, fused passes.** Three kernels: 4-bit weight matmul (3.8x smaller weights),
   fused RMSNorm + residual add, fused SwiGLU. Visual: the explainer GIF.
3. **How the 4-bit matmul works.** Packed int32 tile, unpack in registers, scale and zero per
   group, tensor cores for batch > 1, a plain FMA path and split-K for batch 1. Visual: kernel
   excerpt next to the animation stills.
4. **Measured against the hardware.** CUDA-graph timing with rotating buffers (and the bug where
   outputs stayed in L2 and showed 171% of peak). The L4 throttles to its 72 W cap, so fewer
   instructions per weight meant more bandwidth: 66% -> 82%. The H100 tells the opposite story:
   34-45%, limited by unpacking. Visual: both rooflines side by side.
5. **Triton vs CUDA C++.** After both got the same 4-op dequant, Triton led at every shape:
   77% vs 70% of peak at batch 1, 62% vs 29% at batch 16 (CUDA runs on CUDA cores). Visual: the
   two-panel % of peak chart.
6. **In the real model.** Qwen2.5-1.5B decodes 1.99x faster on L4 (60 -> 118 tok/s) and 1.41x
   on H100 (283 -> 398). Perplexity 10.65 -> 11.90, all from round-to-nearest. The swapped layers
   got 3.5x faster but fp16 lm_head and attention now take 61% of a token. Visual: decode bars
   and the time breakdown.
7. **What it cost and what's next.** 2.6 GPU-hours, about $5.50 on Modal, every run in a ledger.
   Next: AWQ/GPTQ for quality, a Hopper (wgmma) path, a quantized lm_head. Visual: cost card.
