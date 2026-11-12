# Portfolio card

**Pitch.** Custom Triton and CUDA kernels for the memory-bound parts of LLM decoding, measured
against the GPU's limits and swapped into a real model.

**Headline.** 4-bit decode matmul reaches 82% of the L4's peak memory bandwidth, for 1.99x faster
decoding than fp16 on Qwen2.5-1.5B (batch 1; 1.41x on an H100).

**Three bullets.**
- Wrote a 4-bit weight matmul in Triton (batch-1 GEMV, split-K, tensor-core GEMM for prefill) and
  a CUDA C++ version. At batch 1 on an L4, Triton reaches 82-84% of peak bandwidth and runs 2.8x
  faster than fp16 cuBLAS (median over layer shapes).
- Found and fixed three measurement traps: outputs hiding in L2 (171% of peak), a library's
  tensor subclass under-sizing the working set, and eager timing choosing the wrong kernel path.
  Traced a profiler-vs-sustained gap to the L4's 72 W power cap and cut the dequant to 4
  instructions per weight, lifting bandwidth from 66% to 82%.
- Swapped the kernels into Qwen2.5-1.5B with CUDA graphs: decode 60 -> 118 tokens/s on L4 and
  283 -> 398 on H100, with the perplexity cost traced to round-to-nearest quantization, not the
  kernels. Reported the losses too: prefill and the H100's small layers favor cuBLAS. Whole
  project ran on Modal with a spend ledger: about $5.50 of GPU time.

**Tags.** Triton, CUDA, GPU kernels, LLM inference, int4 quantization, roofline, Nsight Compute,
PyTorch, Modal

**CV bullet.** Built Triton and CUDA C++ kernels for LLM decoding (fused RMSNorm, fused SwiGLU,
4-bit weight matmul) reaching 82% of peak memory bandwidth on an NVIDIA L4 and doubling
Qwen2.5-1.5B decode speed over fp16, using a cost-capped cloud benchmarking workflow
(about $5.50 of GPU time).
