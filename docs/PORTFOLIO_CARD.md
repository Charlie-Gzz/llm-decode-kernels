# Portfolio card

**Pitch.** Custom Triton and CUDA kernels for the memory-bound parts of LLM decoding, measured
against the GPU's limits and swapped into a real model.

**Headline.** 4-bit decode matmul reaches 81% of the L4's peak memory bandwidth, for 1.82x faster
decoding than fp16 on Qwen2.5-1.5B (L4, batch 1).

**Three bullets.**
- Wrote a 4-bit weight matmul in Triton (decode GEMV path, split-K, tensor-core GEMM for prefill)
  and a CUDA C++ version; both hit 70-80% of peak bandwidth at batch 1 and beat fp16 cuBLAS by
  2-3.5x at decode sizes.
- Found and fixed two benchmarking traps (outputs hiding in L2 gave 171% of peak; torchao's
  tensor subclass under-sized the working set) and traced a sustained-vs-profiler gap to the L4's
  72 W power cap, which led to a 4-op dequant that lifted bandwidth from 66% to 81%.
- Swapped the kernels into Qwen2.5-1.5B with CUDA graphs: decode 58 -> 105 tokens/s on L4, with
  the perplexity cost shown to come from round-to-nearest quantization, not the kernels. Whole
  project ran on Modal with a spend ledger: about $5 of GPU time.

**Tags.** Triton, CUDA, GPU kernels, LLM inference, int4 quantization, roofline, PyTorch, Modal

**CV bullet.** Built Triton and CUDA C++ kernels for LLM decoding (fused RMSNorm, fused SwiGLU,
4-bit weight matmul) reaching 81% of peak memory bandwidth on an NVIDIA L4 and speeding up
Qwen2.5-1.5B decoding 1.8x over fp16, with a cost-capped cloud benchmarking workflow.
