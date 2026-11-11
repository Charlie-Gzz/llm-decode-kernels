# Demo script (2 minutes)

Audience: a recruiter or engineer watching a short screen recording.

**0:00-0:15 · The problem.** Open the site overview. "When an LLM writes one token at a time, the
GPU spends most of its time reading weights from memory, not doing math. I wrote GPU kernels to
read fewer bytes and make fewer trips."

**0:15-0:40 · The headline kernel.** Play the explainer animation. "Weights are stored as 4-bit
integers, eight per 32-bit word. The kernel loads a packed tile, unpacks it in registers, applies
the scale and zero for its group, and feeds the result to the multiply. The full-size weights
never touch memory."

**0:40-1:05 · Measured honestly.** Open the W4A16 page, point at the speedup grid and the
roofline. "Against fp16 cuBLAS, the 4-bit kernel is several times faster at decode sizes. At
prefill sizes it is compute-bound and cuBLAS wins, and the chart says so. Every number is a
CUDA-graph median of three sweeps, with inputs rotated so nothing hides in cache."

**1:05-1:25 · Triton vs CUDA.** Open the Triton vs CUDA chart. "I also wrote the decode case in
CUDA C++. It is about 10% faster at batch 1 and twice as slow at batch 16, because it runs on CUDA
cores. Triton gets most of the speed for a fraction of the effort."

**1:25-1:50 · The real model.** Open the end-to-end page. "Swapped into Qwen2.5-1.5B, decoding
goes from fp16 to the 4-bit kernels at the speeds shown here. Perplexity rises a bit, and the
dequantize-then-matmul reference gives the same number, so that cost belongs to round-to-nearest
quantization, not to the kernels."

**1:50-2:00 · Cost.** Open the cost page. "Everything ran on Modal with a spend ledger and hard
caps. The whole project cost a few dollars of GPU time."
