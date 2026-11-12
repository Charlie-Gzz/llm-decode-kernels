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
CUDA C++. With the same math, Triton ended up faster everywhere: 77% vs 70% of peak at batch 1,
and twice as fast at batch 16, where my CUDA kernel runs on CUDA cores instead of tensor cores."

**1:25-1:50 · The real model.** Open the end-to-end page. "Swapped into Qwen2.5-1.5B, decoding
goes from 60 to 118 tokens per second on an L4, and from 283 to 398 on an H100. Perplexity rises
from 10.65 to 11.90, and the dequantize-then-matmul reference gives the same number, so that cost
belongs to round-to-nearest quantization, not to the kernels."

**1:50-2:00 · Cost.** Open the cost page. "Everything ran on Modal with a spend ledger and hard
caps. The whole project cost about five and a half dollars of GPU time."
