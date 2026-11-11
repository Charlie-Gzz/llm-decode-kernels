# Triton or CUDA C++ for the 4-bit decode matmul?

Both kernels compute the same thing on the same weight format (int4 packed 8 per int32, fp16 scale and zero per group of 128) and pass the same tests. Numbers below are from the L4 (CUDA-graph timing, median of 3 sweeps); the H100 numbers are on the site.

**Speed.** At batch 1 the hand-written CUDA kernel is a little faster: a median of 69% of peak bandwidth across six layer shapes, against 63% for Triton. On the big layers the gap is larger (8b_up: 78% vs 66%). Each CUDA lane loads 16 bytes (32 weights) with one instruction and keeps 4 output columns in registers, and nothing goes through shared memory except the activations. At batch 16 the picture flips: Triton holds 61% while CUDA drops to 28%. The CUDA kernel does its multiply-adds on CUDA cores, so 16 rows mean 16 times the arithmetic per weight and it becomes compute-bound. Triton's `tl.dot` path uses tensor cores and barely notices the extra rows. In the full model at batch 1 the difference is small: 109.0 vs 105.5 tokens/s.

**Code.** The Triton GEMM kernel is 58 lines and the Triton batch-1 GEMV kernel 41; the CUDA kernel is 72 lines plus a 52-line host launcher with manual shape checks, alignment rules (N must be a multiple of 4 for 16-byte loads) and template dispatch on the batch size (`scripts/loc.py`). Triton's autotuner picked tile sizes, warps, pipeline depth and split-K for every shape; for CUDA we tuned only split-K and hand-picked everything else. One Triton kernel also covers prefill (batch 2048), which the CUDA kernel does not do at all.

**When each is worth it.**
- Triton first: one kernel for decode and prefill, autotuning for free, tensor cores without writing MMA code, and readable by anyone who knows NumPy-style indexing. It got within ~10% of the hand-written kernel at batch 1 and beat it 2x at batch 16.
- CUDA C++ when one shape matters most and the last 10% is worth it, for example a batch-1 GEMV that runs billions of times. The control over vector widths, register tiles and shared-memory reductions is what wins there. Covering batch 16 well would mean writing an MMA (tensor-core) path by hand, which is much more code.
