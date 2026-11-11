# llm-decode-kernels

Triton and CUDA kernels for the slow parts of LLM decoding: fused RMSNorm, fused SwiGLU, and a
4-bit weight matmul. Each one is tested against PyTorch, measured against the GPU's limits, and
swapped into Qwen2.5-1.5B.

<!-- gen:headline -->
_Headline pending: results not in yet._
<!-- /gen:headline -->

![How the 4-bit matmul works](assets/explainer.gif)

## Results

![Decode speed](assets/charts/decode_tokens.png)

![W4A16 speedup on L4](assets/charts/speedup_w4a16_l4.png)

![Roofline on L4](assets/charts/roofline_l4.png)

### End to end, Qwen2.5-1.5B-Instruct

<!-- gen:e2e -->
| GPU | Backend | Decode tok/s (batch 1) | Swapped layers, ms/token | Prefill tok/s (T=2048) | WikiText-2 ppl | Linear weights |
|---|---|---|---|---|---|---|
<!-- /gen:e2e -->

### 4-bit matmul (W4A16)

<!-- gen:w4a16 -->
| Layer (N x K) | M | GPU | Triton (us) | CUDA (us) | fp16 cuBLAS (us) | torchao (us) | torch.compile (us) | Triton vs fp16 | Triton % peak BW |
|---|---|---|---|---|---|---|---|---|---|
| qwen_gate_up (17920x1536) | 1 | L4 | 68.4 | - | 208.6 | 79.0 | 82.6 | 3.05x | 71% |
| qwen_gate_up (17920x1536) | 16 | L4 | 81.3 | - | 218.1 | 149.3 | 454.1 | 2.68x | 63% |
| qwen_down (1536x8960) | 1 | L4 | 39.6 | - | 102.9 | 34.1 | 45.4 | 2.60x | 62% |
| qwen_down (1536x8960) | 16 | L4 | 42.6 | - | 110.6 | 83.0 | 167.5 | 2.60x | 60% |
| 8b_up (14336x4096) | 1 | L4 | 156.8 | - | 431.4 | 123.9 | 143.8 | 2.75x | 66% |
| 8b_up (14336x4096) | 16 | L4 | 167.5 | - | 463.6 | 295.0 | 1155.6 | 2.77x | 63% |
<!-- /gen:w4a16 -->

### Fused RMSNorm and SwiGLU

<!-- gen:kernels -->
| Kernel | Shape | GPU | Ours (us) | PyTorch eager (us) | torch.compile (us) | Speedup vs eager | % peak BW |
|---|---|---|---|---|---|---|---|
| RMSNorm + residual | N=4096, M=16384 | L4 | 2268.6 | 12036.1 | 2207.1 | 5.31x | 79% |
| RMSNorm + residual | N=1536, M=1 | L4 | 1.3 | 10.3 | 1.3 | 7.80x | 4% |
| SwiGLU | I=8960, M=4096 | L4 | 932.0 | 1559.8 | 878.8 | 1.67x | 79% |
| SwiGLU | I=8960, M=1 | L4 | 1.1 | 2.7 | 1.0 | 2.44x | 16% |
<!-- /gen:kernels -->

Times are CUDA-graph medians of 3 sweeps, with enough rotating input copies that every call reads
from DRAM (working set at least 4x the L2 cache).
<!-- gen:spread -->
Median spread between the 3 sweeps: 0.6%. 0 of 255 measurements spread more than 10% (marked in the site tables).
<!-- /gen:spread -->

### Cost

<!-- gen:cost -->
Total GPU time: **0.42 GPU-hours**, estimated **$0.39** (cap $30), of which H100: 0 min. Every run is in [docs/GPU_LEDGER.md](docs/GPU_LEDGER.md).
<!-- /gen:cost -->

## Quickstart

```bash
pip install -e ".[local]" && python -m modal setup   # once
docker compose run --rm cpu                          # all kernel tests on CPU, free
python -m kernels.gpu pytest                         # same tests on an L4 (about $0.05)
```

Rebuild every chart, the site and this README's tables from the saved results (no GPU):
`python -m kernels.assets`.

## How it works

```mermaid
flowchart LR
    A[Kernel source<br/>Triton / CUDA C++] -->|CPU, free| B[Docker<br/>TRITON_INTERPRET=1, nvcc]
    B -->|passes| C[modal run on L4]
    C --> D[GPU tests vs PyTorch]
    D --> E[Benchmark sweeps<br/>CUDA graphs, rotating buffers]
    E --> F[(results/raw/*.json)]
    C -->|billed seconds| L[docs/GPU_LEDGER.md]
    F --> G[Qwen2.5 end to end]
    G --> F
    F --> H[charts, site, README]
```

- **Fused RMSNorm + residual** ([kernels/rmsnorm.py](kernels/rmsnorm.py)): one program per row,
  adds the residual, writes the new residual stream and the normalized output in one pass.
- **Fused SwiGLU** ([kernels/swiglu.py](kernels/swiglu.py)): reads both halves of the merged
  gate/up output in place and writes `silu(gate) * up`.
- **W4A16 matmul** ([kernels/w4a16.py](kernels/w4a16.py)): int4 weights packed 8 per int32,
  fp16 scale and zero per group of 128, unpacked and dequantized in registers, then `tl.dot` on
  tensor cores. Autotuned. Decode shapes (batch 1 to 16) can split K across programs so a small
  matmul still fills the GPU.
- **CUDA C++ decode kernel** ([kernels/cuda/w4a16_decode.cu](kernels/cuda/w4a16_decode.cu)): the
  same math as a hand-written GEMV with 16-byte loads, shared-memory reduction and split-K.
- **End to end** ([kernels/e2e/model.py](kernels/e2e/model.py)): a small Qwen2 runner on the
  Hugging Face weights with a static KV cache, captured in a CUDA graph. fp16 and 4-bit use the
  same runner, so the difference is the kernels.

Benchmark method and every judgment call: [docs/DECISIONS.md](docs/DECISIONS.md).

## Limitations

- Quantization is plain round-to-nearest. AWQ or GPTQ would lose less quality; that was out of
  scope. The perplexity change belongs to the quantization method, not the kernels.
- Embeddings, `lm_head`, attention and RoPE stay fp16. On a 1.5B model the fp16 `lm_head` is a
  large share of the bytes read per token, which caps the end-to-end speedup.
- Batch 1 decode only for the end-to-end numbers. The CUDA kernel covers decode (M up to 16) only.
- Cloud GPUs vary from run to run. We report the median of 3 sweeps in one container per GPU,
  not across many machines.
- The torchao baseline runs in bf16 because its int4 kernel (tinygemm) only supports bf16.
- Two GPU types (L4, H100). No AMD, Apple, or multi-GPU.
