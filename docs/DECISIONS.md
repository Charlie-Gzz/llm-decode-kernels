# Decisions

| Decision | Options considered | Why | Easy to change? |
|---|---|---|---|
| Pinned stack: torch 2.15.0 (CUDA 13.2 wheels), triton 3.9.0, torchao 0.18.0, transformers 5.22.2 | older pinned stack; latest | Latest stable on PyPI on 2026-11-11. torch 2.15 requires triton ~=3.9. Kept in `kernels/config.py`. | Yes, one list |
| Modal base image `nvidia/cuda:13.2.1-devel-ubuntu24.04` | `debian_slim` + pip CUDA libs | `nvcc` is needed for the CUDA C++ kernel. 13.2 matches torch's CUDA 13.2 runtime. The same image is mirrored locally in `docker/Dockerfile.cuda` for free compile checks. | Yes |
| H100 requested as `gpu="H100!"` | `"H100"` | Modal can silently move `"H100"` to an H200. The `!` pins it, which matters for benchmarks (Modal GPU guide). L4 has no documented upgrade path. Every run also aborts if `nvidia-smi` reports a different GPU. | Yes |
| Billed seconds measured locally around the `.remote()` call | in-container time only | Local wall time includes scheduling and cold start, so it over-estimates slightly. Safer for a budget. Reconciled against `modal billing report` after each milestone. | Yes |
| Ledger source of truth is `docs/gpu_ledger.jsonl`, rendered to `GPU_LEDGER.md` | hand-edited markdown | The budget guard reads it before every run, so it must be machine-readable. | Yes |
| "One H100 session, max 1.5 h" implemented as a cumulative cap of 5400 H100-seconds across M6 runs, each run under the 30 min function timeout | single 1.5 h function | The plan also caps any function at 30 min, so one session is a few back-to-back runs. | Yes |
| Model: Qwen2.5-1.5B-Instruct (Apache-2.0). Data: WikiText-2 raw test split (CC BY-SA 3.0) | - | Named in the plan. Licenses checked on the Hugging Face API. | - |
| Benchmark method, frozen before any result (see section below) | `triton.testing.do_bench` | See below. | No: changing it invalidates earlier results |

## Benchmark method (frozen 2026-11-11, before any GPU result)

- **Timing:** CUDA graphs with rotating buffers. For each implementation and shape we allocate
  enough copies of the inputs (weights included) that the working set is at least 4x the GPU's L2
  cache (minimum 2 copies, capped by a memory budget). One CUDA graph runs one call per copy, so
  every call reads its inputs from DRAM, like a real model where each layer has its own weights.
  Time per call = graph time / number of calls.
- **Why not plain `do_bench`:** it flushes L2 (good) but times each call with events around a
  Python launch, so at decode sizes it measures CPU launch overhead, not the kernel. The end-to-end
  model also runs under CUDA graphs, so graph timing is the matching number.
  `do_bench_cudagraph` was rejected because it does not flush L2, and L4's 48 MB L2 holds a whole
  4-bit weight matrix, which would inflate bandwidth numbers.
- **Secondary "eager" number:** CUDA events around a loop of calls without a graph, which includes
  Python and launch overhead. Reported alongside, never as the headline.
- **Warmup:** 3 untimed calls per implementation before capture (autotuning and `torch.compile`
  happen here), then 1 untimed graph replay.
- **Repeats:** each graph is replayed until at least 50 ms has passed; 7 measurements per shape,
  median kept. The whole sweep runs **3 times** in the same container; we report the median of the
  3 sweeps with min and max, and flag spreads over 10%.
- **Metrics:** achieved GB/s = minimum bytes the op must move (each input read once, each output
  written once) / time. TFLOPS = 2*M*N*K / time for matmuls. % of peak uses the NVIDIA spec sheet
  numbers in `kernels/config.py`. We also measure a plain device-to-device copy as the practical
  bandwidth ceiling.
- **`torch.compile` baseline:** default mode, `fullgraph=True`, `dynamic=False`, compiled per shape.
- **Correctness:** `torch.testing.assert_close` against an fp32 PyTorch reference with per-kernel
  tolerances logged in the test files, random data and odd shapes.
