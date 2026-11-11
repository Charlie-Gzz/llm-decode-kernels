# Decisions

| Decision | Options considered | Why | Easy to change? |
|---|---|---|---|
| Pinned stack: torch 2.15.0 (CUDA 13.2 wheels), triton 3.9.0, torchao 0.18.0, transformers 5.22.2 | older pinned stack; latest | Latest stable on PyPI on 2026-11-11. torch 2.15 requires triton ~=3.9. Kept in `kernels/config.py`. | Yes, one list |
| Modal base image `nvidia/cuda:13.2.1-devel-ubuntu24.04` | `debian_slim` + pip CUDA libs | `nvcc` is needed for the CUDA C++ kernel. 13.2 matches torch's CUDA 13.2 runtime. The same image is mirrored locally in `docker/Dockerfile.cuda` for free compile checks. | Yes |
| H100 requested as `gpu="H100!"` | `"H100"` | Modal can silently move `"H100"` to an H200. The `!` pins it, which matters for benchmarks (Modal GPU guide). L4 has no documented upgrade path. Every run also aborts if `nvidia-smi` reports a different GPU. | Yes |
| Billed seconds = container wall clock (from Python start) + 20 s startup allowance; local wall time only if a run returns nothing | local wall time around `.remote()` (first plan) | Changed after the smoke test: the first L4 call queued for ~15 min with no container, and local wall time logged 929 s for a 10 s run. Modal bills container time, not queue time. Reconciled against `modal billing report` after each milestone. | Yes |
| Ledger source of truth is `docs/gpu_ledger.jsonl`, rendered to `GPU_LEDGER.md` | hand-edited markdown | The budget guard reads it before every run, so it must be machine-readable. | Yes |
| "One H100 session, max 1.5 h" implemented as a cumulative cap of 5400 H100-seconds across M6 runs, each run under the 30 min function timeout | single 1.5 h function | The plan also caps any function at 30 min, so one session is a few back-to-back runs. | Yes |
| Model: Qwen2.5-1.5B-Instruct (Apache-2.0). Data: WikiText-2 raw test split (CC BY-SA 3.0) | - | Named in the plan. Licenses checked on the Hugging Face API. | - |
| Library baseline: torchao `Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")`, which runs tinygemm (`aten._weight_int4pack_mm`) | torchao "plain" format (needs the separate `mslk` package, Hopper-focused); bitsandbytes; Marlin | tinygemm is torchao's int4 path that runs on both L4 (sm_89) and H100. It only supports **bf16** activations, so this baseline runs in bf16. It moves the same number of bytes as our fp16 path, so the speed comparison is fair; numerics are not compared. | Yes |
| End-to-end uses our own small Qwen2 runner (HF weights, own static KV cache, RoPE, SDPA), not HF `generate` | HF `generate` + StaticCache + `torch.compile` | At batch 1 both sides need CUDA graphs. HF's mask/cache code is hard to capture reliably with custom kernels. The runner is ~150 lines, verified against HF on CPU (logits match to 1e-4) and again on GPU (parity check in every e2e result). | Yes |
| q/k/v and gate/up projections are fused (concatenated weights) on **both** fp16 and 4-bit sides | fuse only on our side | Fusing only our side would credit the kernels with a gain that comes from fusion. Same graph structure on both sides isolates the kernels. | Yes |
| Residual add fused into the next RMSNorm across layers (stretch item 1) done inside the runner from the start | separate add kernel | It falls out of the runner structure for free: each layer's input norm takes the previous MLP output plus the residual. The fp16 side runs the identical math as separate eager ops. | Yes |
| Chart colors: the plan's series palette, plus a distinct marker and line style per series and direct labels | change the hexes | The dataviz validator fails the plan palette for color-vision deficiency (sky vs violet dE 5.2 deutan, gray vs violet 14.3). The plan fixes the colors, so identity never relies on color alone. | Yes |
| GPU runs longer than 10 min end up "background-tracked" by the Bash tool | detached runs | The tool moves any command over 10 min to background tracking. The `modal run` process is still a non-detached child of this session (never `--detach`), so if the session dies the app stops. | - |
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
