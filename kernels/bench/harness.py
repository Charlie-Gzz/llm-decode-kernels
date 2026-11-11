"""Benchmark harness. Method frozen in docs/DECISIONS.md before any results were taken.

- Rotating input copies so the working set is >= 4x L2: every call reads from DRAM.
- CUDA graph of one call per copy -> kernel time without Python launch overhead (headline).
- Eager loop with CUDA events -> includes launch overhead (secondary).
- 7 measurements, median kept. Callers repeat whole sweeps 3 times.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable

import torch

WARMUP_CALLS = 3
MEASUREMENTS = 7
MIN_MS = 50.0
MEMORY_BUDGET = 3 * 1024**3  # bytes of rotating copies per case


def tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(o) for o in obj.values())
    if isinstance(obj, torch.nn.Module):  # quantized subclasses report their logical size
        return sum(tensor_bytes(p) for p in [*obj.parameters(), *obj.buffers()])
    return 0


def n_copies(bytes_per_set: int, l2_mb: float) -> int:
    want = math.ceil(4 * l2_mb * 2**20 / max(bytes_per_set, 1))
    cap = max(2, MEMORY_BUDGET // max(bytes_per_set, 1))
    return int(min(max(want, 2), cap, 512))


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def time_graph(fn: Callable, sets: list) -> float:
    """Median ms per call with all calls captured in one CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for args in sets[:WARMUP_CALLS]:
            fn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in sets:
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()

    start, end = _events()
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    replays = max(1, math.ceil(MIN_MS / max(start.elapsed_time(end), 1e-3)))

    samples = []
    for _ in range(MEASUREMENTS):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (replays * len(sets)))
    del graph
    return statistics.median(samples)


def time_eager(fn: Callable, sets: list) -> float:
    """Median ms per call, launched from Python (includes launch overhead)."""
    for args in sets[:WARMUP_CALLS]:
        fn(*args)
    torch.cuda.synchronize()
    start, end = _events()
    start.record()
    for args in sets:
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    loops = max(1, math.ceil(MIN_MS / max(start.elapsed_time(end), 1e-3)))
    samples = []
    for _ in range(MEASUREMENTS):
        start.record()
        for _ in range(loops):
            for args in sets:
                fn(*args)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (loops * len(sets)))
    return statistics.median(samples)


def measure(fn: Callable, make_inputs: Callable[[], tuple], l2_mb: float,
            eager: bool = True, set_bytes: int | None = None) -> dict:
    """Build rotating copies with `make_inputs()`, time `fn(*inputs)` both ways.

    `set_bytes` overrides the size of one input set, for inputs whose logical size differs from
    what they occupy (torchao's int4 tensor subclass reports its bf16 size)."""
    first = make_inputs()
    fn(*first)  # autotuning / torch.compile happen here, outside any timing or capture
    torch.cuda.synchronize()
    copies = n_copies(set_bytes or tensor_bytes(first), l2_mb)
    sets = [first] + [make_inputs() for _ in range(copies - 1)]
    out = {"copies": copies, "graph_ms": time_graph(fn, sets)}
    if eager:
        out["eager_ms"] = time_eager(fn, sets)
    del sets
    torch.cuda.empty_cache()
    return out


def metrics(ms: float, bytes_moved: int, flops: int, peak_gbps: float,
            peak_tflops: float) -> dict:
    s = ms * 1e-3
    gbps = bytes_moved / s / 1e9
    tflops = flops / s / 1e12 if flops else 0.0
    return {
        "gbps": gbps,
        "pct_peak_bw": 100 * gbps / peak_gbps,
        "tflops": tflops,
        "pct_peak_flops": 100 * tflops / peak_tflops if flops else 0.0,
        "intensity": flops / bytes_moved if flops else 0.0,
    }
