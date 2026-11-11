"""End-to-end task: Qwen2.5-1.5B-Instruct, fp16 vs RTN 4-bit with our kernels.

Measures (each backend, same runner, CUDA graphs on both sides):
- decode tokens/sec at batch 1 (median of 3 runs of `decode_steps` greedy steps)
- prefill tokens/sec for prompt lengths in `prefill_lens`
- time in the swapped layers alone (all 28 layers' projections + norms + activation, M=1)
- quality: WikiText-2 perplexity on `ppl_windows` x 1024 tokens, and greedy outputs on prompts
- a parity check of the fp16 runner against Hugging Face's own model
"""

from __future__ import annotations

import gc
import statistics
import time

import torch
import torch.nn.functional as F

from kernels.e2e.model import QwenRunner, make_backend

PROMPTS = [
    "Explain in two sentences why LLM decoding is memory-bound.",
    "Write a haiku about a GPU that runs out of memory.",
    "What is the capital of Australia, and why is it not Sydney?",
]


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def load_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kernels.e2e.data import model_dir

    path = model_dir()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    model.eval()
    return tok, model


def chat_ids(tok, prompt: str) -> torch.Tensor:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.cuda()


# ----------------------------------------------------------------------------- decoding
class GraphDecoder:
    """Captures one greedy decode step (forward + argmax + bookkeeping) in a CUDA graph."""

    def __init__(self, runner: QwenRunner):
        self.r = runner
        dev = runner.embed.device
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(runner.max_len, dtype=torch.long, device=dev)
        self.graph = None

    def _step(self):
        logits = self.r.forward(self.tok, self.pos)
        nxt = logits[:, -1].argmax(-1)
        self.out.index_copy_(0, self.pos, nxt)
        self.tok.copy_(nxt.view(1, 1))
        self.pos.add_(1)

    def prefill(self, ids: torch.Tensor):
        P = ids.shape[1]
        logits = self.r.forward(ids, torch.arange(P, device=ids.device))
        self.tok.copy_(logits[:, -1].argmax(-1).view(1, 1))
        self.pos.fill_(P)
        self.out.zero_()
        self.out[P - 1] = self.tok.view(-1)[0]  # position P-1 holds the first new token
        return P

    def capture(self, ids):
        # Warm up (autotuning, CUDA kernel build) outside the graph, then reset state.
        self.prefill(ids)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.prefill(ids)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()

    def generate(self, ids, steps: int) -> tuple[list[int], float]:
        P = self.prefill(ids)
        torch.cuda.synchronize()
        start, end = _events()
        start.record()
        for _ in range(steps):
            self.graph.replay()
        end.record()
        torch.cuda.synchronize()
        tokens = self.out[P - 1: P - 1 + steps + 1].tolist()
        return tokens, start.elapsed_time(end) / 1e3


def decode_speed(runner, ids, steps: int, repeats: int = 3) -> dict:
    dec = GraphDecoder(runner)
    dec.capture(ids)
    rates = []
    for _ in range(repeats):
        _, sec = dec.generate(ids, steps)
        rates.append(steps / sec)
    return {"tok_s": statistics.median(rates), "tok_s_min": min(rates), "tok_s_max": max(rates),
            "ms_per_token": 1e3 / statistics.median(rates), "steps": steps}


def greedy_texts(runner, tok, steps: int = 64) -> list[dict]:
    dec = GraphDecoder(runner)
    out = []
    for i, p in enumerate(PROMPTS):
        ids = chat_ids(tok, p)
        if i == 0:
            dec.capture(ids)
        tokens, _ = dec.generate(ids, steps)
        eos = tok.eos_token_id
        if eos in tokens:
            tokens = tokens[: tokens.index(eos)]
        out.append({"prompt": p, "tokens": tokens,
                    "text": tok.decode(tokens, skip_special_tokens=True)})
    return out


# ----------------------------------------------------------------------------- prefill
def prefill_speed(runner, T: int, repeats: int = 5) -> dict:
    dev = runner.embed.device
    ids = torch.randint(0, 32000, (1, T), device=dev)
    pos = torch.arange(T, device=dev)
    for _ in range(2):
        runner.forward(ids, pos)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(ids, pos)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = _events()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    del graph
    t = statistics.median(times)
    return {"T": T, "ms": t * 1e3, "tok_s": T / t}


# ----------------------------------------------------------------------------- layers alone
def layers_only_ms(runner, M: int = 1) -> float:
    """Graph time of just the swapped ops for one decode step: per layer qkv, o, gate_up, down
    projections, two fused add+norms and the activation (no attention, embedding or lm_head)."""
    be, eps = runner.backend, runner.eps
    dev = runner.embed.device
    H = runner.embed.shape[1]
    x = torch.randn(1, M, H, device=dev, dtype=torch.float16)
    res = torch.randn_like(x)
    nq, hd = runner.n_heads, runner.head_dim

    def step():
        h, r = be.norm(x, runner.layers[0].ln1, eps, residual=res)
        for L in runner.layers:
            qkv = be.linear(h, L.qkv)
            o = be.linear(qkv[..., : nq * hd].contiguous(), L.o)
            h, r = be.norm(o, L.ln2, eps, residual=r)
            h = be.linear(be.act(be.linear(h, L.gate_up)), L.down)
            h, r = be.norm(h, L.ln1, eps, residual=r)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = _events()
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 10)
    del graph
    return statistics.median(times)


# ----------------------------------------------------------------------------- quality
def wikitext_ids(tok, n_tokens: int) -> torch.Tensor:
    from kernels.e2e.data import WIKITEXT

    text = WIKITEXT.read_text()
    ids = tok(text[: n_tokens * 8], return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


def perplexity(forward_logits, ids: torch.Tensor, window: int = 1024) -> float:
    nll, count = 0.0, 0
    for i in range(0, len(ids) - window + 1, window):
        chunk = ids[i: i + window].cuda().view(1, -1)
        logits = forward_logits(chunk)  # [1, T, V]
        loss = F.cross_entropy(logits[0, :-1].float(), chunk[0, 1:], reduction="sum")
        nll += loss.item()
        count += window - 1
    return float(torch.exp(torch.tensor(nll / count)))


def runner_logits(runner):
    def fn(chunk):
        return runner.forward(chunk, torch.arange(chunk.shape[1], device=chunk.device),
                              all_logits=True)
    return fn


# ----------------------------------------------------------------------------- main
def run(params: dict) -> dict:
    backends = params.get("backends", ["fp16", "ours"])
    decode_steps = int(params.get("decode_steps", 256))
    prefill_lens = params.get("prefill_lens", [128, 512, 2048])
    ppl_windows = int(params.get("ppl_windows", 40))
    quick = bool(params.get("quick", False))
    if quick:
        decode_steps, prefill_lens, ppl_windows = 32, [128], 2

    t0 = time.time()
    tok, hf = load_hf()
    result = {"model": hf.config._name_or_path, "decode_steps": decode_steps, "backends": {}}
    ids_ppl = wikitext_ids(tok, ppl_windows * 1024)
    prompt = chat_ids(tok, PROMPTS[0])

    # Parity: HF model vs our fp16 runner on the same prompt.
    with torch.no_grad():
        hf_logits = hf(prompt).logits[:, -1].float()
        hf_gen = hf.generate(prompt, max_new_tokens=32, do_sample=False)[0, prompt.shape[1]:]
        result["hf_ppl"] = perplexity(lambda c: hf(c).logits, ids_ppl)

    for name in backends:
        print(f"[e2e] backend {name} ({time.time() - t0:.0f}s)")
        runner = QwenRunner(hf, make_backend(name), max_len=2048)
        r = {"weights": runner.weight_bytes()}
        r["ppl"] = perplexity(runner_logits(runner), ids_ppl)
        r["greedy"] = greedy_texts(runner, tok)
        if name == "fp16":
            with torch.no_grad():
                ours = runner.forward(prompt, torch.arange(prompt.shape[1], device="cuda"))
            r["parity_max_abs_logit_diff"] = (ours[:, -1].float() - hf_logits).abs().max().item()
            dec = GraphDecoder(runner)
            dec.capture(prompt)
            toks, _ = dec.generate(prompt, 32)
            r["parity_greedy_match"] = sum(a == b for a, b in zip(toks, hf_gen.tolist())) / 32
        if name != "w4_ref":  # w4_ref is for quality only
            r["decode"] = decode_speed(runner, prompt, decode_steps)
            r["prefill"] = [prefill_speed(runner, T) for T in prefill_lens]
            r["layers_only_ms"] = layers_only_ms(runner)
        result["backends"][name] = r
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    if "ours" in backends:
        from kernels.w4a16 import best_configs

        result["triton_configs"] = best_configs()
    lines = []
    for name, r in result["backends"].items():
        d = r.get("decode", {})
        lines.append(f"{name:10s} ppl {r['ppl']:.3f}  decode {d.get('tok_s', 0):.1f} tok/s  "
                     f"layers {r.get('layers_only_ms', 0):.2f} ms  "
                     f"prefill {[round(p['tok_s']) for p in r.get('prefill', [])]}")
    result["summary"] = "\n".join(lines) + f"\nHF ppl {result['hf_ppl']:.3f}"
    result["ok"] = True
    return result
