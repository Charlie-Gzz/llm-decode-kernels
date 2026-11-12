"""A small Qwen2 runner built on the Hugging Face weights, with swappable kernels.

Why not HF `generate`: at batch 1, Python and launch overhead hide kernel gains, so both sides
must run under CUDA graphs. HF's mask/cache plumbing is hard to capture reliably, so this runner
has its own static KV cache, RoPE and SDPA call. fp16 and 4-bit use the *same* runner and the same
fusions (fused qkv and gate/up projections); only the linear/norm/activation ops differ.
The fp16 runner is checked against HF's own model (logits and greedy tokens) in kernels/e2e/run.py.

Kept in fp16 on every backend: embeddings, attention (SDPA), RoPE. lm_head (tied to the
embeddings) stays fp16 too, except on the `_lm` backends (e.g. `ours_lm`), which also quantize a
copy of it; the embedding lookup keeps the fp16 table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kernels import config


# ------------------------------------------------------------------------------------- ops
@dataclass
class Linear:
    """One projection. fp16 holds `w`; 4-bit holds the packed tensors."""

    w: torch.Tensor | None = None
    qweight: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    zeros: torch.Tensor | None = None
    bias: torch.Tensor | None = None


class Backend:
    """fp16 eager PyTorch ops, matching HF Qwen2 numerics."""

    name = "fp16"
    quantized = False

    def linear(self, x, p: Linear):
        return F.linear(x, p.w, p.bias)

    def norm(self, x, w, eps, residual=None):
        from kernels.reference import rmsnorm_ref

        return rmsnorm_ref(x, w, eps, residual)

    def act(self, gate_up):
        i = gate_up.shape[-1] // 2
        return F.silu(gate_up[..., :i]) * gate_up[..., i:]


class W4RefBackend(Backend):
    """RTN 4-bit weights, dequantized to fp16 then cuBLAS. Same quality as our kernels, used to
    show that any quality change comes from quantization, not from the kernels."""

    name = "w4_ref"
    quantized = True

    def linear(self, x, p: Linear):
        from kernels.reference import dequantize_ref

        if p.w is None:  # dequantize once, cache
            p.w = dequantize_ref(p.qweight, p.scales, p.zeros, config.GROUP_SIZE)
        return F.linear(x, p.w, p.bias)


class OursBackend(Backend):
    """Our kernels: Triton W4A16 (or the CUDA decode kernel), fused RMSNorm+residual, SwiGLU."""

    quantized = True

    def __init__(self, decode_kernel: str = "triton"):
        self.decode_kernel = decode_kernel
        self.name = "ours" if decode_kernel == "triton" else f"ours_{decode_kernel}"

    def linear(self, x, p: Linear):
        from kernels.w4a16 import w4a16_matmul

        rows = x.numel() // x.shape[-1]
        if self.decode_kernel == "cuda" and rows <= 16:
            from kernels.cuda.w4a16_cuda import w4a16_decode_cuda

            return w4a16_decode_cuda(x, p.qweight, p.scales, p.zeros, bias=p.bias)
        return w4a16_matmul(x, p.qweight, p.scales, p.zeros, bias=p.bias)

    def norm(self, x, w, eps, residual=None):
        from kernels.rmsnorm import rmsnorm

        return rmsnorm(x, w, eps, residual)

    def act(self, gate_up):
        from kernels.swiglu import swiglu_fused

        return swiglu_fused(gate_up)


class OursFP8Backend(OursBackend):
    """Stretch: FP8 (e4m3) weight-only linears, same fused norm and activation kernels."""

    name = "ours_fp8"

    def __init__(self):
        super().__init__("triton")
        self.name = "ours_fp8"

    def quantize(self, w, b) -> Linear:
        from kernels.fp8 import quantize_fp8

        wq, s = quantize_fp8(w)
        return Linear(qweight=wq, scales=s, bias=b)

    def linear(self, x, p: Linear):
        from kernels.fp8 import w8a16_matmul

        return w8a16_matmul(x, p.qweight, p.scales, bias=p.bias)


def make_backend(name: str) -> Backend:
    """`<name>_lm` = the quantized backend `<name>` with lm_head quantized as well."""
    base, quant_lm_head = (name[:-3], True) if name.endswith("_lm") else (name, False)
    be = {
        "fp16": Backend,
        "w4_ref": W4RefBackend,
        "ours": lambda: OursBackend("triton"),
        "ours_cuda": lambda: OursBackend("cuda"),
        "ours_fp8": OursFP8Backend,
    }[base]()
    if quant_lm_head:
        if not be.quantized:
            raise ValueError(f"{name}: only quantized backends can quantize lm_head")
        be.name = name
    be.quant_lm_head = quant_lm_head
    return be


# ------------------------------------------------------------------------------------- model
@dataclass
class Layer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    qkv: Linear
    o: Linear
    gate_up: Linear
    down: Linear


class QwenRunner:
    def __init__(self, hf_model, backend: Backend, max_len: int = 2048, batch: int = 1):
        cfg = hf_model.config
        self.backend = backend
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.max_len = max_len
        m = hf_model.model
        dev = m.embed_tokens.weight.device
        self.embed = m.embed_tokens.weight
        self.lm_head = hf_model.lm_head.weight
        self.final_norm = m.norm.weight
        theta = cfg.rope_parameters["rope_theta"]
        d = self.head_dim
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=dev, dtype=torch.float) / d))

        def lin(*mods) -> Linear:
            w = torch.cat([mm.weight.data for mm in mods], 0)
            b = None
            if mods[0].bias is not None:
                b = torch.cat([mm.bias.data for mm in mods], 0)
            if not backend.quantized:
                return Linear(w=w, bias=b)
            if hasattr(backend, "quantize"):  # formats other than int4 RTN (FP8)
                return backend.quantize(w, b)
            from kernels.quant import quantize_rtn

            qw, s, z = quantize_rtn(w, config.GROUP_SIZE)
            return Linear(qweight=qw, scales=s, zeros=z, bias=b)

        # Optional quantized copy of lm_head (the fp16 table stays for the embedding lookup).
        self.head_q = lin(hf_model.lm_head) if getattr(backend, "quant_lm_head", False) else None
        self.layers = []
        for layer in m.layers:
            a, mlp = layer.self_attn, layer.mlp
            self.layers.append(Layer(
                ln1=layer.input_layernorm.weight, ln2=layer.post_attention_layernorm.weight,
                qkv=lin(a.q_proj, a.k_proj, a.v_proj), o=lin(a.o_proj),
                gate_up=lin(mlp.gate_proj, mlp.up_proj), down=lin(mlp.down_proj),
            ))
        shape = (len(self.layers), batch, self.n_kv, max_len, self.head_dim)
        self.k_cache = torch.zeros(shape, device=dev, dtype=self.embed.dtype)
        self.v_cache = torch.zeros(shape, device=dev, dtype=self.embed.dtype)

    def weight_bytes(self) -> dict:
        """Bytes of the swapped layers vs the parts that stay fp16."""
        def nbytes(t):
            return 0 if t is None else t.numel() * t.element_size()

        def size(p: Linear):
            if self.backend.quantized:
                return nbytes(p.qweight) + nbytes(p.scales) + nbytes(p.zeros) + nbytes(p.bias)
            return nbytes(p.w) + nbytes(p.bias)

        layers = sum(size(p) for L in self.layers for p in (L.qkv, L.o, L.gate_up, L.down))
        head = size(self.head_q) if self.head_q is not None else nbytes(self.lm_head)
        return {"linear_layers": layers, "embed": nbytes(self.embed), "lm_head_read": head,
                "lm_head_quantized": self.head_q is not None,
                "lm_head_shared_with_embed": self.lm_head.data_ptr() == self.embed.data_ptr()}

    def _rope(self, x, cos, sin):
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, pos: torch.Tensor, all_logits: bool = False):
        """tokens [B, T]; pos: int64 tensor [T] of absolute positions. Graph-capturable."""
        be, eps = self.backend, self.eps
        B, T = tokens.shape
        x = F.embedding(tokens, self.embed)
        freqs = pos[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        keys = torch.arange(self.max_len, device=x.device)
        mask = keys[None, :] <= pos[:, None]  # [T, max_len], True = attend
        nq, nkv, hd = self.n_heads, self.n_kv, self.head_dim

        residual, h = x, None
        for i, L in enumerate(self.layers):
            if h is None:
                h = be.norm(x, L.ln1, eps)
            else:  # fused: residual += previous MLP output, then input norm of this layer
                h, residual = be.norm(h, L.ln1, eps, residual=residual)
            qkv = be.linear(h, L.qkv)
            q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            q = self._rope(q.view(B, T, nq, hd).transpose(1, 2), cos, sin)
            k = self._rope(k.view(B, T, nkv, hd).transpose(1, 2), cos, sin)
            v = v.view(B, T, nkv, hd).transpose(1, 2)
            self.k_cache[i].index_copy_(2, pos, k)
            self.v_cache[i].index_copy_(2, pos, v)
            attn = F.scaled_dot_product_attention(
                q, self.k_cache[i], self.v_cache[i], attn_mask=mask, enable_gqa=True)
            attn = attn.transpose(1, 2).reshape(B, T, nq * hd)
            o = be.linear(attn, L.o)
            h, residual = be.norm(o, L.ln2, eps, residual=residual)
            h = be.linear(be.act(be.linear(h, L.gate_up)), L.down)
        h, _ = be.norm(h, self.final_norm, eps, residual=residual)
        if not all_logits:
            h = h[:, -1:]
        if self.head_q is not None:
            return be.linear(h, self.head_q)
        return F.linear(h, self.lm_head)
