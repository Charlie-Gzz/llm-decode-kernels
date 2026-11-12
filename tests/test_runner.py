"""The end-to-end runner on a tiny random Qwen2, checked on CPU before any GPU time is spent.

1. fp32 runner vs Hugging Face's own Qwen2 forward: same logits (atol 1e-4).
2. our kernels vs the dequantize-then-matmul reference, same 4-bit weights: close logits.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from kernels.e2e.model import QwenRunner, make_backend  # noqa: E402


def _tiny_model(dtype, device):
    cfg = transformers.Qwen2Config(
        vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return model


def test_runner_matches_hf_fp32(device):
    model = _tiny_model(torch.float32, device)
    ids = torch.randint(0, 512, (1, 12), device=device)
    with torch.no_grad():
        ref = model(ids).logits
    runner = QwenRunner(model, make_backend("fp16"), max_len=64)  # fp16 backend = plain torch ops
    out = runner.forward(ids, torch.arange(12, device=device), all_logits=True)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_runner_incremental_decode_matches_full(device):
    """Prefill 8 tokens then decode 4 one at a time == one 12-token forward (KV cache works)."""
    model = _tiny_model(torch.float32, device)
    ids = torch.randint(0, 512, (1, 12), device=device)
    runner = QwenRunner(model, make_backend("fp16"), max_len=64)
    full = runner.forward(ids, torch.arange(12, device=device), all_logits=True)
    runner.forward(ids[:, :8], torch.arange(8, device=device))
    for t in range(8, 12):
        step = runner.forward(ids[:, t: t + 1], torch.tensor([t], device=device))
        torch.testing.assert_close(step[0, -1], full[0, t], atol=1e-4, rtol=1e-4)


def test_ours_matches_w4_reference(device):
    model = _tiny_model(torch.float16, device)
    ids = torch.randint(0, 512, (1, 5), device=device)
    pos = torch.arange(5, device=device)
    ref = QwenRunner(model, make_backend("w4_ref"), max_len=32).forward(ids, pos, all_logits=True)
    ours = QwenRunner(model, make_backend("ours"), max_len=32).forward(ids, pos, all_logits=True)
    # Same weights; differences come from fp16 rounding order across 2 layers.
    torch.testing.assert_close(ours.float(), ref.float(), atol=5e-2, rtol=5e-2)


def test_ours_lm_matches_w4_ref_lm(device):
    """lm_head quantized too: our kernels still match the dequantize-then-matmul reference."""
    model = _tiny_model(torch.float16, device)
    ids = torch.randint(0, 512, (1, 5), device=device)
    pos = torch.arange(5, device=device)
    ref_r = QwenRunner(model, make_backend("w4_ref_lm"), max_len=32)
    ours_r = QwenRunner(model, make_backend("ours_lm"), max_len=32)
    assert ours_r.weight_bytes()["lm_head_quantized"]
    ref = ref_r.forward(ids, pos, all_logits=True)
    ours = ours_r.forward(ids, pos, all_logits=True)
    torch.testing.assert_close(ours.float(), ref.float(), atol=5e-2, rtol=5e-2)
    plain = QwenRunner(model, make_backend("w4_ref"), max_len=32).forward(ids, pos, all_logits=True)
    assert not torch.equal(ref, plain)  # the head really is quantized


def test_lm_suffix_needs_quantized_backend():
    with pytest.raises(ValueError):
        make_backend("fp16_lm")


def test_ours_fp8_close_to_fp16(device):
    """FP8 weights are much closer to fp16 than int4: logits within a few percent."""
    model = _tiny_model(torch.float16, device)
    ids = torch.randint(0, 512, (1, 5), device=device)
    pos = torch.arange(5, device=device)
    ref = QwenRunner(model, make_backend("fp16"), max_len=32).forward(ids, pos, all_logits=True)
    f8 = QwenRunner(model, make_backend("ours_fp8"), max_len=32).forward(ids, pos, all_logits=True)
    torch.testing.assert_close(f8.float(), ref.float(), atol=5e-2, rtol=5e-2)
