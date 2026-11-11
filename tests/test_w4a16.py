"""W4A16 Triton matmul vs an fp32 reference on the same dequantized weights.

Tolerance: atol=rtol=1e-2. Kernel and reference dequantize with identical fp16 rounding, so the only
differences are fp32 accumulation order and the final fp16 rounding of the output.
Weights are N(0, 0.02) like a trained LLM layer, so outputs are O(1).
"""

import pytest
import torch

from kernels.quant import quantize_rtn
from kernels.reference import w4a16_ref
from kernels.w4a16 import w4a16_matmul

TOL = dict(atol=1e-2, rtol=1e-2)


def _make(M, N, K, device, bias=False):
    w = (torch.randn(N, K, device=device) * 0.02).half()
    qw, s, z = quantize_rtn(w, 128)
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    b = (torch.randn(N, device=device) * 0.1).half() if bias else None
    return x, qw, s, z, b


def _check(M, N, K, device, bias=False, config=None):
    x, qw, s, z, b = _make(M, N, K, device, bias)
    out = w4a16_matmul(x, qw, s, z, 128, bias=b, config=config)
    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), w4a16_ref(x, qw, s, z, 128, b), **TOL)


CFG = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, SPLIT_K=1, GROUP_M=8)


@pytest.mark.parametrize("M,N,K", [(1, 64, 256), (3, 40, 128), (17, 96, 384)])
def test_direct_small(M, N, K, device):
    _check(M, N, K, device, config=CFG)


@pytest.mark.parametrize("split", [2, 3])
def test_split_k_small(split, device):
    _check(2, 48, 512, device, config={**CFG, "SPLIT_K": split})


def test_bias_both_paths(device):
    _check(1, 64, 256, device, bias=True, config=CFG)
    _check(1, 64, 256, device, bias=True, config={**CFG, "SPLIT_K": 2})


GEMV = dict(GEMV=True, BLOCK_N=32, BLOCK_K=64, SPLIT_K=1)


@pytest.mark.parametrize("N,K", [(64, 256), (40, 384), (96, 128)])
@pytest.mark.parametrize("split", [1, 3])
def test_gemv(N, K, split, device):
    _check(1, N, K, device, config={**GEMV, "SPLIT_K": split})


@pytest.mark.parametrize("split", [1, 2])
def test_gemv_factored(split, device):
    _check(1, 96, 384, device, config={**GEMV, "SPLIT_K": split, "FACTORED": True})
    _check(1, 64, 256, device, bias=True, config={**GEMV, "FACTORED": True, "BLOCK_K": 128})


def test_gemv_bias(device):
    _check(1, 64, 256, device, bias=True, config=GEMV)
    _check(1, 64, 256, device, bias=True, config={**GEMV, "SPLIT_K": 2, "BLOCK_K": 128})


def test_block_k_128(device):
    _check(5, 64, 256, device, config={**CFG, "BLOCK_K": 128, "BLOCK_N": 64})


def test_3d_input(device):
    x, qw, s, z, _ = _make(6, 32, 128, device)
    out = w4a16_matmul(x.view(2, 3, 128), qw, s, z, config=CFG)
    assert out.shape == (2, 3, 32)
    torch.testing.assert_close(out.view(6, 32).float(), w4a16_ref(x, qw, s, z), **TOL)


# Autotuned path on GPU: decode and prefill shapes from Qwen2.5-1.5B plus odd ones.
QWEN = [(1536, 1536), (256, 1536), (17920, 1536), (1536, 8960), (8960, 1536)]


@pytest.mark.slow
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N,K", QWEN)
def test_autotuned_decode(M, N, K, device):
    _check(M, N, K, device)


@pytest.mark.slow
@pytest.mark.parametrize("M", [33, 128, 2048])
@pytest.mark.parametrize("N,K", [(1536, 1536), (8960, 1536), (1536, 8960)])
def test_autotuned_prefill(M, N, K, device):
    _check(M, N, K, device)


@pytest.mark.slow
def test_autotuned_odd_and_bias(device):
    _check(3, 1000, 1024, device, bias=True)
    _check(100, 200, 640, device, bias=True)
