"""Stretch: FP8 weight-only matmul vs an fp32 reference on the same dequantized weights.

Tolerance atol=rtol=1e-2: kernel and reference use the same fp8 values and scales; only
accumulation order and the output rounding differ.
"""

import pytest
import torch

from kernels.fp8 import dequantize_fp8, quantize_fp8, w8a16_matmul

TOL = dict(atol=1e-2, rtol=1e-2)


def _check(M, N, K, device, bias=False, config=None):
    w = (torch.randn(N, K, device=device) * 0.02).half()
    wq, s = quantize_fp8(w)
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    b = (torch.randn(N, device=device) * 0.1).half() if bias else None
    ref = x.float() @ dequantize_fp8(wq, s).float().t() + (b.float() if bias else 0)
    out = w8a16_matmul(x, wq, s, bias=b, config=config)
    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, **TOL)


def test_quantize_roundtrip_close(device):
    w = (torch.randn(64, 256, device=device) * 0.02).half()
    wq, s = quantize_fp8(w)
    assert wq.shape == (256, 64) and s.shape == (64,)
    err = (dequantize_fp8(wq, s).float() - w.float()).abs()
    # e4m3 has 3 mantissa bits: rounding error at most 1/16 of the value, plus a subnormal floor
    # of 2^-9 of the channel scale, plus fp16 rounding of the dequantized value.
    bound = w.float().abs() / 16 + s.float()[:, None] * 2**-9 + 1e-5
    assert (err <= bound).all()


CFG = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, SPLIT_K=1, GROUP_M=8)


@pytest.mark.parametrize("M,N,K", [(1, 64, 128), (5, 40, 96), (17, 96, 256)])
def test_fp8_direct(M, N, K, device):
    _check(M, N, K, device, config=CFG)


def test_fp8_split_and_gemv(device):
    _check(2, 48, 256, device, config={**CFG, "SPLIT_K": 3})
    _check(1, 48, 256, device, bias=True, config={"GEMV": True, "BLOCK_N": 32, "BLOCK_K": 64,
                                                  "SPLIT_K": 2})


@pytest.mark.slow
@pytest.mark.parametrize("M", [1, 4, 16, 512])
@pytest.mark.parametrize("N,K", [(17920, 1536), (1536, 8960), (2048, 1536)])
def test_fp8_autotuned(M, N, K, device):
    _check(M, N, K, device, bias=(N == 2048))
