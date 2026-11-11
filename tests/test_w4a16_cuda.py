"""CUDA C++ decode kernel vs the same fp32 reference as the Triton kernel (atol=rtol=1e-2).
GPU only; on CPU the kernel is compile-checked instead (`docker compose run --rm nvcc`)."""

import pytest
import torch

from kernels.quant import quantize_rtn
from kernels.reference import w4a16_ref

pytestmark = pytest.mark.gpu
TOL = dict(atol=1e-2, rtol=1e-2)


def _check(M, N, K, split=None, bias=False):
    from kernels.cuda.w4a16_cuda import w4a16_decode_cuda

    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    qw, s, z = quantize_rtn(w, 128)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = (torch.randn(N, device="cuda") * 0.1).half() if bias else None
    out = w4a16_decode_cuda(x, qw, s, z, bias=b, split_k=split)
    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), w4a16_ref(x, qw, s, z, 128, b), **TOL)


@pytest.mark.parametrize("M", [1, 2, 3, 4, 7, 8, 16])
@pytest.mark.parametrize("split", [1, 4])
def test_cuda_decode_qwen(M, split):
    _check(M, 1536, 1536, split)


@pytest.mark.parametrize("N,K", [(2048, 1536), (17920, 1536), (1536, 8960), (4096, 14336)])
def test_cuda_decode_autosplit(N, K):
    _check(1, N, K)
    _check(16, N, K)


def test_cuda_odd_n_and_bias():
    _check(5, 1000, 1024, split=3, bias=True)  # N not a multiple of 128
    _check(1, 132, 256, split=1, bias=True)


def test_cuda_rejects_large_m():
    with pytest.raises(RuntimeError):
        _check(17, 128, 128, split=1)
