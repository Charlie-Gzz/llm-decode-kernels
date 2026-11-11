"""RMSNorm kernel vs the HF-style PyTorch reference.

Tolerance: atol=rtol=1e-2 in fp16. The kernel and reference use the same rounding steps, so
differences come only from the order of the fp32 sum (a few fp16 ulps at most).
"""

import pytest
import torch

from kernels.reference import rmsnorm_ref
from kernels.rmsnorm import rmsnorm

TOL = dict(atol=1e-2, rtol=1e-2)
SMALL = [(1, 128), (3, 1000), (5, 1536)]
LARGE = [(1, 1536), (16, 4096), (4096, 1536), (2048, 8192), (7, 8960)]


def _inputs(M, N, device):
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    r = torch.randn(M, N, device=device, dtype=torch.float16)
    w = (1 + 0.1 * torch.randn(N, device=device)).half()
    return x, r, w


def _check(M, N, device):
    x, r, w = _inputs(M, N, device)
    torch.testing.assert_close(rmsnorm(x, w, 1e-6), rmsnorm_ref(x, w, 1e-6), **TOL)
    out, res = rmsnorm(x, w, 1e-6, residual=r)
    out_ref, res_ref = rmsnorm_ref(x, w, 1e-6, residual=r)
    torch.testing.assert_close(res, res_ref, atol=0, rtol=0)  # same fp16 add, must be exact
    torch.testing.assert_close(out, out_ref, **TOL)


@pytest.mark.parametrize("M,N", SMALL)
def test_rmsnorm_small(M, N, device):
    _check(M, N, device)


@pytest.mark.slow
@pytest.mark.parametrize("M,N", LARGE)
def test_rmsnorm_large(M, N, device):
    _check(M, N, device)


def test_rmsnorm_3d_input(device):
    x = torch.randn(2, 3, 256, device=device, dtype=torch.float16)
    w = torch.ones(256, device=device, dtype=torch.float16)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_ref(x, w), **TOL)
