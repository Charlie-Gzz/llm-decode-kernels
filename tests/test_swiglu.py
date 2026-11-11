"""SwiGLU kernel vs F.silu(gate) * up. Tolerance atol=rtol=1e-2 in fp16 (sigmoid: a few ulps)."""

import pytest
import torch

from kernels.reference import swiglu_ref
from kernels.swiglu import swiglu, swiglu_fused

TOL = dict(atol=1e-2, rtol=1e-2)
SMALL = [(1, 64), (3, 1000), (4, 2048)]
LARGE = [(1, 8960), (16, 8960), (2048, 8960), (333, 4097)]


def _check(M, I, device):
    gate = torch.randn(M, I, device=device, dtype=torch.float16) * 3
    up = torch.randn(M, I, device=device, dtype=torch.float16)
    torch.testing.assert_close(swiglu(gate, up), swiglu_ref(gate, up), **TOL)
    fused = torch.cat([gate, up], dim=-1)
    torch.testing.assert_close(swiglu_fused(fused), swiglu_ref(gate, up), **TOL)


@pytest.mark.parametrize("M,I", SMALL)
def test_swiglu_small(M, I, device):
    _check(M, I, device)


@pytest.mark.slow
@pytest.mark.parametrize("M,I", LARGE)
def test_swiglu_large(M, I, device):
    _check(M, I, device)


def test_swiglu_3d(device):
    gu = torch.randn(2, 5, 512, device=device, dtype=torch.float16)
    out = swiglu_fused(gu)
    assert out.shape == (2, 5, 256)
    torch.testing.assert_close(out, swiglu_ref(gu[..., :256], gu[..., 256:]), **TOL)
