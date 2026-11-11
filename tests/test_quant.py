import pytest
import torch

from kernels.quant import pack_int4, quantize_rtn, unpack_int4, weight_bytes
from kernels.reference import dequantize_ref


def test_pack_roundtrip():
    q = torch.randint(0, 16, (64, 24), dtype=torch.int32)
    packed = pack_int4(q)
    assert packed.shape == (8, 24) and packed.dtype == torch.int32
    assert torch.equal(unpack_int4(packed), q)


def test_pack_extremes():
    q = torch.full((16, 3), 15, dtype=torch.int32)  # all bits set -> int32 -1
    assert torch.equal(pack_int4(q), torch.full((2, 3), -1, dtype=torch.int32))
    assert torch.equal(unpack_int4(pack_int4(q)), q)


@pytest.mark.parametrize("N,K", [(32, 128), (40, 384)])
def test_quantize_error_bound(N, K):
    w = (torch.randn(N, K) * 0.02).half()
    qw, s, z = quantize_rtn(w, 128)
    assert qw.shape == (K // 8, N) and s.shape == (K // 128, N) and z.shape == (K // 128, N)
    deq = dequantize_ref(qw, s, z, 128)
    assert deq.shape == w.shape
    # RTN error is at most half a step (plus fp16 rounding of scale/zero).
    step = s.float().repeat_interleave(128, 0).t()
    err = (deq.float() - w.float()).abs()
    assert (err <= 0.5 * step + 1e-3).all()


def test_weight_bytes():
    assert weight_bytes(1536, 1536) == 1536 * 1536 // 2 + 2 * 12 * 1536 * 2
