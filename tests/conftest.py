"""Tests run in two places with the same code:

- CPU: `docker compose run --rm cpu` sets TRITON_INTERPRET=1, kernels run in Triton's interpreter.
- GPU: `python -m kernels.gpu pytest` runs them on an L4, including the `slow` (large) shapes.
"""

import os

import pytest
import torch

ON_GPU = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture
def device():
    return "cuda" if ON_GPU else "cpu"


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="large shape: GPU only (too slow in the interpreter)")
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "slow" in item.keywords and not ON_GPU:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not ON_GPU:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
