"""Compile the CUDA C++ kernel with nvcc, no GPU needed: `docker compose run --rm nvcc`."""

import time

t0 = time.time()
from kernels.cuda.w4a16_cuda import _ext  # noqa: E402

mod = _ext()
print(f"built {mod.__name__} in {time.time() - t0:.1f}s, exports: "
      f"{[n for n in dir(mod) if not n.startswith('_')]}")
