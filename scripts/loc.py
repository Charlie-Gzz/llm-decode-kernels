"""Count non-blank, non-comment lines of the W4A16 kernels (Triton vs CUDA C++).

    python scripts/loc.py
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def py_function_loc(path: Path, name: str) -> int:
    src = path.read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = src.splitlines()[fn.lineno - 1: fn.end_lineno]
    doc = ast.get_docstring(fn, clean=False)
    text = "\n".join(lines)
    if doc:
        text = text.replace(doc, "")
    keep = [s for s in (line.strip() for line in text.splitlines())
            if s and not s.startswith("#") and s not in ('""""""', '"""')]
    return len(keep)


def cu_function_loc(path: Path, name: str) -> int:
    src = re.sub(r"//[^\n]*", "", path.read_text())
    start = src.index(f"{name}(")
    start = src.rfind("template", 0, start)
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        depth += src[j] == "{"
        depth -= src[j] == "}"
        if depth == 0:
            end = j
            break
    return sum(1 for l in src[start: end + 1].splitlines() if l.strip())


def counts() -> dict:
    tri = ROOT / "kernels" / "w4a16.py"
    cu = ROOT / "kernels" / "cuda" / "w4a16_decode.cu"
    return {
        "triton_gemm_kernel": py_function_loc(tri, "_w4a16_kernel"),
        "triton_gemv_kernel": py_function_loc(tri, "_w4a16_gemv_kernel"),
        "cuda_kernel": cu_function_loc(cu, "w4a16_decode_kernel"),
    }


if __name__ == "__main__":
    for k, v in counts().items():
        print(f"{k}: {v}")
