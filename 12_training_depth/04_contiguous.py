"""
contiguous(): some ops need row-major contiguous layout for fast kernels.
"""

from __future__ import annotations

import torch


def main() -> None:
    x = torch.arange(12).reshape(3, 4)
    t = x.t()  # transpose view, often non-contiguous

    print("x contiguous:", x.is_contiguous())
    print("x.t() contiguous:", t.is_contiguous())

    t2 = t.contiguous()
    print("after .contiguous():", t2.is_contiguous())
    print("same values:", torch.equal(t, t2))

    print()
    print("before matmul on CPU, PyTorch may call contiguous() internally")


if __name__ == "__main__":
    main()
