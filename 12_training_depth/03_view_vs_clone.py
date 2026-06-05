"""
view / reshape vs clone: memory sharing vs independent copy.
"""

from __future__ import annotations

import torch


def main() -> None:
    x = torch.arange(6)
    v = x.view(2, 3)

    print("x:", x)
    print("v:", v)
    print("v shares storage with x:", v.data_ptr() == x.data_ptr())

    v[0, 0] = 99
    print("after v[0,0]=99, x becomes:", x)

    c = x.view(2, 3).clone()
    c[0, 0] = -1
    print("clone change does not affect x:", x)

    print()
    print("view: cheap, shared memory, be careful with in-place")
    print("clone: independent tensor, safe but uses more memory")


if __name__ == "__main__":
    main()
