"""
parameters vs buffers, and in-place pitfalls.
"""

from __future__ import annotations

import torch
from torch import nn


class ModelWithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.register_buffer("running_mean", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.running_mean


def main() -> None:
    m = ModelWithBuffer()

    print("parameters:")
    for n, p in m.named_parameters():
        print(" ", n, tuple(p.shape), "requires_grad=", p.requires_grad)

    print("buffers:")
    for n, b in m.named_buffers():
        print(" ", n, tuple(b.shape), "requires_grad=", b.requires_grad)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = m(x).sum()
    y.backward()
    print("x.grad:", x.grad)

    # in-place danger: breaks autograd if x is needed for backward
    try:
        x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x2.add_(1.0)  # in-place on leaf tensor used in graph
        (x2 * 2).sum().backward()
    except RuntimeError as e:
        print("in-place error (expected):", str(e).split("\n")[0])


if __name__ == "__main__":
    main()
