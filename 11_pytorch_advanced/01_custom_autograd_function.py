"""
Custom autograd: implement a differentiable op with forward/backward.

Use when you need a new primitive in the computation graph.
"""

from __future__ import annotations

import torch


class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


def main() -> None:
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = MySquare.apply(x)
    y.sum().backward()
    print("x:", x.tolist())
    print("y = x^2:", y.tolist())
    print("dy/dx (should be 2x):", x.grad.tolist())

    # gradcheck: use float64 for stable finite-difference vs analytical comparison
    x64 = torch.randn(4, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        MySquare.apply,
        (x64,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print("gradcheck (float64):", ok)


if __name__ == "__main__":
    main()
