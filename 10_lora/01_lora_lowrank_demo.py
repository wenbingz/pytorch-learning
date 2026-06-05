"""
LoRA intuition in pure PyTorch:

  W_eff = W_frozen + (A @ B)   # A: (out, r), B: (r, in), r << in,out

Only train A and B. This is the same idea as PEFT LoRA on Linear layers.
"""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        out_f, in_f = base.weight.shape
        self.A = nn.Parameter(torch.zeros(out_f, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_f))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ (self.A @ self.B).T) * self.scale

    def trainable_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def frozen_params(self) -> int:
        return sum(p.numel() for p in self.base.parameters())


def main() -> None:
    torch.manual_seed(0)

    in_f, out_f, rank = 32, 16, 4
    base = nn.Linear(in_f, out_f)
    lora = LoRALinear(base, rank=rank, alpha=8.0)

    x = torch.randn(8, in_f)
    y = lora(x)

    print("base weight shape:", tuple(base.weight.shape))
    print("LoRA A shape:", tuple(lora.A.shape), "B shape:", tuple(lora.B.shape))
    print("frozen params:", lora.frozen_params())
    print("trainable params:", lora.trainable_params())
    print("trainable ratio:", round(lora.trainable_params() / (lora.frozen_params() + lora.trainable_params()), 6))
    print("output shape:", tuple(y.shape))
    print()
    print("Lesson 6 parallel: freeze backbone ~= freeze W; train head ~= train A@B")


if __name__ == "__main__":
    main()
