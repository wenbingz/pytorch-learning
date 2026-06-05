"""
Gradient checkpointing: trade compute for memory.

Recompute activations in backward instead of storing all of them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DeepMLP(nn.Module):
    def __init__(self, depth: int = 12, width: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)]
        )
        self.head = nn.Linear(width, 10)
        self.width = width

    def forward(self, x: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 2048, requires_grad=False)
    y = torch.randint(0, 10, (32,))

    model = DeepMLP()
    loss_fn = nn.CrossEntropyLoss()

    for use_ckpt, name in [(False, "normal"), (True, "checkpoint")]:
        model.zero_grad(set_to_none=True)
        logits = model(x, use_ckpt=use_ckpt)
        loss = loss_fn(logits, y)
        loss.backward()
        print(f"[{name}] loss={loss.item():.4f}, grad on block0.weight norm={model.blocks[0][0].weight.grad.norm().item():.4f}")

    print("checkpointing usually lowers peak activation memory during backward")


if __name__ == "__main__":
    main()
