"""
EMA (exponential moving average) of model weights — common inference trick.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.99) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def main() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    ema = copy.deepcopy(model)

    x = torch.randn(4, 3)
    optim = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss()

    print("train model; keep shadow EMA weights for stabler inference")
    for step in range(5):
        loss = loss_fn(model(x), torch.randn(4, 2))
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        update_ema(ema, model, decay=0.9)
        print(f"step={step+1} loss={loss.item():.4f} w_norm={model.weight.norm().item():.4f} ema_norm={ema.weight.norm().item():.4f}")


if __name__ == "__main__":
    main()
