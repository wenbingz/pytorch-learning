"""
Learning-rate schedulers: change lr during training.
"""

from __future__ import annotations

import torch
from torch import nn, optim


def main() -> None:
    model = nn.Linear(4, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("StepLR: every 3 epochs, lr *= 0.5")
    for epoch in range(1, 8):
        # fake train step
        loss = model(torch.randn(8, 4)).pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} lr={lr:.4f}")

    print()
    print("tip: scheduler.step() is usually called once per epoch (or per step)")


if __name__ == "__main__":
    main()
