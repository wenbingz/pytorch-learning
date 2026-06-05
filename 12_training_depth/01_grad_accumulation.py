"""
Gradient accumulation: simulate a larger batch without larger memory.

effective_batch = batch_size * accumulation_steps
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    accum_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0

    for step, (xb, yb) in enumerate(loader, start=1):
        logits = model(xb)
        loss = loss_fn(logits, yb) / accum_steps
        loss.backward()

        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        n_batches += 1

    if n_batches % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, n_batches)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    y = torch.randint(0, 2, (256,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = TinyClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    loss_normal = train_one_epoch(model, loader, optim, loss_fn, accum_steps=1)
    loss_accum = train_one_epoch(model, loader, optim, loss_fn, accum_steps=4)

    print("batch_size:", 16)
    print("accum_steps=1  (effective batch 16) loss:", round(loss_normal, 4))
    print("accum_steps=4  (effective batch 64) loss:", round(loss_accum, 4))
    print("idea: call optimizer.step() every N micro-batches")


if __name__ == "__main__":
    main()
