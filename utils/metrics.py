from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        if not math.isfinite(value):
            return
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    if targets.device != preds.device:
        targets = targets.to(preds.device)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())

