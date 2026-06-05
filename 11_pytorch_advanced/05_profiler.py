"""
torch.profiler: find bottlenecks in forward/backward.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    model = MLP()
    x = torch.randn(64, 512)
    y = torch.randint(0, 10, (64,))
    loss_fn = nn.CrossEntropyLoss()

    activities = [ProfilerActivity.CPU]

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("train_step"):
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
