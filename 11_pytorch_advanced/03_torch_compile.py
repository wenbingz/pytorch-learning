"""
torch.compile (PyTorch 2+): graph capture + optimization.

Works on CPU; larger models benefit more.
"""

from __future__ import annotations

import time

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bench(model: nn.Module, x: torch.Tensor, steps: int = 50) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
        return time.perf_counter() - t0


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 1024)

    eager = MLP()
    compiled = torch.compile(MLP())

    t_eager = bench(eager, x)
    t_compiled = bench(compiled, x)

    print("eager seconds:", round(t_eager, 4))
    print("compile seconds:", round(t_compiled, 4))
    print("speedup:", round(t_eager / max(t_compiled, 1e-9), 2), "x")
    print("first compile run is slower (warmup); rerun script to see steady-state")


if __name__ == "__main__":
    main()
