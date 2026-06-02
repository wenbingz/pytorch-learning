import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.device import get_default_device


class LinearRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_synthetic_regression(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # y = 3x + 2 + noise
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand(n, 1, generator=g).to(device) * 2 - 1  # [-1, 1]
    noise = 0.1 * torch.randn(n, 1, generator=g).to(device)
    y = 3.0 * x + 2.0 + noise
    return x, y


def main() -> None:
    device = get_default_device()
    torch.manual_seed(0)

    x, y = make_synthetic_regression(n=2048, device=device)
    model = LinearRegressor().to(device)

    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    print("device:", device)
    print("init weight:", model.linear.weight.item(), "bias:", model.linear.bias.item())

    for step in range(1, 201):
        pred = model(x)
        loss = loss_fn(pred, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step in {1, 2, 3, 5, 10, 20, 50, 100, 200}:
            w = model.linear.weight.item()
            b = model.linear.bias.item()
            print(f"step={step:3d} loss={loss.item():.6f} w={w:.4f} b={b:.4f}")

    print("final weight:", model.linear.weight.item(), "bias:", model.linear.bias.item())
    print("target weight ~3.0 bias ~2.0")


if __name__ == "__main__":
    main()

