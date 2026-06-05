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
    # Ground truth: y = 3x + 2 + noise
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.rand(n, 1, generator=g) * 2 - 1).to(device)  # [-1, 1]
    noise = (0.1 * torch.randn(n, 1, generator=g)).to(device)
    y = 3.0 * x + 2.0 + noise
    return x, y


def train_with_optimizer(x: torch.Tensor, y: torch.Tensor, steps: int, lr: float) -> tuple[float, float, float]:
    """
    Math:
      θ <- θ - lr * ∇θ L(θ)

    Code:
      loss.backward() computes gradients into p.grad
      optimizer.step() applies the update to parameters
    """
    model = LinearRegressor().to(x.device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        pred = model(x)
        loss = loss_fn(pred, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    w = model.linear.weight.item()
    b = model.linear.bias.item()
    final_loss = loss_fn(model(x), y).item()
    return w, b, final_loss


def train_manually(x: torch.Tensor, y: torch.Tensor, steps: int, lr: float) -> tuple[float, float, float]:
    """
    Math:
      g_w = ∂L/∂w, g_b = ∂L/∂b
      w <- w - lr * g_w
      b <- b - lr * g_b

    Code:
      loss.backward() computes gradients into p.grad
      with torch.no_grad(): p -= lr * p.grad applies the update
      p.grad = None clears gradients (otherwise they accumulate)
    """
    model = LinearRegressor().to(x.device)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        for p in model.parameters():
            p.grad = None

    w = model.linear.weight.item()
    b = model.linear.bias.item()
    final_loss = loss_fn(model(x), y).item()
    return w, b, final_loss


def main() -> None:
    device = get_default_device()
    torch.manual_seed(0)

    x, y = make_synthetic_regression(n=2048, device=device)
    steps = 200
    lr = 0.1

    w1, b1, l1 = train_with_optimizer(x, y, steps=steps, lr=lr)
    w2, b2, l2 = train_manually(x, y, steps=steps, lr=lr)

    print("device:", device)
    print("steps:", steps, "lr:", lr)
    print()
    print("[optimizer.step()] w=", round(w1, 4), "b=", round(b1, 4), "loss=", round(l1, 6))
    print("[manual update  ] w=", round(w2, 4), "b=", round(b2, 4), "loss=", round(l2, 6))
    print()
    print("abs diff w:", abs(w1 - w2))
    print("abs diff b:", abs(b1 - b2))
    print("abs diff loss:", abs(l1 - l2))


if __name__ == "__main__":
    main()

