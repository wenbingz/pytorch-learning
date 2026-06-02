import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.device import get_default_device


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = get_default_device()
    model = TinyMLP().to(device)

    print("device:", device)
    print("model:", model)
    print()

    total_params = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        print(
            f"{name:20s} shape={tuple(p.shape)!s:12s} "
            f"dtype={str(p.dtype):12s} device={str(p.device):6s} requires_grad={p.requires_grad} numel={n}"
        )
    print()
    print("total params:", total_params)

    x = torch.randn(2, 4, device=device)
    y = model(x)
    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))


if __name__ == "__main__":
    main()

