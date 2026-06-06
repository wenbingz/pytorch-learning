"""
torch.export: capture a model graph for deployment (PyTorch 2.x).

Run:
    python3 15_deployment/01_torch_export_basic.py
"""

from __future__ import annotations

import torch
from torch import nn


class MNISTMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    model = MNISTMLP().eval()
    example = (torch.randn(1, 1, 28, 28),)
    ep = torch.export.export(model, example)

    # exported graph currently fixes batch=1; re-export with dynamic_shapes for variable batch
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        y_eager = model(x)
        y_export = ep.module()(x)

    max_diff = (y_eager - y_export).abs().max().item()
    print("exported program:", type(ep).__name__)
    print("max diff eager vs exported:", max_diff)
    print()
    print("this demo exports with batch=1; ONNX demo below uses dynamic batch")
    print("torch.export captures a graph for serving / edge / other runtimes")
    print("next: export the same model to ONNX for onnxruntime")


if __name__ == "__main__":
    main()
