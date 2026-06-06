"""
Dynamic int8 quantization (PyTorch eager mode): quantize Linear weights at load time.

Works well for CPU inference demos on Mac.

Run:
    python3 14_inference/03_dynamic_quant_int8.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.quant import setup_quant_engine


class MNISTMLP(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_dict_bytes(model: nn.Module) -> int:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / max(1, total)


def bench(model: nn.Module, x: torch.Tensor, steps: int = 100) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
    return (time.perf_counter() - t0) / steps * 1000.0


def main() -> None:
    engine = setup_quant_engine()
    print(f"quant engine: {engine}")

    torch.manual_seed(0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="data/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MNISTMLP()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("quick fp32 train (3 epochs) to get a real checkpoint for comparison")
    for epoch in range(1, 4):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
        acc = accuracy(model, test_loader)
        print(f"  epoch={epoch} test_acc={acc:.4f}")

    quant_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    xb, _ = next(iter(test_loader))
    fp32_ms = bench(model, xb)
    int8_ms = bench(quant_model, xb)

    fp32_acc = accuracy(model, test_loader)
    int8_acc = accuracy(quant_model, test_loader)

    print()
    print(f"checkpoint bytes fp32: {state_dict_bytes(model) / 1024:.1f} KB")
    print(f"checkpoint bytes int8: {state_dict_bytes(quant_model) / 1024:.1f} KB")
    print(f"test acc fp32: {fp32_acc:.4f}")
    print(f"test acc int8: {int8_acc:.4f}")
    print(f"latency fp32: {fp32_ms:.3f} ms/step (batch={xb.size(0)})")
    print(f"latency int8: {int8_ms:.3f} ms/step")
    if int8_ms > 0:
        print(f"speedup int8 vs fp32: {fp32_ms / int8_ms:.2f}x")
    print()
    print("dynamic quant: only Linear weights -> int8; activations stay fp32")
    print("accuracy drop is often small for MLP/CNN; always measure on your task")


if __name__ == "__main__":
    main()
