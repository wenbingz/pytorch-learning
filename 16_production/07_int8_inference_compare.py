"""
Compare fp32 vs int8 dynamic quant inference (post-training, CPU).

Works with production or basic checkpoints.

Run:
    python3 16_production/07_int8_inference_compare.py --task mnist
    python3 16_production/07_int8_inference_compare.py --task cifar --ckpt runs/cifar10_production/ckpt.pt --weights ema
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint
from utils.quant import setup_quant_engine

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class MNISTMLP(nn.Module):
    def __init__(self, hidden: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def default_ckpt(task: str) -> str:
    if task == "mnist":
        return "runs/mnist_production/ckpt.pt"
    return "runs/cifar10_production/ckpt.pt"


def make_loader(task: str, data_dir: str, batch_size: int) -> DataLoader:
    if task == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


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
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
    return correct / total


def bench(model: nn.Module, loader: DataLoader, batches: int = 50) -> float:
    model.eval()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(5):
            xb, _ = next(it)
            model(xb)
        it = iter(loader)
        t0 = time.perf_counter()
        for _ in range(batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                it = iter(loader)
                xb, _ = next(it)
            model(xb)
    return (time.perf_counter() - t0) / batches * 1000.0


def build_model(task: str, ckpt: dict, weights: str) -> nn.Module:
    if task == "mnist":
        hidden = int(ckpt.get("config", {}).get("hidden", 256))
        model = MNISTMLP(hidden=hidden)
    else:
        model = CIFAR10CNN()
    key = weights if weights in ckpt else "model"
    model.load_state_dict(ckpt[key])
    return model.eval()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fp32 vs int8 inference compare")
    p.add_argument("--task", choices=["mnist", "cifar"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights", default="ema", choices=["model", "ema"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt or default_ckpt(args.task))
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}; train production script first")
        return

    data_dir = args.data_dir or ("data/mnist" if args.task == "mnist" else "data/cifar10")
    setup_quant_engine()

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model(args.task, ckpt, args.weights)
    quant_layers = {nn.Linear} if args.task == "mnist" else {nn.Linear, nn.Conv2d}
    quant_model = torch.quantization.quantize_dynamic(model, quant_layers, dtype=torch.qint8)

    loader = make_loader(args.task, data_dir, args.batch_size)
    fp32_acc = accuracy(model, loader)
    int8_acc = accuracy(quant_model, loader)
    fp32_ms = bench(model, loader)
    int8_ms = bench(quant_model, loader)

    print(f"task={args.task} ckpt={ckpt_path} weights={args.weights}")
    print(f"checkpoint bytes fp32: {state_dict_bytes(model) / 1024:.1f} KB")
    print(f"checkpoint bytes int8: {state_dict_bytes(quant_model) / 1024:.1f} KB")
    print(f"test acc fp32: {fp32_acc:.4f}")
    print(f"test acc int8: {int8_acc:.4f}")
    print(f"latency fp32: {fp32_ms:.3f} ms/batch")
    print(f"latency int8: {int8_ms:.3f} ms/batch")
    print()
    print("route B note: int8 often shrinks size; speed on small models may not improve on CPU")
    print("ONNX int8 export is separate; this script is PyTorch ORT-free inference compare")


if __name__ == "__main__":
    main()
