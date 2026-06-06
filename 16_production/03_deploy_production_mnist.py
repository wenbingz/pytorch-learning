"""
Deploy production MNIST checkpoint (EMA weights) via ONNXRuntime.

Run after:
    python3 16_production/01_train_mnist_production.py --epochs 10

Then:
    python3 16_production/03_deploy_production_mnist.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint


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


def make_test_loader(data_dir: str, batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def eval_pytorch(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


def eval_onnx(session, loader: DataLoader) -> float:
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for xb, yb in loader:
        preds = session.run(None, {input_name: xb.numpy()})[0].argmax(axis=1)
        correct += (preds == yb.numpy()).sum()
        total += yb.size(0)
    return correct / total


def main() -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("missing deps; run: pip install -r 15_deployment/requirements.txt")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/mnist_production/ckpt.pt")
    parser.add_argument("--out-dir", default="runs/deployment/mnist_production")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}; run 01_train_mnist_production.py first")
        return

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    hidden = int(ckpt.get("config", {}).get("hidden", 256))
    best_ema_acc = float(ckpt.get("best_ema_acc", float("nan")))

    model = MNISTMLP(hidden=hidden).eval()
    model.load_state_dict(ckpt["ema"])
    print(f"loaded EMA weights from {ckpt_path} (best_ema_acc={best_ema_acc:.4f})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "mnist_mlp_ema.onnx"

    torch.onnx.export(
        model,
        torch.randn(1, 1, 28, 28),
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"exported {onnx_path}")

    loader = make_test_loader("data/mnist", batch_size=256)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pt_acc = eval_pytorch(model, loader)
    ort_acc = eval_onnx(session, loader)
    print(f"test acc pytorch(ema)={pt_acc:.4f} onnx={ort_acc:.4f}")


if __name__ == "__main__":
    main()
