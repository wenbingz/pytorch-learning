"""
End-to-end: L4 MNIST checkpoint -> ONNX -> ONNXRuntime inference.

Prerequisite (if you don't have a checkpoint yet):
    python3 04_mlp_mnist/train_mnist.py --epochs 10

Deploy flow:
    python3 15_deployment/05_deploy_mnist_checkpoint.py
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
def eval_pytorch(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in loader:
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def eval_onnx(session, loader: DataLoader) -> tuple[float, float]:
    import onnxruntime as ort  # noqa: F401

    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    total_loss = 0.0

    for xb, yb in loader:
        logits = session.run(None, {input_name: xb.numpy()})[0]
        logits_t = torch.from_numpy(logits)
        total_loss += nn.functional.cross_entropy(logits_t, yb).item() * xb.size(0)
        preds = logits_t.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def max_logit_diff(model: nn.Module, session, loader: DataLoader, batches: int = 10) -> float:
    input_name = session.get_inputs()[0].name
    max_diff = 0.0
    for i, (xb, _) in enumerate(loader):
        if i >= batches:
            break
        y_torch = model(xb).numpy()
        y_onnx = session.run(None, {input_name: xb.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(y_torch - y_onnx).max()))
    return max_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy L4 MNIST checkpoint via ONNXRuntime")
    parser.add_argument("--ckpt", default="runs/mnist/ckpt.pt")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out-dir", default="runs/deployment/mnist")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("missing deps; run: pip install -r 15_deployment/requirements.txt")
        return

    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"missing checkpoint: {ckpt_path}")
        print("train first:")
        print("  python3 04_mlp_mnist/train_mnist.py --epochs 10")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "mnist_mlp.onnx"

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    hidden = int(ckpt.get("config", {}).get("hidden", 256))
    best_val_acc = float(ckpt.get("best_val_acc", float("nan")))

    model = MNISTMLP(hidden=hidden).eval()
    model.load_state_dict(ckpt["model"])

    print("step 1/4: loaded L4 checkpoint")
    print(f"  path: {ckpt_path}")
    print(f"  hidden: {hidden}")
    print(f"  best_val_acc in ckpt: {best_val_acc:.4f}")

    example = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    print()
    print("step 2/4: exported ONNX")
    print(f"  path: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    loader = make_test_loader(args.data_dir, args.batch_size)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    pt_loss, pt_acc = eval_pytorch(model, loader)
    ort_loss, ort_acc = eval_onnx(session, loader)
    diff = max_logit_diff(model, session, loader)

    print()
    print("step 3/4: test set metrics")
    print(f"  pytorch   loss={pt_loss:.4f} acc={pt_acc:.4f}")
    print(f"  onnxruntime loss={ort_loss:.4f} acc={ort_acc:.4f}")
    print(f"  max logit diff (sample batches): {diff:.6f}")

    xb, yb = next(iter(loader))
    x1 = xb[:1]
    y1 = int(yb[0].item())
    with torch.no_grad():
        pred_pt = int(model(x1).argmax(dim=1).item())
    pred_ort = int(session.run(None, {"images": x1.numpy()})[0].argmax())
    print()
    print("step 4/4: single-sample sanity check")
    print(f"  label={y1} pytorch_pred={pred_pt} onnx_pred={pred_ort}")
    print()
    print("deployment done: serve runs/deployment/mnist/mnist_mlp.onnx with onnxruntime")


if __name__ == "__main__":
    main()
