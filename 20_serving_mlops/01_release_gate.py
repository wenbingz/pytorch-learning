"""
Write a release manifest after deploy checks pass.

Run after export + acc verification:
    python3 20_serving_mlops/01_release_gate.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx --min-acc 0.97
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def make_test_loader(task: str, data_dir: str, batch_size: int) -> DataLoader:
    if task == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def eval_onnx_acc(onnx_path: Path, loader: DataLoader) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for xb, yb in loader:
        logits = session.run(None, {input_name: xb.numpy()})[0]
        preds = logits.argmax(axis=1)
        correct += (preds == yb.numpy()).sum()
        total += yb.size(0)
    return correct / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy gate: block bad ONNX releases")
    p.add_argument("--task", choices=["mnist", "cifar"], required=True)
    p.add_argument("--onnx", required=True, help="path to onnx model")
    p.add_argument("--min-acc", type=float, required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="manifest json path")
    p.add_argument("--note", default="")
    return p.parse_args()


def main() -> None:
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except ImportError:
        print("pip install -r 15_deployment/requirements.txt")
        return

    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"missing onnx: {onnx_path}")
        return

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    data_dir = args.data_dir or ("data/mnist" if args.task == "mnist" else "data/cifar10")
    loader = make_test_loader(args.task, data_dir, args.batch_size)
    acc = eval_onnx_acc(onnx_path, loader)

    print(f"task={args.task} onnx={onnx_path}")
    print(f"ort test acc={acc:.4f}  required>={args.min_acc:.4f}")

    if acc < args.min_acc:
        print("GATE FAILED: acc below threshold; do not deploy")
        raise SystemExit(1)

    manifest_dir = Path("runs/releases")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else manifest_dir / f"{args.task}_{version}.json"

    manifest = {
        "task": args.task,
        "version": version,
        "onnx_path": str(onnx_path.resolve()),
        "test_acc": round(acc, 6),
        "min_acc": args.min_acc,
        "pytorch_version": torch.__version__,
        "exported_at_utc": version,
        "note": args.note,
        "status": "approved",
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"GATE PASSED: manifest -> {out_path}")

    log_path = Path("runs/experiments.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "release_approved", **manifest}) + "\n")
    print(f"logged -> {log_path}")


if __name__ == "__main__":
    main()
