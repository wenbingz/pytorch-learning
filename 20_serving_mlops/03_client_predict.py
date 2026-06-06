"""
HTTP client for the FastAPI ONNX server.

Terminal 1:
    python3 20_serving_mlops/02_serve_fastapi.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx

Terminal 2:
    python3 20_serving_mlops/03_client_predict.py --task mnist
"""

from __future__ import annotations

import argparse

import numpy as np
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Call /health and /predict")
    p.add_argument("--task", choices=["mnist", "cifar"], default="mnist")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--index", type=int, default=0, help="test sample index")
    return p.parse_args()


def load_sample(task: str, index: int) -> tuple[list[float], int]:
    if task == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tf)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tf)

    x, y = ds[index]
    return x.numpy().reshape(-1).tolist(), int(y)


def main() -> None:
    try:
        import httpx
    except ImportError:
        print("pip install -r 20_serving_mlops/requirements.txt")
        return

    args = parse_args()
    values, label = load_sample(args.task, args.index)

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health").json()
        print("GET /health ->", health)
        resp = client.post("/predict", json={"values": values})
        resp.raise_for_status()
        body = resp.json()

    print(f"POST /predict -> pred={body['pred']} label={label} version={body['model_version']}")
    top_prob = max(body["probs"])
    print(f"top prob={top_prob:.4f}")


if __name__ == "__main__":
    main()
