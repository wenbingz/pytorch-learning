"""
Compare PyTorch eager vs ONNXRuntime inference on the same random input.

Install once:
    pip install -r 15_deployment/requirements.txt

Run:
    python3 15_deployment/03_pytorch_vs_onnx_inference.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
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


def bench_torch(model: nn.Module, x: torch.Tensor, steps: int = 200) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
    return (time.perf_counter() - t0) / steps * 1000.0


def bench_onnx(session, x_np: np.ndarray, steps: int = 200) -> float:
    input_name = session.get_inputs()[0].name
    for _ in range(20):
        session.run(None, {input_name: x_np})
    t0 = time.perf_counter()
    for _ in range(steps):
        session.run(None, {input_name: x_np})
    return (time.perf_counter() - t0) / steps * 1000.0


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("missing deps; run: pip install -r 15_deployment/requirements.txt")
        return

    out_dir = Path("runs/deployment")
    onnx_path = out_dir / "mnist_mlp.onnx"
    weights_path = out_dir / "mnist_mlp.pt"
    if not onnx_path.exists():
        print(f"missing {onnx_path}; run 02_onnx_export.py first")
        return
    if not weights_path.exists():
        print(f"missing {weights_path}; re-run 02_onnx_export.py first")
        return

    model = MNISTMLP().eval()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    x = torch.randn(32, 1, 28, 28)
    x_np = x.numpy()

    with torch.no_grad():
        y_torch = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    y_onnx = session.run(None, {input_name: x_np})[0]

    max_diff = np.abs(y_torch - y_onnx).max()
    ms_torch = bench_torch(model, x)
    ms_onnx = bench_onnx(session, x_np)

    print(f"max diff pytorch vs onnx: {max_diff:.6f}")
    print(f"latency pytorch: {ms_torch:.3f} ms/step (batch=32)")
    print(f"latency onnxruntime: {ms_onnx:.3f} ms/step")
    if ms_onnx > 0:
        print(f"speedup onnx vs pytorch: {ms_torch / ms_onnx:.2f}x")
    print()
    print("deployment path: train PyTorch -> export ONNX -> serve with onnxruntime")
    print("for tiny models on CPU, speedup may be small; value is portability + prod stack")


if __name__ == "__main__":
    main()
