"""
fp16 inference: store/compute weights in half precision.

Real speedups are most visible on CUDA. On Mac CPU this is mostly an API demo.

Run:
    python3 14_inference/02_fp16_inference.py
"""

from __future__ import annotations

import time

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bench(model: nn.Module, x: torch.Tensor, steps: int = 200) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x32 = torch.randn(64, 1, 28, 28, device=device)
    model32 = SmallMLP().to(device)

    model16 = SmallMLP().to(device).half()
    x16 = x32.half()

    with torch.no_grad():
        out32 = model32(x32)
        out16 = model16(x16).float()

    max_diff = (out32 - out16).abs().max().item()
    ms32 = bench(model32, x32)
    ms16 = bench(model16, x16)

    print(f"max output diff (fp32 vs fp16): {max_diff:.6f}")
    print(f"latency fp32: {ms32:.3f} ms/step")
    print(f"latency fp16: {ms16:.3f} ms/step")
    if ms16 > 0:
        print(f"speedup fp16 vs fp32: {ms32 / ms16:.2f}x")
    print()
    if device.type != "cuda":
        print("on CPU, fp16 may be slower; use this script to learn the API")
        print("pattern: model.half(); x = x.half(); logits = model(x)")
    else:
        print("on CUDA, fp16 inference often helps most with larger batch/matmul")


if __name__ == "__main__":
    main()
