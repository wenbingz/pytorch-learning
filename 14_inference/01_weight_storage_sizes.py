"""
Compare how much memory model weights take in fp32 / fp16 / int8.

Run:
    python3 14_inference/01_weight_storage_sizes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.quant import setup_quant_engine


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_dict_bytes(model: nn.Module) -> int:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def main() -> None:
    engine = setup_quant_engine()
    print(f"quant engine: {engine}")

    fp32_model = SmallMLP()
    fp32_bytes = state_dict_bytes(fp32_model)

    fp16_model = SmallMLP()
    fp16_model.half()
    fp16_bytes = state_dict_bytes(fp16_model)

    int8_model = torch.quantization.quantize_dynamic(
        SmallMLP(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    int8_bytes = state_dict_bytes(int8_model)

    print("checkpoint size via state_dict() (good for deployment comparison):")
    print(f"  fp32: {fp32_bytes / 1024:.1f} KB")
    print(f"  fp16: {fp16_bytes / 1024:.1f} KB")
    print(f"  int8 dynamic quant (Linear weights): {int8_bytes / 1024:.1f} KB")
    print()
    print(f"fp16 / fp32 ratio: {fp16_bytes / fp32_bytes:.2f}")
    print(f"int8 / fp32 ratio: {int8_bytes / fp32_bytes:.2f}")
    print()
    print("inference idea:")
    print("  fp16: mainly saves bandwidth + some compute on GPU")
    print("  int8: smaller weights; strong speedups often on CPU with quant kernels")


if __name__ == "__main__":
    main()
