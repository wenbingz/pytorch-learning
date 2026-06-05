"""
Mixed precision (AMP): float16/bfloat16 compute + float32 master weights.

Most useful on CUDA. On Mac CPU/MPS, read this as API pattern.
"""

from __future__ import annotations

import torch
from torch import nn


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = nn.Linear(128, 64).to(device)
    x = torch.randn(32, 128, device=device)
    y = torch.randint(0, 64, (32,), device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    optim.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

    if device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

    print("loss:", float(loss))
    print("pattern: autocast forward + (scaler on CUDA) + optimizer.step")
    print("on CPU/MPS this demo falls back to normal fp32 training")


if __name__ == "__main__":
    main()
