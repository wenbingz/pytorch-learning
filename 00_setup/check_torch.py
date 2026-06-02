import platform
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.device import device_summary, get_default_device


def main() -> None:
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())

    info = device_summary()
    print("torch:", info["torch_version"])
    print("mps available:", info["mps_available"])
    print("cuda available:", info["cuda_available"])

    device = get_default_device()
    x = torch.randn(3, 3, device=device)
    print("device:", x.device)
    print("x mean:", x.mean().item())


if __name__ == "__main__":
    main()

