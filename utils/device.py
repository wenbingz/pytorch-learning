from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    # Prefer Apple Silicon MPS when available; otherwise use CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary() -> dict[str, object]:
    return {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "default_device": str(get_default_device()),
    }

