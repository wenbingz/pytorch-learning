from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    # Prefer Apple Silicon MPS when available; otherwise use CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str) -> torch.device:
    """
    Resolve a training device string.

    - cpu: stable default for small models on Mac
    - mps: Apple GPU (faster, but some ops can be unstable)
    - auto: mps if available else cpu
    """
    key = name.lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return torch.device("mps")
    if key == "auto":
        return get_default_device()
    raise ValueError(f"unknown device: {name!r} (expected cpu, mps, or auto)")


def device_summary() -> dict[str, object]:
    return {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "default_device": str(get_default_device()),
    }

