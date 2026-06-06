"""Helpers for int8 quantization demos."""

from __future__ import annotations

import torch


def setup_quant_engine() -> str:
    """Pick a supported eager-mode quant backend (qnnpack on ARM Mac, fbgemm on x86)."""
    supported = torch.backends.quantized.supported_engines
    for name in ("qnnpack", "fbgemm", "onednn"):
        if name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(f"no quantized engine available; supported={supported}")
