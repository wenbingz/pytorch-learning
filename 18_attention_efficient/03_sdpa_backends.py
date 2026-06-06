"""
Inspect which SDPA backends PyTorch can use (CUDA shows flash/mem_efficient/math).

Run:
    python3 18_attention_efficient/03_sdpa_backends.py
"""

from __future__ import annotations

import torch
from torch.backends.cuda import sdp_kernel


def main() -> None:
    print("PyTorch SDPA backend flags (CUDA only):\n")
    if not torch.cuda.is_available():
        print("cuda not available — on Mac/CPU only the math fallback applies.")
        print("FlashAttention-style kernels require NVIDIA GPU + recent PyTorch.")
        return

    print("enable/disable backends when benchmarking:")
    print("  torch.backends.cuda.enable_flash_sdp(True)")
    print("  torch.backends.cuda.enable_mem_efficient_sdp(True)")
    print("  torch.backends.cuda.enable_math_sdp(True)")
    print()
    print("context manager for A/B tests:")
    print("  with sdp_kernel(enable_flash=True, enable_math=False):")
    print("      F.scaled_dot_product_attention(q, k, v)")
    print()
    print("current defaults:")
    print("  flash_sdp:", torch.backends.cuda.flash_sdp_enabled())
    print("  mem_efficient_sdp:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("  math_sdp:", torch.backends.cuda.math_sdp_enabled())
    print()
    print("tip: if flash is enabled and shapes/dtypes supported, SDPA picks it automatically")


if __name__ == "__main__":
    main()
