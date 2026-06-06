"""
Minimal Triton kernel: out = x + y

Mac: prints kernel source (triton pip wheel needs Linux+NVIDIA).
Linux+CUDA:
    pip install triton
    python3 19_triton/02_add_kernel.py

Run:
    python3 19_triton/02_add_kernel.py
"""

from __future__ import annotations

ADD_KERNEL_SOURCE = '''
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
'''


def run_if_possible() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl
    except ImportError:
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    # define kernel in-process when triton+cuda exist
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    import triton as triton_mod

    x = torch.randn(10000, device="cuda")
    y = torch.randn(10000, device="cuda")
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton_mod.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    ref = x + y
    print("cuda run ok, max diff:", (out - ref).abs().max().item())
    return True


def main() -> None:
    print("=== Triton vector add kernel ===\n")
    print(ADD_KERNEL_SOURCE.strip())
    print()
    if run_if_possible():
        print("executed on CUDA above.")
    else:
        print("triton not installed or cuda unavailable on this machine.")
        print("copy the source above to a Linux+NVIDIA box, then pip install triton.")


if __name__ == "__main__":
    main()
