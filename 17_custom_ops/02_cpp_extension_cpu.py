"""
C++ extension on CPU: same binding pattern as CUDA ops, runnable on Mac.

First run compiles the extension (may take ~30s).

Run:
    python3 17_custom_ops/02_cpp_extension_cpu.py
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor clamp_relu_cpu(torch::Tensor x) {
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  return torch::clamp_min(x, 0);
}
"""

module = load_inline(
    name="myops_cpu",
    cpp_sources=cpp_source,
    functions=["clamp_relu_cpu"],
    verbose=False,
)


def main() -> None:
    x = torch.linspace(-2, 2, 8, requires_grad=True)
    y_ref = torch.relu(x)
    y_cpp = module.clamp_relu_cpu(x)

    print("x:", x.detach().tolist())
    print("torch.relu:", y_ref.detach().tolist())
    print("cpp clamp_relu:", y_cpp.detach().tolist())
    print("max diff:", (y_ref - y_cpp).abs().max().item())

    loss = y_cpp.sum()
    loss.backward()
    print("grad (expect 1 where x>0 else 0):", x.grad.tolist())

    print()
    print("pattern: write C++ forward -> pybind via load_inline -> call from Python")
    print("CUDA version: same binding, but kernel runs on GPU with <<<blocks, threads>>>")


if __name__ == "__main__":
    main()
