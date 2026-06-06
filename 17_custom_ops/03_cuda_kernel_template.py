"""
CUDA extension template: structure + build notes (compile on Linux + NVIDIA toolkit).

On Mac this prints the template only.

Run:
    python3 17_custom_ops/03_cuda_kernel_template.py
"""

from __future__ import annotations

import textwrap

import torch


def main() -> None:
    cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void axpy_kernel(const float* x, float a, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

torch::Tensor axpy_cuda(torch::Tensor x, double a, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be CUDA");
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  axpy_kernel<<<blocks, threads>>>(x.data_ptr<float>(), (float)a, y.data_ptr<float>(), n);
  return y;
}
"""

    cpp_source = r"""
#include <torch/extension.h>
torch::Tensor axpy_cuda(torch::Tensor x, double a, torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axpy", &axpy_cuda, "y += a * x on CUDA");
}
"""

    print("minimal CUDA op: y += a * x\n")
    print("--- device kernel (.cu) ---")
    print(textwrap.indent(cuda_source.strip(), "  "))
    print("\n--- pybind binding (.cpp) ---")
    print(textwrap.indent(cpp_source.strip(), "  "))

    print("\nPython build (Linux machine with CUDA toolkit):")
    print(
        textwrap.dedent(
            """
      from torch.utils.cpp_extension import load
      mod = load(name="axpy_cuda", sources=["axpy.cu", "bind.cpp"], verbose=True)
      x = torch.randn(1024, device="cuda")
      y = torch.zeros(1024, device="cuda")
      mod.axpy(x, 2.0, y)  # y += 2*x
    """
        ).strip()
    )

    print("\nkey ideas:")
    for line in [
        "__global__ void kernel: GPU function; launch with <<<blocks, threads>>>",
        "each thread computes one index via blockIdx/blockDim/threadIdx",
        "PyTorch binding exposes C++ to Python (same pattern as 02 on CPU)",
        "backward: wrap in torch.autograd.Function or register with torch.library",
    ]:
        print(f"  - {line}")

    if torch.cuda.is_available():
        print("\ncuda is available here — save sources to files and use load() to compile.")
    else:
        print("\nno cuda on this machine — read template; compile when you have Linux + GPU.")


if __name__ == "__main__":
    main()
