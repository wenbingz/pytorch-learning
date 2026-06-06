"""
When do you need custom CUDA/C++ ops?

Run:
    python3 17_custom_ops/01_when_and_why.py
"""


def main() -> None:
    print("PyTorch op stack (slow -> fast, flexible -> rigid):\n")
    layers = [
        ("Python loop", "slowest, prototyping only"),
        ("torch.autograd.Function", "custom backward in Python; L11 demo"),
        ("compose existing ops", "e.g. einsum + matmul; often enough"),
        ("C++ extension (CPU/CUDA)", "new forward kernel; pybind11 binding"),
        ("CUDA kernel __global__", "hand-written GPU parallel code"),
        ("cuBLAS / cuDNN / FlashAttention", "vendor libraries PyTorch calls"),
    ]
    for name, note in layers:
        print(f"  {name:<28} {note}")

    print("\nreach for custom CUDA when:")
    reasons = [
        "no fused op exists and PyTorch issues many small kernels (memory bound)",
        "research needs a new primitive (e.g. special activation, sparse pattern)",
        "inference must fuse ops for TensorRT-like gains at source level",
    ]
    for r in reasons:
        print(f"  - {r}")

    print("\nusually skip custom CUDA when:")
    skips = [
        "standard matmul/conv/softmax — already heavily optimized",
        "model still in Python research phase — optimize after profiler (L11 05)",
        "you only have Mac/CPU — write C++ CPU ext first, CUDA later on Linux",
    ]
    for s in skips:
        print(f"  - {s}")

    print("\nnext scripts:")
    print("  02_cpp_extension_cpu.py   C++ op on CPU (works on Mac)")
    print("  03_cuda_kernel_template.py CUDA source + build notes (needs NVIDIA toolkit)")


if __name__ == "__main__":
    main()
