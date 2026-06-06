"""
Triton: write GPU kernels in Python-like syntax (CUDA machines only to run).

Run (Mac: read concepts; Linux+CUDA: install triton and run 02/03):
    python3 19_triton/01_what_is_triton.py
"""


def main() -> None:
    print("Triton vs hand-written CUDA C++\n")
    rows = [
        ("language", "Python + @triton.jit", "C++/CUDA + pybind11"),
        ("who uses it", "PyTorch/indie kernel hackers", "CUDA extension authors"),
        ("autotune", "built-in block size search", "manual tuning"),
        ("install", "pip install triton (Linux+NVIDIA)", "CUDA toolkit + ninja"),
        ("Mac", "read/learn only; no official wheel", "same — needs NVIDIA GPU"),
    ]
    print(f"{'':<12} {'Triton':<28} {'CUDA C++'}")
    print("-" * 60)
    for label, tri, cuda in rows:
        print(f"{label:<12} {tri:<28} {cuda}")

    print("\nmental model (same as CUDA):")
    print("  program_id(0)     ~ blockIdx.x")
    print("  tl.arange(BLOCK)  ~ threadIdx.x + offsets")
    print("  tl.load / tl.store ~ read/write global memory")
    print("  BLOCK_SIZE: tl.constexpr ~ template compile-time constant")

    print("\nwhen Triton shines:")
    for item in [
        "fused elementwise ops (bias + activation + scale)",
        "custom attention / softmax tiles (research prototypes)",
        "quick GPU kernel experiments without C++ compile cycles",
    ]:
        print(f"  - {item}")

    print("\nwhen to skip Triton:")
    for item in [
        "production already covered by cuBLAS / FlashAttention / SDPA",
        "you only have Mac — use L17 C++ CPU ext + read L19 templates",
    ]:
        print(f"  - {item}")

    print("\nnext:")
    print("  02_add_kernel.py        minimal vector add kernel")
    print("  03_blocked_matmul.py    tile idea behind FlashAttention")


if __name__ == "__main__":
    main()
