"""
Blocked matmul in Triton — same *idea* as FlashAttention tiling (simplified).

Read on Mac; run on Linux+CUDA with triton installed.

Run:
    python3 19_triton/03_blocked_matmul.py
"""

from __future__ import annotations

MATMUL_KERNEL_SNIPPET = '''
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # each program handles one BLOCK_M x BLOCK_N output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    c = acc.to(tl.float16)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)
'''


def main() -> None:
    print("why blocked (tiled) matmul matters\n")
    print("naive: read full rows/cols from HBM repeatedly")
    print("tiled: each GPU program keeps a BLOCK_M x BLOCK_N accumulator in fast SRAM")
    print("       inner loop loads BLOCK_K slices of A and B -> tl.dot -> acc")
    print()
    print("FlashAttention uses the same philosophy:")
    print("  tile over sequence blocks instead of materializing full L x L scores")
    print()
    print("--- Triton matmul kernel skeleton ---")
    print(MATMUL_KERNEL_SNIPPET.strip())
    print()
    print("Triton extras vs raw CUDA:")
    for line in [
        "tl.dot maps to tensor-core-friendly MMA on supported GPUs",
        "@triton.autotune can benchmark BLOCK_M/BLOCK_N/BLOCK_K configs",
        "Python syntax -> compiled to GPU code (no separate .cu file)",
    ]:
        print(f"  - {line}")
    print()
    print("full tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html")


if __name__ == "__main__":
    main()
