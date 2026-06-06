"""
PyTorch SDPA: scaled_dot_product_attention (may use FlashAttention on CUDA).

Run:
    python3 18_attention_efficient/02_sdpa_vs_naive.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def naive_attention(q, k, v, is_causal: bool = False):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
    if is_causal:
        seq = scores.shape[-1]
        mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def bench(fn, warmup: int = 10, steps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, heads, seq_len, head_dim = 2, 8, 512, 64

    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    with torch.no_grad():
        y_naive = naive_attention(q, k, v, is_causal=True)
        y_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = (y_naive - y_sdpa).abs().max().item()
    ms_naive = bench(lambda: naive_attention(q, k, v, is_causal=True))
    ms_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

    print("device:", device)
    print("causal self-attention, seq_len=", seq_len)
    print(f"max output diff naive vs sdpa: {max_diff:.6f}")
    print(f"latency naive: {ms_naive:.3f} ms")
    print(f"latency sdpa:  {ms_sdpa:.3f} ms")
    if ms_sdpa > 0:
        print(f"speedup sdpa vs naive: {ms_naive / ms_sdpa:.2f}x")
    print()
    if device.type == "cuda":
        print("on CUDA, sdpa often picks flash or memory-efficient kernels")
    else:
        print("on CPU/Mac, sdpa uses math backend; flash needs NVIDIA GPU")
    print("API: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)")


if __name__ == "__main__":
    main()
