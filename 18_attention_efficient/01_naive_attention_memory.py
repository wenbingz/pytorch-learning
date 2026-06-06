"""
Naive attention materializes the full (seq x seq) score matrix in memory.

Run:
    python3 18_attention_efficient/01_naive_attention_memory.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard textbook attention: explicitly build softmax(QK^T)V."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)  # (B, H, L, L)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out, scores


def main() -> None:
    batch, heads, seq_len, head_dim = 1, 8, 1024, 64
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    out, scores = naive_attention(q, k, v)
    score_mb = scores.numel() * scores.element_size() / (1024**2)

    print("shapes (batch, heads, seq, head_dim):", batch, heads, seq_len, head_dim)
    print("scores (QK^T) shape:", tuple(scores.shape))
    print(f"scores memory (fp32): {score_mb:.2f} MB")
    print("output shape:", tuple(out.shape))
    print()
    print("problem: seq_len doubles -> score matrix memory ~4x")
    print("FlashAttention idea: avoid storing full LxL matrix; tile in GPU SRAM")


if __name__ == "__main__":
    main()
