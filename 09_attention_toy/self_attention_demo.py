"""
Hands-on: see MultiheadAttention input/output shapes (mini Transformer building block).
"""

from __future__ import annotations

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(0)

    batch, seq_len, embed_dim, num_heads = 2, 5, 8, 2
    x = torch.randn(batch, seq_len, embed_dim)

    attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    out, weights = attn(x, x, x, need_weights=True)

    print("input x shape:", tuple(x.shape))          # (B, L, E)
    print("output shape:", tuple(out.shape))        # (B, L, E)
    print("attn weights shape:", tuple(weights.shape))  # (B, heads, L, L)
    print()
    print("Interpretation:")
    print("- each token attends to all tokens in the same sequence")
    print("- weights[b, h, i, j] = how much token i uses token j (head h)")
    print("- this is the core idea behind Transformer self-attention")


if __name__ == "__main__":
    main()
