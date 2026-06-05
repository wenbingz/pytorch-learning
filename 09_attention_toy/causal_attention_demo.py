"""
Hands-on: causal (masked) self-attention used in GPT-style decoders.

Each position can only attend to previous positions (including itself).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    batch, seq_len, embed_dim = 1, 4, 6
    x = torch.randn(batch, seq_len, embed_dim)

    # scores: (B, L, L)
    scores = x @ x.transpose(-2, -1) / (embed_dim**0.5)

    # causal mask: upper triangle = -inf
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores_masked = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores_masked, dim=-1)
    out = attn @ x

    print("scores shape:", tuple(scores.shape))
    print("causal mask (1 = blocked):")
    print(mask.int())
    print()
    print("attention weights (row i sums to 1):")
    print(attn[0].round(decimals=3))
    print()
    print("output shape:", tuple(out.shape))
    print("GPT generates token by token using this kind of masked attention.")


if __name__ == "__main__":
    main()
