"""
Hands-on: one Transformer encoder block = Attention + MLP + residuals.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)  # (batch, seq, embed)
    block = TransformerBlock(embed_dim=32, num_heads=4, mlp_dim=64)

    y = block(x)
    print("in :", tuple(x.shape))
    print("out:", tuple(y.shape))
    print()
    print("Stack many blocks -> Transformer.")
    print("Decoder adds causal mask (see causal_attention_demo.py).")
    print("LLM ~= token embedding + N decoder blocks + lm_head.")


if __name__ == "__main__":
    main()
