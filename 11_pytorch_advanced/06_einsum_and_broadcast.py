"""
einsum: express batched matrix ops clearly (attention-style).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_einsum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q,k,v: (B, L, E)
    scores = torch.einsum("ble,bse->bls", q, k) / (q.shape[-1] ** 0.5)
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("bls,bse->ble", weights, v)
    return out


def attention_bmm(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


def main() -> None:
    torch.manual_seed(0)
    b, l, e = 2, 5, 8
    q = torch.randn(b, l, e)
    k = torch.randn(b, l, e)
    v = torch.randn(b, l, e)

    o1 = attention_einsum(q, k, v)
    o2 = attention_bmm(q, k, v)
    print("max diff einsum vs bmm:", (o1 - o2).abs().max().item())
    print("einsum is often clearer for multi-axis tensor contracts")


if __name__ == "__main__":
    main()
