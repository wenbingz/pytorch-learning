import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # vocab size 10, embed dim 4
    embed = nn.Embedding(num_embeddings=10, embedding_dim=4, padding_idx=0)

    # batch of token ids; 0 = padding
    token_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 7],
        ]
    )

    vectors = embed(token_ids)
    print("token_ids shape:", tuple(token_ids.shape))      # (batch, seq_len)
    print("embedding vectors shape:", tuple(vectors.shape))  # (batch, seq_len, embed_dim)
    print()
    print("token_ids[0]:", token_ids[0].tolist())
    print("vectors[0, 0]:", vectors[0, 0].tolist())
    print()
    print("padding row (id=0) is all zeros:", vectors[0, 3].tolist())


if __name__ == "__main__":
    main()
