from __future__ import annotations

import re

import torch
from torch import nn
from torch.utils.data import Dataset

PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_toy_reviews() -> tuple[list[str], list[int]]:
    positive = [
        "i love this movie it is great",
        "what a wonderful film absolutely fantastic",
        "great acting and a lovely story",
        "this is amazing and heartwarming",
        "i really enjoyed it so much fun",
        "excellent movie highly recommend it",
        "beautiful visuals and great music",
        "so good i watched it twice",
        "a delightful experience from start to finish",
        "pure joy entertaining and smart",
    ] * 20

    negative = [
        "i hate this movie it is awful",
        "what a terrible film completely boring",
        "bad acting and a weak story",
        "this is horrible and disappointing",
        "i really disliked it waste of time",
        "poor movie do not recommend it",
        "ugly visuals and annoying music",
        "so bad i stopped watching early",
        "a painful experience from start to finish",
        "pure trash dull and pointless",
    ] * 20

    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    return texts, labels


class Vocab:
    def __init__(self, itos: list[str] | None = None, tokens: list[str] | None = None) -> None:
        if itos is not None:
            self.itos = itos
        else:
            words = sorted(set(tokens or []))
            self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: list[str]) -> list[int]:
        unk_id = 1  # self.itos[1] is UNK
        return [self.stoi.get(w, unk_id) for w in words]

    def __len__(self) -> int:
        return len(self.itos)


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int) -> None:
        self.samples = []
        for text, label in zip(texts, labels):
            ids = vocab.encode(tokenize(text))
            ids = ids[:max_len]
            ids = ids + [0] * (max_len - len(ids))
            self.samples.append((torch.tensor(ids, dtype=torch.long), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.samples[idx]
        return x, y


class TextLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, num_classes: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n.squeeze(0))


def encode_text(text: str, vocab: Vocab, max_len: int) -> torch.Tensor:
    ids = vocab.encode(tokenize(text))
    ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)
