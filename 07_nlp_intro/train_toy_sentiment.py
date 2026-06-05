from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_model import ReviewDataset, TextLSTMClassifier, Vocab, build_toy_reviews, tokenize
from utils.checkpoint import save_checkpoint
from utils.device import resolve_device
from utils.loss import cross_entropy_loss
from utils.loop import eval_epoch, train_epoch


@dataclass
class Config:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    embed_dim: int = 64
    hidden: int = 64
    max_len: int = 12
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0
    ckpt_path: str = "runs/text/ckpt.pt"


def fmt(value: Optional[float], spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Toy text sentiment classifier")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    return Config(epochs=args.epochs, device=args.device)


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    texts, labels = build_toy_reviews()
    all_tokens = [tok for text in texts for tok in tokenize(text)]
    vocab = Vocab(tokens=all_tokens)

    n = len(texts)
    split = int(0.8 * n)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:split]
    test_idx = perm[split:]

    train_ds = ReviewDataset([texts[i] for i in train_idx], [labels[i] for i in train_idx], vocab, cfg.max_len)
    test_ds = ReviewDataset([texts[i] for i in test_idx], [labels[i] for i in test_idx], vocab, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    model = TextLSTMClassifier(len(vocab), cfg.embed_dim, cfg.hidden).to(device)
    loss_fn = cross_entropy_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print("device:", device)
    print("vocab size:", len(vocab), "max_len:", cfg.max_len)
    print("train size:", len(train_ds), "test size:", len(test_ds))
    print("model: Embedding -> LSTM -> Linear (binary sentiment)")

    best_test_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            max_grad_norm=cfg.max_grad_norm,
        )
        te = eval_epoch(model=model, loader=test_loader, loss_fn=loss_fn, device=device)

        print(
            f"epoch={epoch:02d} "
            f"train loss={fmt(tr.loss)} acc1={fmt(tr.acc1)} | "
            f"test loss={fmt(te.loss)} acc1={fmt(te.acc1)}"
        )

        if te.acc1 > best_test_acc:
            best_test_acc = te.acc1
            save_checkpoint(
                cfg.ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab": vocab.itos,
                    "best_test_acc": best_test_acc,
                },
            )

    print("best test acc1:", round(best_test_acc, 4))
    print("checkpoint saved at:", cfg.ckpt_path)


if __name__ == "__main__":
    main()
