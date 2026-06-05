import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_default_device
from utils.loop import eval_epoch, train_epoch


@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-2
    hidden: int = 64
    seed: int = 0
    ckpt_path: str = "runs/toy/ckpt.pt"


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_2d_blobs(n_per_class: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 3 Gaussian blobs in 2D. Easy classification to validate the training loop.
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.tensor([[-2.0, -2.0], [2.0, 0.0], [0.0, 2.0]])
    xs = []
    ys = []
    for k, c in enumerate(centers):
        xk = c + 0.8 * torch.randn(n_per_class, 2, generator=g)
        yk = torch.full((n_per_class,), k, dtype=torch.long)
        xs.append(xk)
        ys.append(yk)

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y


def main() -> None:
    cfg = Config()
    device = get_default_device()
    torch.manual_seed(cfg.seed)

    # Data
    x, y = make_2d_blobs(n_per_class=4000, device=device, seed=cfg.seed)
    ds = TensorDataset(x, y)

    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = MLP(in_dim=2, hidden=cfg.hidden, num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 1
    if Path(cfg.ckpt_path).exists():
        ckpt = load_checkpoint(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"resume from {cfg.ckpt_path} @ epoch {start_epoch}")

    print("device:", device)
    print("train size:", len(train_ds), "val size:", len(val_ds))

    best_val_acc = -1.0
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_epoch(model=model, loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        va = eval_epoch(model=model, loader=val_loader, loss_fn=loss_fn, device=device)

        print(
            f"epoch={epoch:02d} "
            f"train loss={tr.loss:.4f} acc1={tr.acc1:.4f} | "
            f"val loss={va.loss:.4f} acc1={va.acc1:.4f}"
        )

        if va.acc1 > best_val_acc:
            best_val_acc = va.acc1
            save_checkpoint(
                cfg.ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                },
            )

    print("best val acc1:", round(best_val_acc, 4))
    print("checkpoint saved at:", cfg.ckpt_path)


if __name__ == "__main__":
    main()

