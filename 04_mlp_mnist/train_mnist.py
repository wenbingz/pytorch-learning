from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import resolve_device
from utils.loss import cross_entropy_loss
from utils.loop import eval_epoch, train_epoch


@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 256
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0
    data_dir: str = "data/mnist"
    ckpt_path: str = "runs/mnist/ckpt.pt"


class MNISTMLP(nn.Module):
    def __init__(self, hidden: int, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def params_are_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def fmt(value: Optional[float], spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    return Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_dir = Path(cfg.data_dir)
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = make_loaders(cfg)
    model = MNISTMLP(hidden=cfg.hidden).to(device)
    loss_fn = cross_entropy_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 1
    best_val_acc = -1.0
    if Path(cfg.ckpt_path).exists():
        ckpt = load_checkpoint(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        print(f"resume from {cfg.ckpt_path} @ epoch {start_epoch}")
        if not params_are_finite(model):
            raise RuntimeError(
                f"checkpoint {cfg.ckpt_path} has non-finite weights; delete it and re-run"
            )

    print("device:", device)
    if device.type == "mps":
        print("note: MPS can be unstable for loss/backprop; use --device cpu if you see NaN loss")
    print("train size:", len(train_loader.dataset), "test size:", len(test_loader.dataset))

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            max_grad_norm=cfg.max_grad_norm,
        )
        tr_eval = eval_epoch(model=model, loader=train_loader, loss_fn=loss_fn, device=device)
        va = eval_epoch(model=model, loader=test_loader, loss_fn=loss_fn, device=device)

        if not params_are_finite(model):
            raise RuntimeError(f"non-finite weights after epoch {epoch}; delete {cfg.ckpt_path} and retry on CPU")

        clip = "n/a" if tr.grad_clip_ratio is None else f"{tr.grad_clip_ratio:.1%}"
        print(
            f"epoch={epoch:02d} "
            f"train loss={fmt(tr.loss)} acc1={fmt(tr.acc1)} "
            f"train_eval acc1={fmt(tr_eval.acc1)} | "
            f"test loss={fmt(va.loss)} acc1={fmt(va.acc1)} | "
            f"grad_norm avg={fmt(tr.grad_norm_avg)} max={fmt(tr.grad_norm_max)} clipped={clip}"
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

    print("best test acc1:", round(best_val_acc, 4))
    print("checkpoint saved at:", cfg.ckpt_path)


if __name__ == "__main__":
    main()
