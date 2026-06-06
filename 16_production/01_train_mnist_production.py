"""
Production-style MNIST training: grad accumulation + StepLR + EMA.

Combines patterns from lesson 12 into one loop.

Run:
    python3 16_production/01_train_mnist_production.py --epochs 10
    python3 16_production/01_train_mnist_production.py --epochs 10 --accum-steps 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import resolve_device
from utils.ema import clone_model, update_ema
from utils.loss import cross_entropy_loss
from utils.loop import eval_epoch
from utils.metrics import accuracy_top1


@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 128
    accum_steps: int = 2
    lr: float = 1e-3
    hidden: int = 256
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    scheduler_step: int = 3
    scheduler_gamma: float = 0.5
    device: str = "cpu"
    seed: int = 0
    data_dir: str = "data/mnist"
    ckpt_path: str = "runs/mnist_production/ckpt.pt"


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


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MNIST with grad accum + scheduler + EMA")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=args.device,
    )


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


def train_epoch_with_accum(
    *,
    model: nn.Module,
    ema_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accum_steps: int,
    ema_decay: float,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.long()
        if device.type != "mps":
            yb = yb.to(device)

        logits = model(xb)
        loss = cross_entropy_loss(logits, yb) / accum_steps
        loss.backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_model, model, decay=ema_decay)

        bs = yb.size(0)
        total_loss += loss.item() * accum_steps * bs
        with torch.no_grad():
            total_acc += accuracy_top1(logits, yb) * bs
        n_samples += bs

    if len(loader) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema_model, model, decay=ema_decay)

    return total_loss / n_samples, total_acc / n_samples


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = make_loaders(cfg)
    model = MNISTMLP(hidden=cfg.hidden).to(device)
    ema_model = clone_model(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )

    start_epoch = 1
    best_ema_acc = -1.0
    ckpt_path = Path(cfg.ckpt_path)
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_ema_acc = float(ckpt.get("best_ema_acc", -1.0))
        print(f"resume from {ckpt_path} @ epoch {start_epoch}")

    effective_batch = cfg.batch_size * cfg.accum_steps
    print("device:", device)
    print(
        f"micro_batch={cfg.batch_size} accum_steps={cfg.accum_steps} "
        f"-> effective_batch={effective_batch}"
    )
    print(f"scheduler: StepLR every {cfg.scheduler_step} epochs, gamma={cfg.scheduler_gamma}")
    print(f"ema_decay={cfg.ema_decay}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_acc = train_epoch_with_accum(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            accum_steps=cfg.accum_steps,
            ema_decay=cfg.ema_decay,
            max_grad_norm=cfg.max_grad_norm,
        )
        test_model = eval_epoch(model=model, loader=test_loader, loss_fn=cross_entropy_loss, device=device)
        test_ema = eval_epoch(model=ema_model, loader=test_loader, loss_fn=cross_entropy_loss, device=device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch={epoch:02d} lr={lr:.4g} "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"test model acc={test_model.acc1:.4f} ema acc={test_ema.acc1:.4f}"
        )

        if test_ema.acc1 > best_ema_acc:
            best_ema_acc = test_ema.acc1
            save_checkpoint(
                cfg.ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": cfg.__dict__,
                    "best_ema_acc": best_ema_acc,
                },
            )

    print("best test ema acc:", round(best_ema_acc, 4))
    print("checkpoint saved at:", cfg.ckpt_path)
    print("tip: deploy/infer with ema weights for stabler predictions")


if __name__ == "__main__":
    main()
