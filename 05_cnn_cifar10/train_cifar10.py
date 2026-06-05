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

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0
    data_dir: str = "data/cifar10"
    ckpt_path: str = "runs/cifar10/ckpt.pt"


class CIFAR10CNN(nn.Module):
    """
    Small CNN for 32x32 RGB images.

    Spatial sizes:
      32x32 -> pool -> 16x16 -> pool -> 8x8 -> pool -> 4x4
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def params_are_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def fmt(value: Optional[float], spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    return Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    data_dir = Path(cfg.data_dir)
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = make_loaders(cfg)
    model = CIFAR10CNN().to(device)
    loss_fn = cross_entropy_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 1
    best_test_acc = -1.0
    if Path(cfg.ckpt_path).exists():
        ckpt = load_checkpoint(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_test_acc = float(ckpt.get("best_test_acc", ckpt.get("best_val_acc", -1.0)))
        print(f"resume from {cfg.ckpt_path} @ epoch {start_epoch}")
        if not params_are_finite(model):
            raise RuntimeError(
                f"checkpoint {cfg.ckpt_path} has non-finite weights; delete it and re-run"
            )

    print("device:", device)
    if device.type == "mps":
        print("note: MPS can be unstable; use --device cpu if you see NaN loss")
    print("train size:", len(train_loader.dataset), "test size:", len(test_loader.dataset))
    print("augmentation: RandomCrop(pad=4), RandomHorizontalFlip")

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
        te = eval_epoch(model=model, loader=test_loader, loss_fn=loss_fn, device=device)

        if not params_are_finite(model):
            raise RuntimeError(f"non-finite weights after epoch {epoch}; delete {cfg.ckpt_path} and retry on CPU")

        clip = "n/a" if tr.grad_clip_ratio is None else f"{tr.grad_clip_ratio:.1%}"
        print(
            f"epoch={epoch:02d} "
            f"train loss={fmt(tr.loss)} acc1={fmt(tr.acc1)} "
            f"train_eval acc1={fmt(tr_eval.acc1)} | "
            f"test loss={fmt(te.loss)} acc1={fmt(te.acc1)} | "
            f"grad_norm avg={fmt(tr.grad_norm_avg)} max={fmt(tr.grad_norm_max)} clipped={clip}"
        )

        if te.acc1 > best_test_acc:
            best_test_acc = te.acc1
            save_checkpoint(
                cfg.ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "best_test_acc": best_test_acc,
                },
            )

    print("best test acc1:", round(best_test_acc, 4))
    print("checkpoint saved at:", cfg.ckpt_path)


if __name__ == "__main__":
    main()
