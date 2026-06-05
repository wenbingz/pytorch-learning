from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import resolve_device
from utils.loss import cross_entropy_loss
from utils.loop import eval_epoch, train_epoch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    epochs: int = 10
    freeze_epochs: int = 3
    batch_size: int = 64
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0
    max_train: int = 0
    data_dir: str = "data/cifar10"
    ckpt_path: str = "runs/finetune/ckpt.pt"


def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            continue
        param.requires_grad = trainable


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_are_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def fmt(value: Optional[float], spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on CIFAR-10")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=0, help="optional train subset size for faster CPU runs")
    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        batch_size=args.batch_size,
        max_train=args.max_train,
        device=args.device,
    )


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    data_dir = Path(cfg.data_dir)
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_transform)

    if cfg.max_train > 0:
        n = min(cfg.max_train, len(train_ds))
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(cfg.seed))[:n].tolist()
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = make_loaders(cfg)
    model = build_resnet18(num_classes=10).to(device)
    loss_fn = cross_entropy_loss

    start_epoch = 1
    best_test_acc = -1.0
    phase = "head"
    ckpt = None

    if Path(cfg.ckpt_path).exists():
        ckpt = load_checkpoint(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_test_acc = float(ckpt.get("best_test_acc", -1.0))
        phase = str(ckpt.get("phase", "head"))
        print(f"resume from {cfg.ckpt_path} @ epoch {start_epoch} (phase={phase})")
        if not params_are_finite(model):
            raise RuntimeError(f"checkpoint {cfg.ckpt_path} has non-finite weights; delete it and re-run")

    print("device:", device)
    print("model: resnet18 (ImageNet pretrained) -> CIFAR-10 head")
    print("input resize: 32x32 -> 224x224 (ImageNet-compatible)")
    print("train size:", len(train_loader.dataset), "test size:", len(test_loader.dataset))
    print(f"schedule: epochs 1-{cfg.freeze_epochs} train head only, then fine-tune all layers")

    if start_epoch > cfg.freeze_epochs or phase == "full":
        set_backbone_trainable(model, trainable=True)
        phase = "full"
        lr = cfg.finetune_lr
    else:
        set_backbone_trainable(model, trainable=False)
        for param in model.fc.parameters():
            param.requires_grad = True
        phase = "head"
        lr = cfg.head_lr

    optimizer = make_optimizer(model, lr=lr)
    if ckpt is not None and "optim" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optim"])
        except ValueError:
            print("warning: optimizer state mismatch after phase change; using fresh optimizer")

    for epoch in range(start_epoch, cfg.epochs + 1):
        if epoch <= cfg.freeze_epochs:
            set_backbone_trainable(model, trainable=False)
            for param in model.fc.parameters():
                param.requires_grad = True
            if phase != "head":
                optimizer = make_optimizer(model, lr=cfg.head_lr)
                phase = "head"
        elif phase != "full":
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, lr=cfg.finetune_lr)
            phase = "full"
            print(f"==> epoch {epoch}: unfroze backbone, lr={cfg.finetune_lr}")

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
            f"epoch={epoch:02d} phase={phase} trainable={count_trainable(model)} "
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
                    "phase": phase,
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
