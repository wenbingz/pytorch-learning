"""
Train MNIST MLP with DDP (multi-GPU or multi-process CPU demo).

Single GPU / CPU sanity check:
    python3 13_distributed/04_train_mnist_ddp.py --epochs 1

Multi-process:
    torchrun --standalone --nproc_per_node=2 13_distributed/04_train_mnist_ddp.py --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ddp import cleanup_ddp, is_main, print0, setup_ddp
from utils.loss import cross_entropy_loss


@dataclass
class Config:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 256
    data_dir: str = "data/mnist"
    seed: int = 0


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


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    if y.device.type == "cpu" and preds.device.type != "cpu":
        preds = preds.cpu()
    return (preds == y).float().mean().item()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train MNIST with DDP")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--data-dir", default="data/mnist")
    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        data_dir=args.data_dir,
    )


def make_loaders(cfg: Config, ctx) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_dir = Path(cfg.data_dir)
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=is_main(ctx), transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=is_main(ctx), transform=transform)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
    )
    test_sampler = DistributedSampler(
        test_ds,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=ctx.device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=ctx.device.type == "cuda",
    )
    return train_loader, test_loader, train_sampler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.long().to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, yb)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.long().to(device)
        logits = model(xb)
        total_loss += cross_entropy_loss(logits, yb).item()
        total_acc += accuracy(logits, yb)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def reduce_mean(ctx, value: float) -> float:
    t = torch.tensor([value], device=ctx.device)
    if ctx.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ctx.world_size
    return t.item()


def main() -> None:
    ctx = setup_ddp()
    cfg = parse_args()

    torch.manual_seed(cfg.seed + ctx.rank)

    if ctx.enabled:
        train_loader, test_loader, train_sampler = make_loaders(cfg, ctx)
        model = MNISTMLP(cfg.hidden).to(ctx.device)
        ddp_model = DDP(
            model,
            device_ids=None if ctx.device.type == "cpu" else [ctx.local_rank],
        )
        optim = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.lr)

        print0(ctx, f"DDP MNIST | world_size={ctx.world_size} device={ctx.device}")
        print0(ctx, f"per-rank batch={cfg.batch_size} -> global batch={cfg.batch_size * ctx.world_size}")

        for epoch in range(1, cfg.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss, train_acc = train_one_epoch(ddp_model, train_loader, optim, ctx.device)
            test_loss, test_acc = eval_one_epoch(ddp_model, test_loader, ctx.device)

            metrics = torch.tensor(
                [train_loss, train_acc, test_loss, test_acc],
                device=ctx.device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= ctx.world_size

            print0(
                ctx,
                f"epoch={epoch} "
                f"train_loss={metrics[0].item():.4f} train_acc={metrics[1].item():.4f} "
                f"test_loss={metrics[2].item():.4f} test_acc={metrics[3].item():.4f}",
            )

        print0(ctx, "save model.module.state_dict() on rank 0 after training")
        cleanup_ddp(ctx)
        return

    print("DDP mode requires torchrun. Example:")
    print("  torchrun --standalone --nproc_per_node=2", __file__)
    print("Falling back: single-process CPU training for a quick sanity check.")

    # Single-process fallback (no torchrun): quick local demo on CPU.
    from torch.utils.data import DataLoader as SingleLoader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_dir = Path(cfg.data_dir)
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    train_loader = SingleLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = SingleLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = MNISTMLP(cfg.hidden)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    print("single-process CPU demo (use torchrun for real DDP)")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optim, torch.device("cpu"))
        test_loss, test_acc = eval_one_epoch(model, test_loader, torch.device("cpu"))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
