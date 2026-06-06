"""
Minimal DDP training loop on synthetic data.

    torchrun --standalone --nproc_per_node=2 13_distributed/03_ddp_train_toy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ddp import cleanup_ddp, is_main, print0, setup_ddp


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    ctx = setup_ddp()
    if not ctx.enabled:
        print("launch with torchrun, e.g.:")
        print("  torchrun --standalone --nproc_per_node=2", __file__)
        return

    torch.manual_seed(0)
    x = torch.randn(128, 16)
    y = torch.randint(0, 2, (128,))
    ds = TensorDataset(x, y)

    sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=0)

    model = TinyClassifier().to(ctx.device)
    model = DDP(model, device_ids=None if ctx.device.type == "cpu" else [ctx.local_rank])
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 4):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)

            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_t = torch.tensor([avg_loss], device=ctx.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= ctx.world_size
        print0(ctx, f"epoch={epoch} avg_loss={loss_t.item():.4f} (mean across ranks)")

    print0(ctx, "DDP wraps the model and all-reduces gradients each backward()")
    cleanup_ddp(ctx)


if __name__ == "__main__":
    main()
