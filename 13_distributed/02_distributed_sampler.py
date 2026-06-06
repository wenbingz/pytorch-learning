"""
DistributedSampler: each rank sees a different shard of the dataset.

    torchrun --standalone --nproc_per_node=2 13_distributed/02_distributed_sampler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ddp import cleanup_ddp, print0, setup_ddp


def main() -> None:
    ctx = setup_ddp()
    if not ctx.enabled:
        print("launch with torchrun, e.g.:")
        print("  torchrun --standalone --nproc_per_node=2", __file__)
        return

    x = torch.arange(10).float().unsqueeze(1)
    y = torch.arange(10)
    ds = TensorDataset(x, y)

    sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=False)
    loader = DataLoader(ds, batch_size=2, sampler=sampler)

    seen: list[int] = []
    for xb, yb in loader:
        seen.extend(yb.tolist())

    print(f"[rank {ctx.rank}] indices this epoch: {seen}")
    print0(ctx, "each rank gets disjoint samples; together they cover the full dataset")
    print0(ctx, "call sampler.set_epoch(epoch) each epoch when shuffle=True")

    cleanup_ddp(ctx)


if __name__ == "__main__":
    main()
