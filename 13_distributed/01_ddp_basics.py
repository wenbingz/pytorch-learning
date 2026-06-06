"""
DDP basics: rank, world_size, and all_reduce.

Single process (for comparison):
    python3 13_distributed/01_ddp_basics.py

Multi-process (Mac CPU uses gloo; Linux+CUDA uses nccl):
    torchrun --standalone --nproc_per_node=2 13_distributed/01_ddp_basics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ddp import cleanup_ddp, print0, setup_ddp


def main() -> None:
    ctx = setup_ddp()

    if ctx.enabled:
        print0(ctx, f"backend: {dist.get_backend()}")
        print0(ctx, f"world_size: {ctx.world_size}")
    else:
        print("not launched with torchrun; running as a single process")
        print("try: torchrun --standalone --nproc_per_node=2", __file__)

    print(f"[rank {ctx.rank}] hello from device={ctx.device}")

    t = torch.tensor([ctx.rank + 1.0], device=ctx.device)
    if ctx.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {ctx.rank}] all_reduce sum -> {t.item():.1f}")

    print0(ctx, "idea: each GPU/process owns a model replica; grads are averaged across ranks")
    cleanup_ddp(ctx)


if __name__ == "__main__":
    main()
