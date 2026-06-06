"""
Helpers for torch.distributed / DDP scripts.

Launch multi-process demos with torchrun, e.g.:

    torchrun --standalone --nproc_per_node=2 13_distributed/01_ddp_basics.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DDPContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


def setup_ddp() -> DDPContext:
    """Initialize the process group when launched via torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DDPContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=torch.device("cpu"),
            enabled=False,
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return DDPContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        enabled=True,
    )


def cleanup_ddp(ctx: DDPContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main(ctx: DDPContext) -> bool:
    return ctx.rank == 0


def barrier(ctx: DDPContext) -> None:
    if ctx.enabled:
        dist.barrier()


def print0(ctx: DDPContext, *args, **kwargs) -> None:
    """Print only from rank 0 to avoid duplicated logs."""
    if is_main(ctx):
        print(*args, **kwargs)
