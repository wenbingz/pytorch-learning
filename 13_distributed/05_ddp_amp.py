"""
DDP + AMP (mixed precision): combine multi-GPU data parallel with fp16/bf16 forward.

CUDA (real AMP):
    torchrun --standalone --nproc_per_node=2 13_distributed/05_ddp_amp.py --epochs 3

Mac / CPU (API demo, fp32 fallback):
    python3 13_distributed/run_ddp.py --nproc 2 13_distributed/05_ddp_amp.py --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ddp import cleanup_ddp, print0, setup_ddp


class TinyClassifier(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP + AMP training demo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"])
    return parser.parse_args()


def amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def train_step(
    *,
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: nn.Module,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device_type: str,
) -> float:
    optim.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
        loss = loss_fn(model(xb), yb)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

    return loss.detach().item()


def main() -> None:
    ctx = setup_ddp()
    if not ctx.enabled:
        print("launch with run_ddp.py or torchrun, e.g.:")
        print("  python3 13_distributed/run_ddp.py --nproc 2", __file__)
        return

    args = parse_args()
    use_amp = ctx.device.type == "cuda"
    dtype = amp_dtype(args.amp_dtype)

    torch.manual_seed(0 + ctx.rank)
    x = torch.randn(512, 64)
    y = torch.randint(0, 2, (512,))
    ds = TensorDataset(x, y)

    sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=ctx.device.type == "cuda",
    )

    model = TinyClassifier().to(ctx.device)
    model = DDP(
        model,
        device_ids=None if ctx.device.type == "cpu" else [ctx.local_rank],
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print0(ctx, f"DDP+AMP | world_size={ctx.world_size} device={ctx.device}")
    print0(
        ctx,
        f"amp={'on' if use_amp else 'off (fp32 fallback)'} "
        f"dtype={args.amp_dtype if use_amp else 'float32'} "
        f"per-rank batch={args.batch_size}",
    )

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(ctx.device, non_blocking=True)
            yb = yb.to(ctx.device, non_blocking=True)
            total_loss += train_step(
                model=model,
                xb=xb,
                yb=yb,
                optim=optim,
                scaler=scaler,
                loss_fn=loss_fn,
                use_amp=use_amp,
                amp_dtype=dtype,
                device_type=ctx.device.type,
            )
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_t = torch.tensor([avg_loss], device=ctx.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= ctx.world_size
        print0(ctx, f"epoch={epoch} avg_loss={loss_t.item():.4f}")

    print0(ctx, "pattern: autocast forward -> scaler.backward/step -> DDP all-reduces grads")
    print0(ctx, "CUDA: use bf16 on Ampere+ if float16 is unstable")
    cleanup_ddp(ctx)


if __name__ == "__main__":
    main()
