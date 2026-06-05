from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter, accuracy_top1


@dataclass
class EpochResult:
    loss: float
    acc1: float
    grad_norm_avg: float | None = None
    grad_norm_max: float | None = None
    grad_clip_ratio: float | None = None


def _move_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.long()
    # Integer labels on MPS can corrupt; keep them on CPU when using Apple GPU.
    if device.type != "mps":
        y = y.to(device)
    return x, y


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float | None = None,
) -> EpochResult:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    grad_norm_max = 0.0
    clip_count = 0
    batch_count = 0
    skipped_nonfinite = 0

    for xb, yb in loader:
        xb, yb = _move_to_device((xb, yb), device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            norm_value = float(total_norm)
            if math.isfinite(norm_value):
                grad_norm_meter.update(norm_value, n=1)
                grad_norm_max = max(grad_norm_max, norm_value)
                batch_count += 1
                if norm_value > max_grad_norm:
                    clip_count += 1
        optimizer.step()

        bs = int(yb.shape[0])
        loss_meter.update(loss.item(), n=bs)
        with torch.no_grad():
            acc_logits = model(xb)
        acc_meter.update(accuracy_top1(acc_logits, yb), n=bs)

    grad_stats: dict[str, float | None] = {
        "grad_norm_avg": None,
        "grad_norm_max": None,
        "grad_clip_ratio": None,
    }
    if max_grad_norm is not None and batch_count > 0:
        grad_stats = {
            "grad_norm_avg": grad_norm_meter.avg,
            "grad_norm_max": grad_norm_max,
            "grad_clip_ratio": clip_count / batch_count,
        }

    if skipped_nonfinite:
        print(f"warning: skipped {skipped_nonfinite} non-finite loss batches")

    return EpochResult(
        loss=loss_meter.avg,
        acc1=acc_meter.avg,
        **grad_stats,
    )


@torch.no_grad()
def eval_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> EpochResult:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for xb, yb in loader:
        xb, yb = _move_to_device((xb, yb), device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        bs = int(yb.shape[0])
        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(accuracy_top1(logits, yb), n=bs)

    return EpochResult(loss=loss_meter.avg, acc1=acc_meter.avg)

