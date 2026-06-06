from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def clone_model(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema
