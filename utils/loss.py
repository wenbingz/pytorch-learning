from __future__ import annotations

import torch
import torch.nn.functional as F

_mps_cpu_loss_announced = False


def _labels_on_cpu(targets: torch.Tensor) -> torch.Tensor:
    if targets.device.type == "cpu":
        return targets.long()
    return targets.detach().cpu().long()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss.

    On Apple MPS, native CE / log_softmax often returns NaN. We compute CE on CPU
    logits via `.to("cpu")`, which keeps autograd connected back to MPS weights.
    Labels always stay on CPU to avoid MPS int64 corruption.
    """
    global _mps_cpu_loss_announced

    logits = logits.float()
    targets_cpu = _labels_on_cpu(targets)

    if logits.device.type == "mps":
        if not _mps_cpu_loss_announced:
            print("note: using CPU cross-entropy for MPS (stability workaround)")
            _mps_cpu_loss_announced = True
        return F.cross_entropy(logits.to("cpu"), targets_cpu)

    return F.cross_entropy(logits, targets_cpu)
