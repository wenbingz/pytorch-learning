"""
Module hooks: observe or modify activations/gradients without changing model code.
"""

from __future__ import annotations

import torch
from torch import nn


def main() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(2, 4, requires_grad=True)  # avoid full_backward_hook warning

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(name: str):
        def hook(_module, _inp, out):
            activations[name] = out.detach()

        return hook

    def backward_hook(name: str):
        def hook(_module, grad_input, grad_output):
            # grad_input[0] = gradient w.r.t. layer input (when input requires_grad)
            if grad_input[0] is not None:
                gradients[name] = grad_input[0].detach()

        return hook

    model[0].register_forward_hook(forward_hook("linear1"))
    model[0].register_full_backward_hook(backward_hook("linear1"))

    logits = model(x)
    loss = logits.pow(2).sum()
    loss.backward()

    print("forward capture linear1 out shape:", tuple(activations["linear1"].shape))
    print("backward capture grad w.r.t linear1 input shape:", tuple(gradients["linear1"].shape))
    print("tip: hooks are useful for debugging, visualization, and feature extraction")


if __name__ == "__main__":
    main()
