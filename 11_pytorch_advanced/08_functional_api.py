"""
nn.functional vs nn.Module: stateless ops vs layers with parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    x = torch.randn(4, 8)

    # functional: no persistent params on the function itself
    y1 = F.relu(x)

    # module: owns parameters/buffers
    linear = nn.Linear(8, 4)
    y2 = linear(x)

    print("F.relu out shape:", tuple(y1.shape))
    print("Linear out shape:", tuple(y2.shape))
    print("Linear weight shape:", tuple(linear.weight.shape))

    # same math, different style
    w = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    y3 = F.linear(x, w, b)
    print("F.linear and nn.Linear are equivalent when weights are explicit")


if __name__ == "__main__":
    main()
