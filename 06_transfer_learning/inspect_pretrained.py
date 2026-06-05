import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

sys.path.append(str(Path(__file__).resolve().parents[1]))


def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            continue
        param.requires_grad = trainable


def count_params(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def main() -> None:
    model = build_resnet18(num_classes=10)

    print("model: resnet18 (ImageNet pretrained)")
    print("head replaced: fc -> Linear(512, 10) for CIFAR-10")
    print()

    set_backbone_trainable(model, trainable=False)
    total, trainable = count_params(model)
    print("[backbone frozen]")
    print("total params:", total)
    print("trainable params:", trainable)
    print("frozen params:", total - trainable)
    print()

    set_backbone_trainable(model, trainable=True)
    total, trainable = count_params(model)
    print("[backbone unfrozen]")
    print("trainable params:", trainable)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print()
    print("dummy input shape:", tuple(x.shape))
    print("logits shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
