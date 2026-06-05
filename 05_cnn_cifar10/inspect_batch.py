import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def main() -> None:
    data_dir = Path("data/cifar10")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    images, labels = next(iter(loader))
    print("images shape:", tuple(images.shape))  # (batch, channels, height, width)
    print("labels shape:", tuple(labels.shape))
    print("labels:", labels.tolist())
    print("pixel min/max:", round(images.min().item(), 4), round(images.max().item(), 4))
    print("class names sample:", train_ds.classes[:3], "...")


if __name__ == "__main__":
    main()
