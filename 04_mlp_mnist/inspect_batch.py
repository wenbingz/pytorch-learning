import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    data_dir = Path("data/mnist")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    images, labels = next(iter(loader))
    print("images shape:", tuple(images.shape))  # (batch, channels, height, width)
    print("labels shape:", tuple(labels.shape))  # (batch,)
    print("labels:", labels.tolist())
    print("pixel min/max:", images.min().item(), images.max().item())
    print("flattened shape would be:", (images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))


if __name__ == "__main__":
    main()
