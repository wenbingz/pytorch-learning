"""Plot training loss from runs/transformer/loss.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_loss(csv_path: Path, out_path: Path, *, window: int = 50) -> None:
    steps: list[int] = []
    losses: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    if not steps:
        raise ValueError(f"no rows in {csv_path}")

    ma = moving_average(losses, window)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(steps, losses, alpha=0.25, linewidth=0.8, color="#4C78A8", label="train loss (per step)")
    ax.plot(steps, ma, linewidth=2, color="#E45756", label=f"{window}-step moving avg")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Transformer training loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved {out_path} ({len(steps)} points)")
    print(f"start={losses[0]:.4f} end={losses[-1]:.4f} min={min(losses):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss.csv as PNG.")
    parser.add_argument("--csv", type=Path, default=Path("runs/transformer/loss.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/transformer/loss.png"))
    parser.add_argument("--window", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_loss(args.csv, args.out, window=args.window)
