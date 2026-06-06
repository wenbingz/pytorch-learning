"""
Launch a DDP script with torch.multiprocessing.spawn (Mac-friendly fallback).

Usage:
    python3 13_distributed/run_ddp.py --nproc 2 13_distributed/01_ddp_basics.py
    python3 13_distributed/run_ddp.py --nproc 2 13_distributed/04_train_mnist_ddp.py --epochs 3
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _worker(rank: int, world_size: int, script: str, script_argv: list[str]) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    old_argv = sys.argv
    sys.argv = [script, *script_argv]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn multiple processes for a DDP demo script")
    parser.add_argument("--nproc", type=int, default=2, help="number of processes")
    parser.add_argument("script", help="path to the DDP script")
    args, script_argv = parser.parse_known_args()

    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(args.nproc, str(Path(args.script).resolve()), script_argv),
        nprocs=args.nproc,
        join=True,
    )


if __name__ == "__main__":
    main()
