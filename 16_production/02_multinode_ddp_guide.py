"""
Multi-node DDP launch guide (read + copy commands; no cluster required).

Run:
    python3 16_production/02_multinode_ddp_guide.py
"""

from __future__ import annotations


def main() -> None:
    print("multi-node DDP = same DDP code, different launch environment\n")

    print("key env vars (torchrun sets most of these):")
    vars_ = [
        ("MASTER_ADDR", "IP/hostname of rank-0 node"),
        ("MASTER_PORT", "free TCP port on master, e.g. 29500"),
        ("WORLD_SIZE", "total processes across ALL nodes"),
        ("RANK", "global rank 0..WORLD_SIZE-1"),
        ("LOCAL_RANK", "GPU id on this node"),
        ("NODE_RANK", "which machine (0,1,...) when using elastic launch"),
    ]
    for name, desc in vars_:
        print(f"  {name:<12} {desc}")

    print("\nexample: 2 nodes x 2 GPUs = 4 processes")
    print("  global batch = per_gpu_batch x 4")
    print("  each node still uses DistributedSampler over the full dataset")

    print("\nnode 0 (master, 10.0.0.1):")
    print(
        """  torchrun \\
    --nnodes=2 --nproc_per_node=2 \\
    --node_rank=0 \\
    --master_addr=10.0.0.1 --master_port=29500 \\
    13_distributed/04_train_mnist_ddp.py --epochs 10"""
    )

    print("\nnode 1:")
    print(
        """  torchrun \\
    --nnodes=2 --nproc_per_node=2 \\
    --node_rank=1 \\
    --master_addr=10.0.0.1 --master_port=29500 \\
    13_distributed/04_train_mnist_ddp.py --epochs 10"""
    )

    print("\nchecklist vs single-machine DDP (lesson 13):")
    checklist = [
        "same code: DDP(model) + DistributedSampler + sampler.set_epoch()",
        "data: each node reads dataset locally or via shared storage",
        "only rank 0 prints / saves checkpoint",
        "backend on GPU clusters: nccl (not gloo)",
        "firewall: open MASTER_PORT between nodes",
    ]
    for item in checklist:
        print(f"  - {item}")

    print("\non your MacBook:")
    print("  use 16_production/01 for single-machine production loop")
    print("  use 13_distributed/run_ddp.py for local 2-process DDP demo")


if __name__ == "__main__":
    main()
