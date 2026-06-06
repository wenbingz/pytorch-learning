"""
MLOps map: what serving + release gate + experiment log cover.

Run:
    python3 20_serving_mlops/04_mlops_map.py
"""


def main() -> None:
    print("minimal MLOps loop in this repo\n")
    flow = [
        "train  ->  ckpt (16_production/04)",
        "export ->  onnx (16_production/05)",
        "gate   ->  01_release_gate.py (acc threshold + manifest.json)",
        "log    ->  runs/experiments.jsonl",
        "serve  ->  02_serve_fastapi.py (load manifest version)",
        "client ->  03_client_predict.py",
    ]
    for step in flow:
        print(" ", step)

    print("\nwhat is new vs previous lessons:")
    items = [
        ("ONNX export", "artifact file on disk"),
        ("release gate", "policy: acc < threshold => block deploy"),
        ("manifest", "model version metadata for ops/rollback"),
        ("jsonl log", "audit trail of approved releases"),
        ("FastAPI", "network boundary: other apps call /predict"),
    ]
    for name, desc in items:
        print(f"  {name:<14} {desc}")

    print("\nnot covered here (real company stack):")
    for item in [
        "Kubernetes / autoscaling",
        "Prometheus metrics + alerts",
        "A/B testing traffic split",
        "data drift monitoring",
        "full MLflow / W&B model registry UI",
    ]:
        print(f"  - {item}")

    print("\nquick start (mnist):")
    print("  python3 20_serving_mlops/01_release_gate.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx --min-acc 0.97")
    print("  python3 20_serving_mlops/02_serve_fastapi.py --manifest runs/releases/<file>.json")


if __name__ == "__main__":
    main()
