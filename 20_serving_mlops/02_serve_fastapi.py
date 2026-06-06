"""
Minimal FastAPI + ONNXRuntime inference server.

Prerequisite:
    pip install -r 20_serving_mlops/requirements.txt
    pip install -r 15_deployment/requirements.txt
    python3 20_serving_mlops/01_release_gate.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx --min-acc 0.97

Run:
    python3 20_serving_mlops/02_serve_fastapi.py --manifest runs/releases/<your_manifest>.json
    # or directly:
    python3 20_serving_mlops/02_serve_fastapi.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError:
    FastAPI = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX model HTTP server")
    p.add_argument("--task", choices=["mnist", "cifar"], default="mnist")
    p.add_argument("--onnx", default=None)
    p.add_argument("--manifest", default=None, help="json from 01_release_gate.py")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        return {
            "task": manifest["task"],
            "onnx_path": manifest["onnx_path"],
            "version": manifest.get("version", "unknown"),
            "test_acc": manifest.get("test_acc"),
        }
    if not args.onnx:
        raise ValueError("pass --manifest or --onnx")
    return {"task": args.task, "onnx_path": args.onnx, "version": "dev", "test_acc": None}


def build_app(settings: dict[str, Any]) -> "FastAPI":
    import onnxruntime as ort

    session = ort.InferenceSession(settings["onnx_path"], providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    app = FastAPI(title="pytorch-learning-serve", version=settings["version"])

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "task": settings["task"],
            "model_version": settings["version"],
            "onnx": settings["onnx_path"],
            "test_acc": settings["test_acc"],
        }

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise HTTPException(status_code=400, detail="JSON body needs non-empty 'values' list")

        task = settings["task"]
        if task == "mnist":
            expected = 28 * 28
            shape = (1, 1, 28, 28)
        else:
            expected = 3 * 32 * 32
            shape = (1, 3, 32, 32)

        if len(values) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"expected {expected} floats for {task}, got {len(values)}",
            )

        x = np.array(values, dtype=np.float32).reshape(shape)
        logits = session.run(None, {input_name: x})[0][0]
        pred = int(logits.argmax())
        probs = torch_softmax(logits)
        return {
            "pred": pred,
            "probs": probs,
            "model_version": settings["version"],
        }

    return app


def torch_softmax(logits: np.ndarray) -> list[float]:
    x = logits - logits.max()
    e = np.exp(x)
    p = e / e.sum()
    return [float(v) for v in p]


def main() -> None:
    if FastAPI is None:
        print("pip install -r 20_serving_mlops/requirements.txt")
        return

    args = parse_args()
    settings = load_settings(args)
    app = build_app(settings)
    print(f"serving {settings['task']} from {settings['onnx_path']}")
    print(f"open http://{args.host}:{args.port}/docs for interactive API")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
