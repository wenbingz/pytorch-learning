"""
Export a PyTorch model to ONNX and inspect the file.

Install once:
    pip install -r 15_deployment/requirements.txt

Run:
    python3 15_deployment/02_onnx_export.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class MNISTMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    try:
        import onnx
    except ImportError:
        print("missing deps; run: pip install -r 15_deployment/requirements.txt")
        return

    out_dir = Path("runs/deployment")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "mnist_mlp.onnx"
    weights_path = out_dir / "mnist_mlp.pt"

    torch.manual_seed(0)
    model = MNISTMLP().eval()
    x = torch.randn(1, 1, 28, 28)
    torch.save(model.state_dict(), weights_path)

    torch.onnx.export(
        model,
        x,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)

    size_kb = onnx_path.stat().st_size / 1024
    print(f"saved: {onnx_path} ({size_kb:.1f} KB)")
    print(f"saved: {weights_path} (for fair compare in script 03)")
    print(f"onnx opset: {proto.opset_import[0].version}")
    print("input:", proto.graph.input[0].name)
    print("output:", proto.graph.output[0].name)
    print()
    print("ONNX is a portable graph format; run it with onnxruntime / TensorRT / mobile runtimes")


if __name__ == "__main__":
    main()
