"""
End-to-end deployment map: from training to serving.

Run:
    python3 15_deployment/04_deployment_pipeline.py
"""


def main() -> None:
    print("typical PyTorch deployment pipeline\n")
    steps = [
        ("1. train", "PyTorch script, save state_dict() or checkpoint"),
        ("2. eval", "fp32 / EMA weights, measure accuracy"),
        ("3. compress (optional)", "fp16 weights, int8 quant, LoRA merge"),
        ("4. export", "torch.export for PyTorch serving; ONNX for cross-platform"),
        ("5. serve", "TorchServe, Triton, onnxruntime, TensorRT, mobile"),
    ]
    for title, detail in steps:
        print(f"{title:<22} {detail}")

    print()
    print("lessons you already covered:")
    print("  L10 LoRA        -> smaller trainable adapter")
    print("  L12 EMA         -> stabler weights for step 2")
    print("  L13 DDP+AMP     -> scale training")
    print("  L14 fp16/int8   -> smaller/faster inference weights")
    print("  L15 export/ONNX -> portable graph for step 4-5")
    print()
    print("suggested commands:")
    print("  python3 15_deployment/01_torch_export_basic.py")
    print("  pip install -r 15_deployment/requirements.txt")
    print("  python3 15_deployment/02_onnx_export.py")
    print("  python3 15_deployment/03_pytorch_vs_onnx_inference.py")


if __name__ == "__main__":
    main()
