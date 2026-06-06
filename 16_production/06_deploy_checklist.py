"""
Route B deploy checklist — run before every ONNX release.

Run:
    python3 16_production/06_deploy_checklist.py
"""


def main() -> None:
    print("Route B: deploy checklist (train -> export -> ORT)\n")
    steps = [
        (
            "1. Train",
            [
                "MNIST: python3 16_production/01_train_mnist_production.py --epochs 10",
                "CIFAR: python3 16_production/04_train_cifar_production.py --epochs 20",
                "checkpoint must include model + ema (+ optim for resume)",
            ],
        ),
        (
            "2. Pick weights",
            [
                "resume training -> load ckpt['model'] + optimizer",
                "validate / deploy -> prefer ckpt['ema']",
            ],
        ),
        (
            "3. Export ONNX",
            [
                "MNIST: python3 16_production/03_deploy_production_mnist.py",
                "CIFAR: python3 16_production/05_deploy_cifar_checkpoint.py --weights ema",
                "run onnx.checker.check_model",
            ],
        ),
        (
            "4. Verify alignment",
            [
                "same test loader for PyTorch and ORT",
                "test acc should match (tiny float diff ok)",
                "max logit diff on sample batches ~ 1e-4 or less",
                "single-sample label/pred sanity check",
            ],
        ),
        (
            "5. Record release metadata",
            [
                "PyTorch version, opset, input names/shapes",
                "ckpt path, best_ema_acc, export date",
                "normalization constants (CIFAR mean/std in deploy script)",
            ],
        ),
        (
            "6. Optional compression",
            [
                "int8 dynamic quant on CPU: 07_int8_inference_compare.py",
                "re-measure acc after quant; ONNX int8 is a separate path",
                "GPU serving: TensorRT / Triton (outside this repo)",
            ],
        ),
    ]

    for title, items in steps:
        print(title)
        for item in items:
            print(f"  [ ] {item}")
        print()

    print("common mistakes:")
    for m in [
        "export with random weights while ORT loads old onnx (re-run export after train)",
        "forget test transforms (CIFAR must use same Normalize as training test)",
        "save model.state_dict() from DDP without .module",
    ]:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
