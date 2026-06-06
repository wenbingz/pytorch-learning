"""
Side-by-side summary: fp32 training, fp16 inference, int8 inference.

Run:
    python3 14_inference/04_quantization_tradeoffs.py
"""

from __future__ import annotations


def main() -> None:
    rows = [
        ("fp32", "training + baseline inference", "high", "1x", "default"),
        ("fp16", "inference (sometimes train with AMP)", "medium-high", "1.1-2x GPU", "Tensor Core matmul"),
        ("bf16", "inference / train on Ampere+", "high", "similar to fp16", "wider dynamic range"),
        ("int8 dynamic", "CPU inference (Linear/LSTM)", "medium", "1.5-4x CPU", "smaller weights"),
        ("int4/awq/gptq", "LLM deployment", "lower", "large LLM gains", "needs calibration"),
    ]

    print("precision tradeoffs (rules of thumb, hardware-dependent):\n")
    print(f"{'dtype':<12} {'typical use':<34} {'accuracy':<12} {'speed':<16} notes")
    print("-" * 90)
    for dtype, use, acc, speed, notes in rows:
        print(f"{dtype:<12} {use:<34} {acc:<12} {speed:<16} {notes}")

    print()
    print("lesson flow:")
    print("  01 -> weight bytes shrink fp32 > fp16 > int8")
    print("  02 -> fp16 inference API + CUDA speedups")
    print("  03 -> int8 dynamic quant on MNIST MLP")
    print()
    print("training AMP (lesson 12/13) != inference fp16/int8:")
    print("  train: fp32 master weights + fp16 compute + GradScaler")
    print("  infer: often store weights directly in fp16/int8 for bandwidth")


if __name__ == "__main__":
    main()
