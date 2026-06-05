"""
Inspect how many parameters are trainable after attaching LoRA to DistilBERT.
"""

from __future__ import annotations

import sys
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification


def count_params(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def main() -> None:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    t0, total0 = count_params(model)
    print("base model:", model_name)
    print("trainable:", t0, "total:", total0)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print()
    print("Compare with Lesson 6: freeze_epochs trains head only first.")
    print("LoRA trains small adapters inside attention while backbone stays frozen.")


if __name__ == "__main__":
    main()
