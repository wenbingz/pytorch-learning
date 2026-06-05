"""
Load saved LoRA adapter and run inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="runs/lora_sentiment/lora_adapter")
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--text", default="i love this pytorch lesson")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        print(f"missing adapter: {adapter_dir}")
        print("train first: python3 10_lora/03_finetune_lora_sentiment.py")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    batch = tokenizer(args.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=1)[0]

    label = "positive" if probs[1] >= probs[0] else "negative"
    print("text:", args.text)
    print("pred:", label, "p_pos=", round(float(probs[1]), 4))


if __name__ == "__main__":
    main()
