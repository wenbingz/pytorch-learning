"""
Hands-on LoRA fine-tuning on DistilBERT using in-repo toy review data.

Install:
  pip install -r 10_lora/requirements.txt
  pip install -r 08_llm_practice/requirements.txt
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "07_nlp_intro"))

from text_model import build_toy_reviews


@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "runs/lora_sentiment"
    max_len: int = 64
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-4
    lora_r: int = 8
    seed: int = 0


def make_dataset(tokenizer, max_len: int) -> tuple[Dataset, Dataset]:
    texts, labels = build_toy_reviews()
    ds = Dataset.from_dict({"text": texts, "label": labels})

    split = ds.train_test_split(test_size=0.2, seed=0)
    train_ds, test_ds = split["train"], split["test"]

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_ds, test_ds


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, test_ds = make_dataset(tokenizer, cfg.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        use_cpu=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("train size:", len(train_ds), "test size:", len(test_ds))
    trainer.train()
    metrics = trainer.evaluate()
    print("eval:", metrics)

    save_dir = Path(cfg.output_dir) / "lora_adapter"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("saved LoRA adapter to:", save_dir)


if __name__ == "__main__":
    main()
