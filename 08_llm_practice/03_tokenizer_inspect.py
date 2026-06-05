"""
Hands-on: see how a Hugging Face tokenizer turns text into token ids.
"""

from __future__ import annotations

from transformers import AutoTokenizer


def main() -> None:
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "PyTorch makes deep learning practical."
    encoded = tokenizer(text, return_tensors="pt")

    print("model:", model_name)
    print("text:", text)
    print("input_ids:", encoded["input_ids"].tolist())
    print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
    print("decoded:", tokenizer.decode(encoded["input_ids"][0]))


if __name__ == "__main__":
    main()
