"""
Hands-on: Hugging Face pipeline for sentiment (pretrained DistilBERT on SST-2).

Install once:
  pip install -r 08_llm_practice/requirements.txt
"""

from __future__ import annotations

import argparse

import torch
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="HF sentiment-analysis pipeline")
    parser.add_argument(
        "--text",
        action="append",
        help="review text (repeat flag for multiple); default runs demo sentences",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    texts = args.text or [
        "I love this movie, it is fantastic.",
        "This film was boring and disappointing.",
        "Not great, not terrible, just okay.",
    ]

    print("model: distilbert-base-uncased-finetuned-sst-2-english")
    print("-" * 60)
    for text in texts:
        result = clf(text)[0]
        print(f"text: {text}")
        print(f"  -> {result['label']}  score={result['score']:.4f}")
        print()


if __name__ == "__main__":
    main()
