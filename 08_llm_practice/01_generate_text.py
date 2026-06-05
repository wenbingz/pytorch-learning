"""
Hands-on: run a small pretrained language model (DistilGPT-2) on CPU.

Install once:
  pip install -r 08_llm_practice/requirements.txt
"""

from __future__ import annotations

import argparse

import torch
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Text generation with DistilGPT-2")
    parser.add_argument("--prompt", default="PyTorch is a library for")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print("device for pipeline:", "cuda" if device == 0 else "cpu")

    generator = pipeline(
        task="text-generation",
        model="distilgpt2",
        device=device,
    )

    print("prompt:", args.prompt)
    print("-" * 60)
    out = generator(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.9)
    print(out[0]["generated_text"])
    print("-" * 60)
    print("done.")


if __name__ == "__main__":
    main()
