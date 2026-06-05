from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from text_model import TextLSTMClassifier, Vocab, encode_text

DEFAULT_CKPT = "runs/text/ckpt.pt"

DEMO_SENTENCES = [
    "i love this movie it is great",
    "this is horrible and disappointing",
    "great acting but the ending was weak",
    "not bad actually pretty fun",
]


def predict_one(
    model: TextLSTMClassifier,
    vocab: Vocab,
    text: str,
    max_len: int,
    device: torch.device,
) -> tuple[str, float, float]:
    x = encode_text(text, vocab, max_len).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    p_neg = float(probs[0].item())
    p_pos = float(probs[1].item())
    label = "positive" if p_pos >= p_neg else "negative"
    return label, p_neg, p_pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment inference with your trained LSTM")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("text", nargs="*", help="optional review text")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}")
        print("train first: python3 07_nlp_intro/train_toy_sentiment.py")
        sys.exit(1)

    device = resolve_device(args.device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    max_len = int(cfg.get("max_len", 12))
    embed_dim = int(cfg.get("embed_dim", 64))
    hidden = int(cfg.get("hidden", 64))

    vocab = Vocab(itos=list(ckpt["vocab"]))
    model = TextLSTMClassifier(len(vocab), embed_dim, hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("device:", device)
    print("checkpoint:", ckpt_path)
    print("best test acc from training:", ckpt.get("best_test_acc", "n/a"))
    print()

    texts = [" ".join(args.text)] if args.text else DEMO_SENTENCES
    for text in texts:
        label, p_neg, p_pos = predict_one(model, vocab, text, max_len, device)
        print(f"text: {text}")
        print(f"  -> {label}  (p_neg={p_neg:.3f}, p_pos={p_pos:.3f})")
        print()


if __name__ == "__main__":
    main()
