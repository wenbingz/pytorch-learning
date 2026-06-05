"""
Compare your toy LSTM checkpoint vs Hugging Face DistilBERT on the same sentences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "07_nlp_intro"))

from text_model import TextLSTMClassifier, Vocab, encode_text
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device

DEFAULT_SENTENCES = [
    "i love this movie it is great",
    "this is horrible and disappointing",
    "great acting but the ending was weak",
    "not bad actually pretty fun",
]


def predict_lstm(text: str, ckpt_path: Path, device: torch.device) -> tuple[str, float]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    vocab = Vocab(itos=list(ckpt["vocab"]))
    model = TextLSTMClassifier(
        len(vocab),
        int(cfg.get("embed_dim", 64)),
        int(cfg.get("hidden", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = encode_text(text, vocab, int(cfg.get("max_len", 12))).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    p_pos = float(probs[1].item())
    label = "positive" if p_pos >= 0.5 else "negative"
    return label, p_pos


def predict_hf(text: str) -> tuple[str, float]:
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )
    out = clf(text)[0]
    label = out["label"].lower()
    score = float(out["score"])
    p_pos = score if label == "positive" else 1.0 - score
    return label, p_pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/text/ckpt.pt")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"missing checkpoint: {ckpt_path}")
        print("run: python3 07_nlp_intro/train_toy_sentiment.py")
        sys.exit(1)

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("install HF deps: pip install -r 08_llm_practice/requirements.txt")
        sys.exit(1)

    device = resolve_device("cpu")
    texts = args.text or DEFAULT_SENTENCES

    print("Loading HF model (first run downloads weights)...")
    print("-" * 72)
    for text in texts:
        lstm_label, lstm_ppos = predict_lstm(text, ckpt_path, device)
        hf_label, hf_ppos = predict_hf(text)
        agree = lstm_label == hf_label
        print(f"text: {text}")
        print(f"  LSTM: {lstm_label:8s}  p_pos={lstm_ppos:.3f}")
        print(f"  HF  : {hf_label:8s}  p_pos={hf_ppos:.3f}")
        print(f"  agree: {agree}")
        print()


if __name__ == "__main__":
    main()
