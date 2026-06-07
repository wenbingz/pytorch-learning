from __future__ import annotations

import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

BPE_PATH = "data/bpe_tokenizer.json"


class WordTokenizer:
    def __init__(self, corpus_path: str, reload: bool = False) -> None:
        if reload and os.path.isfile(BPE_PATH):
            self.tokenizer = Tokenizer.from_file(BPE_PATH)
            return

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=8000,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_frequency=2,
        )
        tokenizer.train([corpus_path], trainer)
        os.makedirs(os.path.dirname(BPE_PATH) or ".", exist_ok=True)
        tokenizer.save(BPE_PATH)
        self.tokenizer = tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
