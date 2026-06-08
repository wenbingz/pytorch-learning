from __future__ import annotations

import os
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BPE_PATH = "data/bpe_tokenizer.json"


class WordTokenizer:
    def __init__(self, corpus_path: list[str], reload: bool = False) -> None:
        # reload=True  → load data/bpe_tokenizer.json
        # reload=False → train BPE on corpus and save
        if reload:
            if not os.path.isfile(BPE_PATH):
                raise FileNotFoundError(
                    f"{BPE_PATH} not found; run --train --tokenizer_name word first"
                )
            self.tokenizer = Tokenizer.from_file(BPE_PATH)
            if self.tokenizer.decoder is None:
                self.tokenizer.decoder = decoders.ByteLevel()
            return

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=8000,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_frequency=2,
        )
        tokenizer.train(corpus_path, trainer)
        os.makedirs(os.path.dirname(BPE_PATH) or ".", exist_ok=True)
        tokenizer.save(BPE_PATH)
        self.tokenizer = tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        if hasattr(ids, "view"):
            ids = ids.view(-1).tolist()
        else:
            ids = list(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)
