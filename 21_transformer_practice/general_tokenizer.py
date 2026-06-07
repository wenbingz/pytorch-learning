from typing import Protocol, runtime_checkable

@runtime_checkable
class GeneralTokenizer(Protocol):

    def __init__(self, corpus_path: str, reload = False):
        pass
    def encode(self, text: str):
        pass
    
    def decode(self, ids: list[int]):
        pass
    
    def get_vocab_size(self):
        pass   
