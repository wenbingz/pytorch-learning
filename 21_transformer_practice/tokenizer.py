
class CharTokenizer:
    def __init__(self, corpus_path: str, reload = False):
        with open(corpus_path, "r") as f:
            corpus = f.read()
        self.chars = sorted(set(corpus))
        self.itos = {id:ch for id, ch in enumerate(self.chars)}
        self.stoi = {ch:id for id, ch in self.itos.items()}
    def get_vocab_size(self):
        return len(self.chars)
    
    def encode(self, text: str):
        return [self.stoi[ch] if ch in self.stoi else self.stoi["<unk>"] for ch in text]
    def decode(self, ids: list[int]):
        return "".join([self.itos[id] for id in ids])
    
