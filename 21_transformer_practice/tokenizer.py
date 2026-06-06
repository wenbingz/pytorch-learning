class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.itos = {id:ch for id, ch in enumerate(self.chars)}
        self.stoi = {ch:id for id, ch in self.itos.items()}

    def get_vocab_size(self):
        return len(self.chars)
    
    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids: list[int]):
        return "".join([self.itos[id] for id in ids])
    
