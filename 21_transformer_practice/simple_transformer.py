import torch
from torch import nn
from tokenizer import CharTokenizer
import argparse
from typing import Tuple, Union  # Add this import
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embeding_dim: int, head_size: int):
        super().__init__()
        self.embeding_dim = embeding_dim
        self.head_size = head_size
        self.q = nn.Linear(embeding_dim, head_size)
        self.k = nn.Linear(embeding_dim, head_size)
        self.v = nn.Linear(embeding_dim, head_size)
        self.normal = nn.LayerNorm(head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        wei = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        wei = wei.masked_fill(torch.tril(torch.ones_like(wei))[:time, :time] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return self.normal(out)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embeding_dim: int, head_size: int, head_num: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.heads = nn.ModuleList([SelfAttentionHead(embeding_dim, head_size) for _ in range(head_num)])
        self.pro = nn.Linear(head_size * head_num, embeding_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pro(torch.cat([head(x) for head in self.heads], dim=-1))
        


class TransformerBlock(nn.Module):

    def __init__(self, embeding_dim, block_size, head_num, vocab_size):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(embeding_dim, embeding_dim // head_num, head_num, vocab_size)
        self.ln1 = nn.LayerNorm(embeding_dim)
        self.ln2 = nn.LayerNorm(embeding_dim)
        self.ffn = nn.Sequential(nn.Linear(embeding_dim, embeding_dim*4), nn.GELU(), nn.Linear(embeding_dim*4, embeding_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multiHeadAttention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
        
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embeding_dim, block_size, head_num, num_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeding_dim)
        self.position_embeding = nn.Embedding(block_size, embeding_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embeding_dim, block_size, head_num, vocab_size) for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embeding_dim)
        self.lm_head = nn.Linear(embeding_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:  # Fixed typo and return type
        batch_size, time = x.shape
        x = self.embedding(x)
        x = x + self.position_embeding(torch.arange(time, device=x.device))
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            batch, time, vocab = logits.shape
            target = target.view(batch * time)
            logits = logits.view(batch * time, vocab)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def __sample_next__(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, None)
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id   
    
def build_data(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    max_start = data.shape[0] - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
    
    
    
def train_model(data: torch.Tensor, model: SimpleTransformer, block_size: int, batch_size: int, steps: int, cp_path: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for step in range(steps):
        xb, yb = build_data(data, block_size, batch_size, device)
        logits, loss = model(xb, yb)
        print(f"step {step}, loss: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), cp_path)
    print(f"model saved to {cp_path}")

    

#  def generate(model: SimpleTransformer, data: torch.Tensor, block_size: int, cp_path: str):
#     model.load_state_dict(torch.load(cp_path))
#     model.
        



    # for epoch in range(100):
    #     model.train()
    #     loss_meter = AverageMeter()
    #     acc_meter = AverageMeter()
    #     for xb, yb in data:
    #         xb, yb = _move_to_device((xb, yb), device)
    #         logits, loss = model(xb, yb)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad(set_to_none=True)
    #         loss_meter.update(loss.item(), n=xb.shape[0])
    #         acc_meter.update(accuracy_top1(logits, yb), n=xb.shape[0])
    #     print(f"Epoch {epoch}, Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or sample a mini Transformer.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--generate", action="store_true", help="generate with query")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path = "data/transformer.txt"
    cp_path = "data/transformer_model.cp"
    with open(path, "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    embeding_dim = 512
    block_size = 256
    batch_size = 64
    head_num = 4
    num_blocks = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(vocab_size=tokenizer.get_vocab_size(), embeding_dim=embeding_dim, block_size=block_size, head_num=head_num, num_blocks=num_blocks)
    model.to(device)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
    if args.train:
        train_model(data, model, block_size, batch_size, args.steps, cp_path)
    elif args.generate:
        print("not supported")
    else:
        print("not supported")
    
    

