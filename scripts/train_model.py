import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from transformers import PreTrainedTokenizerFast

# Hyperparameters for ~15M parameter Small Language Model
vocab_size = 32000
n_embd = 256
n_layer = 8
n_head = 8
block_size = 512
dropout = 0.1
batch_size = 16
learning_rate = 3e-4
max_iters = 1000

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections (the x + logic)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ The master GPT Architecture """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Sequentially stack the blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def get_batch(data, device):
    # Generates a random batch of data to feed the model from the memory-mapped binary file
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def train_model():
    print("Checking Hardware...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer_dir = Path("models/finance-tokenizer")
    if not tokenizer_dir.exists():
        print("Error: Run train_tokenizer.py first!")
        return

    print("Loading custom tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    print("Loading memory-mapped binary corpus...")
    corpus_path = Path("data/processed/corpus.bin")
    if not corpus_path.exists():
        print("Error: Run build_corpus.py first!")
        return
        
    # Stream the dataset directly from the SSD without loading into RAM
    data = np.memmap(str(corpus_path), dtype=np.uint16, mode='r')
    print(f"Successfully mapped {len(data):,} tokens into virtual memory.")

    print("Initializing Custom GPT Architecture (Pure PyTorch)...")
    model = GPTLanguageModel().to(device)
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    from tqdm import tqdm
    
    print("Beginning Training Loop! 🚀")
    model.train()
    
    pbar = tqdm(range(max_iters), desc="Training Phase")
    for iter in pbar:
        xb, yb = get_batch(data, device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    print("Training Complete. Saving PyTorch weights...")
    Path("models/finance-llm-custom").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/finance-llm-custom/model.pt")
    print("Model successfully saved to models/finance-llm-custom/model.pt")

if __name__ == "__main__":
    train_model()
