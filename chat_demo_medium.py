"""
AutoResearch Chat Demo - MEDIUM (10M Parameters)
Uses the best record numbers to train a larger model for 20 minutes.
Expect real English words to form!
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

# --- RECORD NUMBERS (Optimized for your CPU) ---
# Scaled up for 10M parameters
ASPECT_RATIO = 16
HEAD_DIM = 64
TOTAL_BATCH_SIZE = 2**14
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
DEPTH = 12             # 6x more layers than the small one
DEVICE_BATCH_SIZE = 4
TIME_BUDGET = 1200     # 20 MINUTES to allow the bigger brain to form

# Device Setup
device = torch.device("cpu")
print(f"Using device: {device}")

# --- MODEL ARCHITECTURE (Same as train.py) ---

@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 8192
    n_layer: int = 12
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    window_pattern: str = "SSSL"

def norm(x): return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = norm(self.transformer.wte(idx))
        for block in self.transformer.h: x = block(x)
        logits = self.lm_head(norm(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:] # context crop
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training Loop ---

tokenizer = Tokenizer.from_directory()
config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), n_layer=DEPTH, n_embd=DEPTH*ASPECT_RATIO)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=MATRIX_LR)
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, 256, "train", device="cpu")

print(f"Training a {DEPTH} layer, {config.n_embd} dim model (Medium Edition)...")
print("This will take about 20 minutes on your CPU.")
t_start = time.time()
step = 0
while time.time() - t_start < TIME_BUDGET:
    x, y, _ = next(train_loader)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    if step % 5 == 0:
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET - elapsed
        print(f"Step {step} | Loss {loss.item():.4f} | Time Remaining: {remaining:.0f}s")
    step += 1

print("\nMedium Training Complete!")

# --- Interaction Mode ---
print("\n--- AI DEMO MODE (MEDIUM) ---")
model.eval()

while True:
    user_input = input("\nEnter a prompt (or 'q' to quit): ")
    if user_input.lower() == 'q': break
    
    input_ids = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
    print("AI is thinking...", end="", flush=True)
    
    generated_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.7)
    output_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\rAI Response: " + output_text)
