"""
AutoResearch Chat Demo - OPTIMIZED VERSION
Uses your best hyperparameters from autonomous experiments!
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

# --- YOUR OPTIMIZED VALUES (from autonomous experiments) ---
ASPECT_RATIO = 16
HEAD_DIM = 32
TOTAL_BATCH_SIZE = 2**13
EMBEDDING_LR = 0.4  # optimized
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.15  # YOUR BEST VALUE!
DEPTH = 3  # optimized
DEVICE_BATCH_SIZE = 8  # optimized
TIME_BUDGET = 600  # 10 minutes
WARMDOWN_RATIO = 0.4  # optimized

# Device Setup
device = torch.device("cpu")
print(f"Using device: {device}")

# --- MODEL ARCHITECTURE ---


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 8192
    n_layer: int = 2
    n_head: int = 1
    n_kv_head: int = 1
    n_embd: int = 32
    window_pattern: str = "L"  # optimized - full attention


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


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
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = norm(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(norm(x))
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Main ---

print("=" * 50)
print("AUTORESEARCH - OPTIMIZED CHAT DEMO")
print("=" * 50)
print(f"Training with YOUR best hyperparameters:")
print(f"  - WEIGHT_DECAY: {WEIGHT_DECAY}")
print(f"  - DEPTH: {DEPTH}")
print(f"  - DEVICE_BATCH_SIZE: {DEVICE_BATCH_SIZE}")
print("=" * 50)

tokenizer = Tokenizer.from_directory()
config = GPTConfig(
    vocab_size=tokenizer.get_vocab_size(), n_layer=DEPTH, n_embd=DEPTH * ASPECT_RATIO
)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=MATRIX_LR, weight_decay=WEIGHT_DECAY
)
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, 256, "train", device="cpu")

print("\nTraining for 10 minutes...")
t_start = time.time()
step = 0
while time.time() - t_start < TIME_BUDGET:
    x, y, _ = next(train_loader)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    if step % 10 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")
    step += 1

print("\n" + "=" * 50)
print("Training Complete! Now chat with your AI!")
print("=" * 50)

# --- Chat Mode ---
model.eval()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        break

    input_ids = torch.tensor(
        [tokenizer.encode(user_input)], dtype=torch.long, device=device
    )
    generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
    output_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"AI: {output_text}")
