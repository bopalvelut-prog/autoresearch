#!/bin/bash
# Tweet thread for autoresearch CPU edition

echo "🧵 I made @karpathy's #autoresearch run on a regular Linux CPU - no GPU needed!

The autonomous agent improved val_bpb from 2.29 → 2.23 overnight.

Here's how:"

echo ""
echo "1/ First, I forked @karpathy's autoresearch and adapted it for CPU-only execution. The original needs an H100 GPU - not exactly accessible for most of us 🫠"

echo ""
echo "2/ The key changes:
- Reduced model to 0.8M params (from 50M+)
- Sequence length 256 (from 2048)  
- Automatic device detection (CPU/MPS/CUDA)"

echo ""
echo "3/ Then I let an AI agent loose on the hyperparameters overnight. It ran 20+ experiments, testing learning rates, depth, batch sizes, and attention patterns."

echo ""
echo "4/ Results:
- Started: val_bpb = 2.287
- After tuning: val_bpb = 2.226
- That's a 2.7% improvement! 

The agent autonomously decided to keep the best changes."

echo ""
echo "5/ You can try it too:

git clone https://github.com/bopalvelut-prog/autoresearch
cd autoresearch
uv sync
uv run prepare.py
uv run train.py

Runs on any computer in ~5 minutes! 🚀"

echo ""
echo "#AI #MachineLearning #OpenSource #Python"
