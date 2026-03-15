# I made @karpathy's Autoresearch run on a regular CPU - no GPU required!

**tl;dr**: Forked the famous autoresearch project and made it work on any Linux CPU. The autonomous agent improved val_bpb from 2.29 → 2.23 overnight.

---

## Background

If you haven't seen [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), it's brilliant: give an AI agent a small LLM training setup, let it experiment autonomously, and wake up to better models.

The catch? Original requires an H100 GPU.

## What I Did

1. **Adapted for CPU** - reduced model to 0.8M params, seq_len 256
2. **Added device detection** - works on CPU/MPS/CUDA automatically  
3. **Ran autonomous experiments** - let an AI agent try 20+ hyperparameter configs overnight

## Results

| Metric | Before | After |
|--------|--------|-------|
| val_bpb | 2.287 | 2.226 |

That's a **2.7% improvement** - all autonomously!

### Key findings:
- DEPTH: 2→3 helped
- Full attention (L) better than sliding window (SSSL)
- WEIGHT_DECAY: 0.2→0.15
- WARMDOWN_RATIO: 0.5→0.4

## Try It Yourself

```bash
git clone https://github.com/bopalvelut-prog/autoresearch
cd autoresearch
uv sync
uv run prepare.py
uv run train.py
```

Runs on any computer in ~5 minutes!

---

Would love feedback! The autonomous agent is running more experiments as we speak 📈

#MachineLearning #AI #OpenSource #Python