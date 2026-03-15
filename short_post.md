# Short Reddit Post - r/LocalLLaMA

**Title:** Made @karpathy's Autoresearch work on CPU - runs on any computer!

**Body:**

Forked [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and made it work on regular Linux CPU - no GPU needed.

The autonomous agent ran overnight and improved val_bpb from 2.29 → 2.23 (2.7% better)

```bash
git clone https://github.com/bopalvelut-prog/autoresearch
cd autoresearch
uv sync
uv run prepare.py
uv run train.py
```

Runs in ~5 minutes on any computer. The agent tests hyperparameters automatically and keeps improvements.

Would love feedback!
