# Viral Promotion Checklist

## Post to Reddit NOW
- [ ] r/MachineLearning
- [ ] r/LocalLLaMA  
- [ ] r/Artificial
- [ ] r/Python
- [ ] r/ArtificialIntelligence

## Twitter/X
- [ ] Post the tweet thread (5 tweets)
- [ ] Reply to @karpathy

## Hacker News
- [ ] Submit to news.ycombinator.com

## LinkedIn
- [ ] Post about your project

## Cross-post to other GitHub forks
- [ ] Comment on original karpathy/autoresearch
- [ ] Message other fork maintainers

---

## Ready-to-post content:

### Tweet 1:
🧵 I made @karpathy's #autoresearch run on a regular Linux CPU - no GPU needed! The autonomous agent improved val_bpb from 2.29 → 2.23 overnight. Here's how:

### Tweet 2:
1/ First, I forked @karpathy's autoresearch and adapted it for CPU-only. The original needs an H100 GPU 🫠

### Tweet 3:
2/ Key changes: 0.8M params, seq_len 256, automatic device detection

### Tweet 4:
3/ Let an AI agent loose on hyperparameters overnight. 20+ experiments. Results: 2.7% improvement!

### Tweet 5:
5/ Try it: `git clone https://github.com/bopalvelut-prog/autoresearch` → `uv sync && uv run train.py` #AI #MachineLearning

---

### Reddit title:
Made @karpathy's Autoresearch work on CPU - runs on any computer!

### Reddit body:
Forked [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and made it work on regular Linux CPU - no GPU needed.

The autonomous agent ran overnight and improved val_bpb from 2.29 → 2.23 (2.7% better)

\`\`\`bash
git clone https://github.com/bopalvelut-prog/autoresearch
cd autoresearch
uv sync
uv run prepare.py
uv run train.py
\`\`\`
