[![CI](https://github.com/bopalvelut-prog/autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/bopalvelut-prog/autoresearch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/bopalvelut-prog/autoresearch?style=social)](https://github.com/bopalvelut-prog/autoresearch/stargazers)

# AutoResearch — CPU Edition

**An autonomous AI researcher that runs on any computer. No GPU required.**

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The original needs an H100. This runs on your laptop while you sleep.

```
val_bpb: 2.287 (baseline) → 2.226 (after autonomous tuning)
```

## What it does

1. A small local LLM (Qwen 2.5 0.5B via Ollama) suggests hyperparameter changes
2. `train.py` runs a 5-minute training experiment
3. If the result improves, it's committed automatically
4. Repeat — ~12 experiments per hour, ~100 while you sleep

```
step 00142 (100.0%) | loss: 2.226145 | epoch: 0 | remaining: 0s
---
val_bpb:          2.226000
training_seconds: 300.0
num_params_M:     0.8
```

## Quick start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/bopalvelut-prog/autoresearch.git
cd autoresearch && uv sync

# 3. Download data (one-time)
uv run prepare.py

# 4. Run a single experiment
uv run train.py

# 5. Let the agent run overnight
python agent.py
```

Works on **Linux, macOS, Windows**. Auto-detects CPU / Apple Silicon / NVIDIA GPU.

## How it works

Three files:

| File | Purpose |
|------|---------|
| `prepare.py` | Data download, tokenizer, evaluation. Don't touch. |
| `train.py` | GPT model + optimizer + training loop. The agent edits this. |
| `program.md` | Instructions for the agent. You edit this. |
| `agent.py` | Autonomous research loop with Ollama + JSON logging. |

All experiments use a **fixed 5-minute time budget**. The metric is **val_bpb** (validation bits per byte) — lower is better.

## Results tracking

Every experiment is logged to:
- `results.tsv` — flat TSV for quick viewing
- `results/run_*.json` — structured JSON per run
- `results/experiments.csv` — aggregate CSV for analysis

View your leaderboard:

```bash
uv run leaderboard.py --format md --top 10
uv run leaderboard.py --format json --export
```

## Tuning for your hardware

The defaults are conservative (DEPTH=2, 0.8M params). For faster machines:

```python
# In train.py:
DEPTH = 4              # More layers = better quality, slower
TOTAL_BATCH_SIZE = 2**15  # 32768 tokens
DEVICE_BATCH_SIZE = 8
WINDOW_PATTERN = "L"   # Full attention (faster on beefy CPUs)
```

For weaker hardware (phones, old laptops):
```python
DEPTH = 1
TOTAL_BATCH_SIZE = 2**12  # 4096 tokens
MAX_SEQ_LEN = 128         # In prepare.py
```

## Notable forks

| Fork | Platform | Notes |
|------|----------|-------|
| [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) | macOS | MPS optimized |
| [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | Windows | NVIDIA RTX |

## License

MIT. Built on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
