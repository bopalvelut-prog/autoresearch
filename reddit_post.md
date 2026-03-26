# I built a toolkit that turns e-waste into an AI cluster

I have a 2009 eMachines E727 with a Pentium T4500 that I couldn't throw away. So I made it run AI.

## The toolkit

Three open-source projects that work together:

### 1. [Primaclaw](https://github.com/bopalvelut-prog/e727-local-ai) — Distributed AI from old hardware
- Connects old laptops, Raspberry Pis, phones into one OpenAI-compatible API
- Workers auto-discover via UDP
- Drop-in replacement for OpenAI endpoints
- Dashboard included

### 2. [AutoResearch CPU Edition](https://github.com/bopalvelut-prog/autoresearch) — Autonomous AI researcher on any computer
- Fork of karpathy/autoresearch, no GPU needed
- AI agent tunes hyperparameters overnight
- Improved val_bpb from 2.29 → 2.23 autonomously

### 3. [Model Efficiency Comparator](https://github.com/bopalvelut-prog/model-efficiency) — Find the best model for your hardware
- Benchmarks speed, quality, security
- Works with prima.cpp and llama.cpp
- Generates HTML/JSON/Markdown reports

## Real results

| Machine | Year | CPU | Model | Speed |
|---------|------|-----|-------|-------|
| eMachines E727 | 2009 | Pentium T4500 | Qwen2.5 1.5B Q4 | 1.7 t/s |
| iPhone 11 (iSH) | 2019 | A13 (emulated) | Qwen2.5 0.5B | ~0.3 t/s |
| Acer Swift 3 | 2020 | Ryzen 5 4500U | Qwen2.5 3B Q4 | 0.5 t/s |

## Quick start

```bash
# Clone Primaclaw
git clone https://github.com/bopalvelut-prog/e727-local-ai
cd e727-local-ai
pip install -e .

# Start the coordinator (brain)
python -m src.coordinator

# On each old machine:
python -m src.worker
```

All MIT licensed. Works on Alpine Linux and even iPhone via iSH.

**Would love feedback from the community!**

---

**Subreddits:** r/LocalLLaMA, r/MachineLearning, r/selfhosted, r/linux
