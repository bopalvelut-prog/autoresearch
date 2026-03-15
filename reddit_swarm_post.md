# I made a "Folding@home" swarm for local LLM research (for @karpathy's autoresearch)

I've been playing with Andrej Karpathy's `autoresearch` repo (where an AI agent autonomously experiments with hyperparameters to improve a small LLM).

The original is single-GPU, but I wanted to speed up the search by using all my idle machines at home (an old Mac Mini, my Linux laptop, and a desktop).

So I built a **LAN Research Swarm** feature:

1.  **Coordinator**: Runs on your main PC. It uses a local LLM (Ollama) to generate experiment ideas (e.g., "Try lowering MATRIX_LR to 0.035"). It broadcasts itself on the LAN via mDNS (like `Exo`).
2.  **Workers**: Run `worker.py` on any other device. They auto-discover the coordinator, grab a task, modify `train.py` locally, run a 5-minute training job, and report the validation loss (BPB) back.
3.  **Dashboard**: A real-time web UI (inspired by `prima.cpp`) showing the global best BPB found so far and the status of all your workers.

It's completely zero-conf. Just run the scripts and watch your swarm optimize your model.

Code is here: https://github.com/bopalvelut-prog/autoresearch

I'd love feedback on the distributed architecture! I'm using FastAPI + ZeroConf.

#LocalLLM #Python #MachineLearning #DistributedComputing
