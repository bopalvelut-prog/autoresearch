# Autoresearch (CPU-friendly fork) × Qwen2.5-0.5B (Target) × Big LLM (Brain) × OpenClaw × LAN Swarm

**Goal**
Leverage a distributed **LAN Research Swarm** to run an autonomous AI research loop (via [bopalvelut-prog/autoresearch](https://github.com/bopalvelut-prog/autoresearch)).
The "Brain" of this swarm, powered by a **larger LLM served locally via `llama-server` (`prima.cpp`/`llama.cpp`)**, continuously experiments on and improves the training of **Qwen2.5-0.5B** (or similar tiny models).
The aim is to achieve the strongest possible chat/perplexity performance on everyday hardware — laptop CPUs, M-series Macs, modest GPUs — and expose the entire loop as an **OpenClaw** skill/agent.

Status: March 2026 — runs well on laptops (even CPU-only), M1/M2/M3/M4, RTX 3060–5090, etc. Achieves meaningful overnight gains (e.g. bits-per-byte / bpb reductions) through parallel experimentation across your local network.

## Key Features & How it Works

*   **Powerful "Brain" (via `llama-server`):** Instead of a small local LLM, a larger model (e.g., Llama 3 8B) running on `llama-server` provides more intelligent hyperparameter suggestions to the research loop.
*   **LAN Research Swarm:** Distribute the workload! Run `coordinator.py` on one machine and `worker.py` on others (CPU, Mac, GPU). They auto-discover each other and run experiments in parallel, drastically accelerating research.
*   **Autonomous Research Loop:** Continuously experiments, trains, evaluates, and integrates improvements into the target model (`train.py`).
*   **Universal Hardware Support:** Optimized for efficiency on consumer-grade hardware.
*   **OpenClaw Integration:** The research loop is designed to be accessible and controllable as an OpenClaw skill/agent.

## Quick Start (OpenClaw + bopalvelut-prog/autoresearch)

1.  **Install OpenClaw** (latest main recommended)
    ```bash
    # follow https://github.com/openclaw/openclaw
    # docker-compose or one-click installers work great
    ```

2.  **Setup `autoresearch` Coordinator & Workers** (after cloning and `uv sync`)
    *   **Coordinator (Main Machine):**
        *   **Start your `llama-server`** (running `prima.cpp`/`llama.cpp` with your chosen large model, typically on port 8081). This is your powerful "Brain."
        *   Then, start the coordinator: `uv run coordinator.py`
    *   **Workers (Other Machines on LAN):**
        *   Ensure `uv` and PyTorch are installed.
        *   Start a worker: `uv run worker.py`

3.  **Monitor Your Swarm:**
    *   Open `http://<coordinator-ip>:8000` in your browser to see the real-time research progress.
