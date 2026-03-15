#!/bin/bash
# Tweet thread for AutoResearch LAN Swarm

echo "🧵 I've just released the 'LAN Research Swarm' for @karpathy's autoresearch! 🐝

Turn every idle device in your house into a distributed AI research cluster (Mac, Linux, Windows).

Here's how it works:"

echo ""
echo "1/ Inspired by @ExoLabs (auto-discovery) and @ggerganov's prima.cpp (real-time dashboards), I've built a zero-config swarm mode for local LLM research."

echo ""
echo "2/ The Coordinator (The Brain):
- Runs on your main PC.
- Uses a local LLM (Ollama) to generate experiment ideas.
- Hosts a web dashboard at localhost:8000.
- Broadcasts itself on the LAN via mDNS."

echo ""
echo "3/ The Workers (The Muscle):
- Run `worker.py` on any other LAN device.
- They auto-discover the coordinator instantly (no IP config).
- They grab tasks, run 5-min training jobs, and report back."

echo ""
echo "4/ It's like 'Folding@home' but you own the model. You can parallelize hyperparameter search across your network. I'm seeing 3x faster results on my home LAN!"

echo ""
echo "5/ The Web Dashboard shows your global best BPB and worker status in real-time. It's built with FastAPI + Jinja2 for maximum efficiency. 📊"

echo ""
echo "6/ You can try it too:

git clone https://github.com/bopalvelut-prog/autoresearch
uv run coordinator.py
(on another PC) uv run worker.py

Runs on any hardware! 🚀"

echo ""
echo "#AI #LocalLLM #OpenSource #DistributedComputing #Python"
