#!/bin/bash
# Autonomous research agent wrapper for autoresearch
# Usage: ./run_agent.sh

cd /home/ma/autoresearch

echo "=============================================="
echo "Autoresearch Autonomous Agent"
echo "Using Ollama (llama3.2)"
echo "=============================================="
echo ""
echo "Current results:"
cat results.tsv
echo ""
echo "Starting interactive agent session..."
echo "The agent will help you experiment with train.py"
echo ""

# Read the program.md content
PROGRAM_MD=$(cat program.md)

# Create a prompt for the agent
PROMPT="You are an autonomous AI researcher working on the autoresearch project.
Your goal is to improve val_bpb (validation bits per byte) - LOWER IS BETTER.

Current baseline: val_bpb = 2.287261

IMPORTANT RULES:
1. Only modify train.py - never modify prepare.py
2. Each experiment runs for ~5 minutes max (CPU)
3. If val_bpb improves (lower), keep the commit
4. If val_bpb gets worse, discard (git reset)
5. Record ALL results in results.tsv (untracked file)
6. NEVER ask the human for permission - run autonomously
7. You're on CPU with a small model (~800K parameters)

First, read train.py and tell me what you would change to improve the model.
Then we can make that change and run an experiment.

The project uses uv for package management. Run 'uv run train.py' to train."

# Use ollama to chat
echo "$PROMPT" | ollama run llama3.2 --verbose
