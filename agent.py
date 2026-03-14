#!/usr/bin/env python3
"""
Autonomous research agent using Ollama.
Runs the experiment loop described in program.md
"""

import subprocess
import json
import os
import sys
import re
import time
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"
REPO_DIR = Path("/home/ma/autoresearch")


def ollama_chat(system: str, user: str, temperature: float = 0.7) -> str:
    """Send a chat request to Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        },
        stream=False,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def git_commit(commit_msg: str) -> str:
    """Git add and commit, return short hash."""
    subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_reset() -> None:
    """Reset to previous commit."""
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=REPO_DIR, check=True)


def run_training() -> dict:
    """Run training and extract results."""
    result = subprocess.run(
        ["uv", "run", "train.py"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr

    # Extract metrics
    val_bpb = None
    training_seconds = None
    peak_vram = None

    for line in output.split("\n"):
        if line.startswith("val_bpb:"):
            val_bpb = float(line.split(":")[1].strip())
        elif line.startswith("training_seconds:"):
            training_seconds = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            peak_vram = float(line.split(":")[1].strip())

    return {
        "val_bpb": val_bpb,
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram,
        "output": output,
        "success": result.returncode == 0 and val_bpb is not None,
    }


def read_tsv() -> list:
    """Read results.tsv"""
    tsv_path = REPO_DIR / "results.tsv"
    if not tsv_path.exists():
        return []
    with open(tsv_path) as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return []
    rows = []
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            rows.append(
                {
                    "commit": parts[0],
                    "val_bpb": float(parts[1]),
                    "memory_gb": float(parts[2]),
                    "status": parts[3],
                    "description": parts[4],
                }
            )
    return rows


def write_tsv(rows: list) -> None:
    """Write results.tsv"""
    tsv_path = REPO_DIR / "results.tsv"
    with open(tsv_path, "w") as f:
        f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        for row in rows:
            f.write(
                f"{row['commit']}\t{row['val_bpb']}\t{row['memory_gb']}\t{row['status']}\t{row['description']}\n"
            )


def get_last_bpb() -> float:
    """Get the best (lowest) val_bpb from results."""
    rows = read_tsv()
    if not rows:
        return float("inf")
    return min(r["val_bpb"] for r in rows if r["status"] == "keep")


def main():
    print("=" * 60)
    print("AUTORESEARCH AUTONOMOUS AGENT")
    print("Using Ollama model:", MODEL)
    print("=" * 60)

    # Read program.md for instructions
    program_md = (REPO_DIR / "program.md").read_text()

    system_prompt = f"""You are an autonomous AI researcher running experiments on LLM training.
You are working on a CPU-based version of the autoresearch project.

The goal is to improve val_bpb (validation bits per byte) - LOWER IS BETTER.

Current baseline from results.tsv:
- val_bpb: {get_last_bpb():.6f}

You must follow these rules strictly:
1. Only modify train.py - never modify prepare.py
2. Each experiment runs for ~5 minutes max
3. If val_bpb improves (lower), the commit is kept
4. If val_bpb gets worse, the commit is discarded (git reset)
5. Always record results in results.tsv
6. NEVER ask the human for permission to continue - run autonomously
7. Be creative but practical - small model on CPU has limited resources

{program_md}

Start by analyzing the current train.py and proposing an experiment. 
Make one small improvement at a time.
"""

    # Get initial analysis from agent
    train_py = (REPO_DIR / "train.py").read_text()

    user_msg = f"""Analyze train.py and propose your first experiment. 

Current train.py (first 100 lines):
{train_py[:5000]}

What change would you like to make to improve val_bpb?
Just tell me what you want to change - I'll make the edit and run the experiment.
"""

    while True:
        try:
            response = ollama_chat(system_prompt, user_msg, temperature=0.7)
            print("\n" + "=" * 40)
            print("AGENT RESPONSE:")
            print("=" * 40)
            print(response)
            print("=" * 40)

            # Check if agent wants to make a change
            if (
                "change" in response.lower()
                or "modify" in response.lower()
                or "experiment" in response.lower()
            ):
                # For now, just run another training cycle with the agent observing
                # In a full implementation, we'd parse the specific change
                pass

            # Ask agent what to do next
            user_msg = """What would you like to do next? Options:
1. Make a specific change to train.py (describe the change)
2. Run training to test current code
3. Check results and decide to keep/discard
4. Something else

Be specific about any code changes you'd like to make.
"""

            # Simple loop - just keep asking
            time.sleep(1)

        except KeyboardInterrupt:
            print("\nAgent stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
