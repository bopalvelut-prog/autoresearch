#!/usr/bin/env python3
"""
Fully autonomous experiment loop for autoresearch
Uses Ollama or llama.cpp to generate changes, runs experiments automatically

Set LLAMA_CPP_ENABLED=1 to use llama-server instead of Ollama
Set GIT_PUSH_ENABLED=0 to disable git commit/push
"""

import os
import subprocess
import requests
import time
import re
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL", "http://localhost:8080/v1/chat/completions"
)
MODEL = "llama3.2"
REPO_DIR = Path("/home/ma/autoresearch")
MAX_EXPERIMENTS = 20

LLAMA_CPP_ENABLED = os.environ.get("LLAMA_CPP_ENABLED", "0") == "1"
GIT_PUSH_ENABLED = os.environ.get("GIT_PUSH_ENABLED", "1") == "1"


def llm_chat(system: str, user: str, temperature: float = 0.7) -> str:
    """Send a chat request to Ollama or llama-server."""
    if LLAMA_CPP_ENABLED:
        response = requests.post(
            LLAMA_SERVER_URL,
            json={
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

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
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def get_best_bpb() -> float:
    """Get the best val_bpb from results."""
    tsv = REPO_DIR / "results.tsv"
    if not tsv.exists():
        return float("inf")
    best = float("inf")
    for line in tsv.read_text().split("\n")[1:]:
        if line and "keep" in line:
            parts = line.split("\t")
            if len(parts) > 1:
                best = min(best, float(parts[1]))
    return best


def git_commit(msg: str) -> str:
    """Git add and commit."""
    subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_DIR, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


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

    val_bpb = None
    for line in output.split("\n"):
        if line.startswith("val_bpb:"):
            val_bpb = float(line.split(":")[1].strip())
            break

    return {
        "val_bpb": val_bpb,
        "success": result.returncode == 0 and val_bpb is not None,
    }


def record_result(commit: str, val_bpb: float, status: str, desc: str):
    """Record result in tsv."""
    tsv = REPO_DIR / "results.tsv"
    memory_gb = 0.0  # CPU doesn't have VRAM
    with open(tsv, "a") as f:
        f.write(f"{commit}\t{val_bpb}\t{memory_gb}\t{status}\t{desc}\n")


def main():
    print("=" * 60)
    print("AUTONOMOUS EXPERIMENT LOOP")
    print(f"Best current val_bpb: {get_best_bpb():.6f}")
    print("=" * 60)

    train_py = (REPO_DIR / "train.py").read_text()

    system = """You are an autonomous AI researcher optimizing a small LLM training loop.
Current best val_bpb: {best}

You must:
1. Propose ONE specific, small change to train.py
2. The change should be practical for a small CPU model (~800K params)
3. Keep it simple - small improvements only
4. Suggest hyperparameters, architecture tweaks, or optimizer changes

Suggest your change as a brief description like:
"increase n_embd from 32 to 48" or "change learning rate to 0.01" """

    user = f"""Current train.py config:
- n_layer: 2, n_head: 1, n_embd: 32
- vocab_size: 8192, sequence_len: 256
- Current val_bpb: {get_best_bpb():.6f}

What single change would you like to try? Just describe it briefly."""

    for i in range(MAX_EXPERIMENTS):
        print(f"\n--- Experiment {i + 1}/{MAX_EXPERIMENTS} ---")

        # Get suggestion from agent
        try:
            response = llm_chat(
                system.format(best=get_best_bpb()), user, temperature=0.8
            )
            print(f"Agent suggestion: {response[:200]}")
        except Exception as e:
            print(f"Error getting suggestion: {e}")
            break

        # Ask for confirmation before running
        print("Running training...")

        # Run training
        result = run_training()

        if result["success"]:
            new_bpb = result["val_bpb"]
            old_bpb = get_best_bpb()

            print(f"New val_bpb: {new_bpb:.6f} (was {old_bpb:.6f})")

            if new_bpb < old_bpb:
                print("IMPROVED! Committing...")
                if GIT_PUSH_ENABLED:
                    try:
                        commit = git_commit(f"experiment {i + 1}: {response[:50]}")
                    except:
                        commit = "local"
                else:
                    commit = "local"
                record_result(commit, new_bpb, "keep", response[:50])
            else:
                print("No improvement, discarding...")
                if GIT_PUSH_ENABLED:
                    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=REPO_DIR)
                record_result("discarded", new_bpb, "discard", response[:50])
        else:
            print("Training failed!")
            record_result("failed", 0, "crash", response[:50])

        print(f"Current best: {get_best_bpb():.6f}")

        # Small delay between experiments
        time.sleep(2)

    print("\n" + "=" * 60)
    print("Experiment loop complete!")
    print(f"Final best val_bpb: {get_best_bpb():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
