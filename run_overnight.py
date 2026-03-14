#!/usr/bin/env python3
"""
Fully automated overnight experiment loop
Runs continuously, making small changes and testing them
"""

import subprocess
import random
import time
import re
from pathlib import Path
import sys

REPO_DIR = Path("/home/ma/autoresearch")
MAX_EXPERIMENTS = 100  # ~8 hours at 5 min/experiment

# Hyperparameter search space
EXPERIMENTS = [
    # Model size changes
    ("ASPECT_RATIO", "20"),
    ("DEPTH", "3"),
    ("DEPTH", "4"),
    # Learning rate changes
    ("MATRIX_LR", "0.05"),
    ("MATRIX_LR", "0.03"),
    ("EMBEDDING_LR", "0.8"),
    ("EMBEDDING_LR", "0.4"),
    # Regularization
    ("WEIGHT_DECAY", "0.15"),
    ("WEIGHT_DECAY", "0.25"),
    # Schedules
    ("WARMUP_RATIO", "0.05"),
    ("WARMDOWN_RATIO", "0.4"),
    ("WARMDOWN_RATIO", "0.6"),
    # Other
    ("WINDOW_PATTERN", '"L"'),
    ("DEVICE_BATCH_SIZE", "6"),
    ("DEVICE_BATCH_SIZE", "8"),
]


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


def read_train_py() -> str:
    return (REPO_DIR / "train.py").read_text()


def write_train_py(content: str):
    (REPO_DIR / "train.py").write_text(content)


def apply_change(param: str, value: str) -> bool:
    """Apply a hyperparameter change to train.py."""
    content = read_train_py()
    lines = content.split("\n")

    for i, line in enumerate(lines):
        if line.strip().startswith(f"{param} ="):
            old_line = line
            # Find the comment and preserve it
            if "#" in line:
                idx = line.index("#")
                comment = line[idx:]
                new_line = f"{param} = {value}  {comment}"
            else:
                new_line = f"{param} = {value}"

            lines[i] = new_line
            write_train_py("\n".join(lines))
            return True
    return False


def reset_change(param: str):
    """Reset a hyperparameter to original."""
    apply_change(param, get_original(param))


def get_original(param: str) -> str:
    """Get original value for a parameter."""
    defaults = {
        "ASPECT_RATIO": "16",
        "DEPTH": "2",
        "MATRIX_LR": "0.04",
        "EMBEDDING_LR": "0.6",
        "WEIGHT_DECAY": "0.2",
        "WARMUP_RATIO": "0.0",
        "WARMDOWN_RATIO": "0.5",
        "WINDOW_PATTERN": '"SSSL"',
        "DEVICE_BATCH_SIZE": "4",
    }
    return defaults.get(param, "")


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
    memory_gb = 0.0
    with open(tsv, "a") as f:
        f.write(f"{commit}\t{val_bpb}\t{memory_gb}\t{status}\t{desc}\n")


def main():
    print("=" * 60)
    print("AUTONOMOUS OVERNIGHT EXPERIMENT LOOP")
    print(f"Best current val_bpb: {get_best_bpb():.6f}")
    print("=" * 60)
    print(f"Will run up to {MAX_EXPERIMENTS} experiments...")
    print("Press Ctrl+C to stop")
    print()
    sys.stdout.flush()

    experiment_idx = 0
    consecutive_failures = 0

    while experiment_idx < MAX_EXPERIMENTS:
        # Pick a random experiment
        param, value = random.choice(EXPERIMENTS)
        original_value = get_original(param)

        print(f"\n--- Experiment {experiment_idx + 1}/{MAX_EXPERIMENTS} ---")
        print(f"Testing {param}: {original_value} -> {value}")

        # Apply change
        if not apply_change(param, value):
            print(f"Could not apply change for {param}")
            continue

        # Run training
        print("Running training...")
        result = run_training()

        if result["success"]:
            new_bpb = result["val_bpb"]
            old_bpb = get_best_bpb()
            improvement = old_bpb - new_bpb

            print(
                f"val_bpb: {new_bpb:.6f} (best: {old_bpb:.6f}, diff: {improvement:.6f})"
            )

            if new_bpb < old_bpb:
                print("IMPROVED! Committing...")
                commit = git_commit(f"exp {experiment_idx + 1}: {param}={value}")
                record_result(
                    commit, new_bpb, "keep", f"{param}:{original_value}->{value}"
                )
                consecutive_failures = 0
            else:
                print("No improvement, reverting...")
                apply_change(param, original_value)
                subprocess.run(["git", "checkout", "train.py"], cwd=REPO_DIR)
                record_result(
                    "reverted", new_bpb, "discard", f"{param}:{original_value}->{value}"
                )
                consecutive_failures += 1
        else:
            print("Training failed!")
            apply_change(param, original_value)
            subprocess.run(["git", "checkout", "train.py"], cwd=REPO_DIR)
            record_result("failed", 0, "crash", f"{param}:{value}")
            consecutive_failures += 1

        print(f"Current best: {get_best_bpb():.6f}")

        # Stop if too many consecutive failures
        if consecutive_failures >= 10:
            print("\nToo many consecutive failures, stopping.")
            break

        experiment_idx += 1
        time.sleep(2)

    print("\n" + "=" * 60)
    print("Experiment loop complete!")
    print(f"Final best val_bpb: {get_best_bpb():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        print(f"Best val_bpb: {get_best_bpb():.6f}")
