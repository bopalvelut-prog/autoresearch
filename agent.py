#!/usr/bin/env python3
"""
AutoResearch "Folding Edition" - Idle Background Researcher
Optimized for low priority and LOW MEMORY consumption.
"""

import os
import sys
import time
import subprocess
import re
import json
import csv
import psutil
import gc
import requests
from datetime import datetime

MODEL = "qwen2.5-0.5b"
API_URL = "http://localhost:8080/v1/chat/completions"
RESULTS_FILE = "results.tsv"
RESULTS_DIR = "results"
TRAIN_FILE = "train.py"
LOG_FILE = "run.log"

# SET PROCESS PRIORITY TO BELOW NORMAL
p = psutil.Process(os.getpid())
if os.name == "nt":
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
else:
    p.nice(19)


def chat(prompt):
    """Chat via OpenAI-compatible API (prima.cpp/llama.cpp server)."""
    try:
        response = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "AI Researcher. Output ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 256,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "{}"


def run_command(cmd, timeout=600):
    """Run training with low priority inherited."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return result.stdout + result.stderr
    except Exception:
        return "ERROR"


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "latest"


def parse_results(output):
    results = {}
    for pattern, key in [
        (r"val_bpb:\s+([\d\.]+)", "val_bpb"),
        (r"training_seconds:\s+([\d\.]+)", "training_seconds"),
        (r"num_params_M:\s+([\d\.]+)", "num_params_M"),
        (r"num_steps:\s+(\d+)", "num_steps"),
        (r"mfu_percent:\s+([\d\.]+)", "mfu_percent"),
        (r"total_tokens_M:\s+([\d\.]+)", "total_tokens_M"),
    ]:
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            results[key] = float(val) if "." in val else int(val)
    return results


def log_result_tsv(commit, bpb, status, desc):
    header = "commit\tval_bpb\tstatus\tdescription\n"
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(header)
        f.write(f"{commit}\t{bpb:.6f}\t{status}\t{desc}\n")


def log_result_json(results, changes, status):
    """Write structured JSON result to results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit = get_git_hash()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "status": status,
        "changes": changes,
        "val_bpb": results.get("val_bpb", 0.0),
        "training_seconds": results.get("training_seconds", 0),
        "num_params_M": results.get("num_params_M", 0),
        "num_steps": results.get("num_steps", 0),
        "mfu_percent": results.get("mfu_percent", 0),
        "total_tokens_M": results.get("total_tokens_M", 0),
    }

    path = os.path.join(RESULTS_DIR, f"run_{ts}_{status}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    # Also append to aggregate CSV
    csv_path = os.path.join(RESULTS_DIR, "experiments.csv")
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(entry)


def main():
    print("=== AutoResearch: FOLDING EDITION (Low Memory Mode) ===")

    subprocess.run("git restore train.py", shell=True)

    best_bpb = 999.0
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == "keep":
                    try:
                        val = float(parts[1])
                        if 0 < val < best_bpb:
                            best_bpb = val
                    except Exception:
                        continue

    if best_bpb == 999.0:
        best_bpb = 2.15525
    print(f"Current best: {best_bpb}. Starting research loop...")

    while True:
        try:
            gc.collect()

            # 1. Suggest change
            prompt = (
                f"Best BPB: {best_bpb}. Suggest ONE change for MATRIX_LR, "
                f"EMBEDDING_LR, SCALAR_LR, or WEIGHT_DECAY. "
                f'Output ONLY JSON. Example: {{"MATRIX_LR": 0.041}}'
            )
            ai_response = chat(prompt)
            match = re.search(r"\{.*\}", ai_response.replace("\n", ""))
            if not match:
                time.sleep(10)
                continue

            changes = json.loads(match.group())

            # 2. Apply change
            with open(TRAIN_FILE, "r", encoding="utf-8") as f:
                code = f.read()

            desc = ""
            new_code = code
            for var, val in changes.items():
                if var in code:
                    new_code = re.sub(
                        rf"({var}\s*=\s*)[\d\.\*e\-]+", rf"\g<1>{val}", new_code
                    )
                    desc += f"{var}={val} "

            if new_code == code:
                time.sleep(5)
                continue

            with open(TRAIN_FILE, "w", encoding="utf-8") as f:
                f.write(new_code)

            # 3. Run training
            print(f"[{time.strftime('%H:%M:%S')}] Testing: {desc}")
            output = run_command("uv run train.py")
            results = parse_results(output)

            # 4. Evaluate and Revert if needed
            if "val_bpb" in results:
                new_bpb = results["val_bpb"]
                if new_bpb < best_bpb:
                    print(f"!!! SUCCESS: {new_bpb}")
                    best_bpb = new_bpb
                    subprocess.run(
                        f'git commit -am "Improve to {new_bpb} via {desc}"', shell=True
                    )
                    subprocess.run("git push mine master", shell=True)
                    log_result_tsv("latest", new_bpb, "keep", desc)
                    log_result_json(results, changes, "keep")
                else:
                    print(f"Discarding {new_bpb}")
                    log_result_tsv("latest", new_bpb, "discard", desc)
                    log_result_json(results, changes, "discard")
                    subprocess.run("git restore train.py", shell=True)
            else:
                print("Crash detected.")
                log_result_tsv("latest", 0.0, "crash", desc)
                log_result_json({}, changes, "crash")
                subprocess.run("git restore train.py", shell=True)

            del output
            gc.collect()

        except Exception as e:
            print(f"Loop error: {e}")
            subprocess.run("git restore train.py", shell=True)
            time.sleep(30)

        time.sleep(5)


if __name__ == "__main__":
    main()
