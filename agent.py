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
import psutil
import gc
import requests

MODEL = "qwen2.5:0.5b"
RESULTS_FILE = "results.tsv"
TRAIN_FILE = "train.py"
LOG_FILE = "run.log"

# Configuration via environment variables:
# - GIT_PUSH_ENABLED=0         : Disable git commit/push (local-only mode)
# - LLAMA_CPP_ENABLED=1        : Use llama.cpp server instead of Ollama
# - LLAMA_SERVER_URL           : URL of the llama.cpp server (default: http://localhost:8082)
# - LLAMA_CPP_MODEL_PATH       : (Optional) Path/name of the model used by llama.cpp server for logging/compliance. Server loads the model.

GIT_PUSH_ENABLED = os.environ.get("GIT_PUSH_ENABLED", "1") == "1"
LLAMA_CPP_ENABLED = os.environ.get("LLAMA_CPP_ENABLED", "0") == "1"
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8082")
LLAMA_CPP_MODEL = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/model.gguf")

# SET PROCESS PRIORITY TO BELOW NORMAL
p = psutil.Process(os.getpid())
if os.name == "nt":
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
else:
    p.nice(19)


def unload_model():
    """Tells Ollama to unload the model to free up RAM."""
    if LLAMA_CPP_ENABLED:
        return
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "keep_alive": 0},
            timeout=5,
        )
    except:
        pass


def chat(prompt):
    """Reliable CLI-based chat. Supports Ollama and llama.cpp."""
    full_prompt = f"System: AI Researcher. Output ONLY JSON.\nUser: {prompt}"
    try:
        if LLAMA_CPP_ENABLED:
            try:
                response = requests.post(
                    f"{LLAMA_SERVER_URL}/completion",
                    json={
                        "prompt": full_prompt,
                        "n_predict": 256,
                        "temperature": 0.7,
                        "mirostat": 2,
                        "model": LLAMA_CPP_MODEL # Include model to be compliant with newer llama.cpp servers
                    },
                    timeout=300,
                )
                response.raise_for_status() # Raise an exception for HTTP errors
                data = response.json()
                return data.get("content", "{}")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with llama.cpp server: {e}")
                return "{}"
        result = subprocess.run(
            ["ollama", "run", MODEL, full_prompt],
            capture_output=True,
            text=True,
            timeout=300,
            errors="replace",
        )
        # Immediately tell ollama we are done for now
        unload_model()
        return result.stdout
    except:
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
    except:
        return "ERROR"


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except:
        return "latest"


def parse_results(output):
    results = {}
    match = re.search(r"val_bpb:\s+([\d\.]+)", output)
    if match:
        results["val_bpb"] = float(match.group(1))
    return results


def log_result(commit, bpb, status, desc):
    header = "commit\tval_bpb\tstatus\tdescription\n"
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(header)
        f.write(f"{commit}\t{bpb:.6f}\t{status}\t{desc}\n")


def main():
    print("=== AutoResearch: FOLDING EDITION (Low Memory Mode) ===")

    # Ensure we start clean
    if GIT_PUSH_ENABLED:
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
                    except:
                        continue

    if best_bpb == 999.0:
        best_bpb = 2.15525
    print(f"Current best: {best_bpb}. Starting research loop...")

    while True:
        try:
            gc.collect()  # Aggressive cleanup

            # 1. Suggest change
            prompt = f'Best BPB: {best_bpb}. Suggest ONE change for MATRIX_LR, EMBEDDING_LR, SCALAR_LR, or WEIGHT_DECAY. Output ONLY JSON. Example: {{"MATRIX_LR": 0.041}}'
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
                    if GIT_PUSH_ENABLED:
                        subprocess.run(
                            f'git commit -am "Improve to {new_bpb} via {desc}"',
                            shell=True,
                            stderr=subprocess.DEVNULL,
                        )
                        subprocess.run(
                            "git push mine master",
                            shell=True,
                            stderr=subprocess.DEVNULL,
                        )
                    log_result("latest", new_bpb, "keep", desc)
                else:
                    print(f"Discarding {new_bpb}")
                    log_result("latest", new_bpb, "discard", desc)
                    if GIT_PUSH_ENABLED:
                        subprocess.run("git restore train.py", shell=True)
                    else:
                    print("Crash detected.")
                    log_result("latest", 0.0, "crash", desc)
                    if GIT_PUSH_ENABLED:
                        subprocess.run("git restore train.py", shell=True)
            # Cleanup after training run
            del output
            gc.collect()

        except Exception as e:
            print(f"Loop error: {e}")
            if GIT_PUSH_ENABLED:
                subprocess.run("git restore train.py", shell=True)
            time.sleep(30)

        time.sleep(5)


if __name__ == "__main__":
    main()
