#!/usr/bin/env python3
"""
AutoResearch "Folding Edition" - Idle Background Researcher
Optimized to run in the background with low priority.
"""

import os
import sys
import time
import subprocess
import re
import json
import psutil # We use this to set low priority

MODEL = "qwen2.5:0.5b"
RESULTS_FILE = "results.tsv"
TRAIN_FILE = "train.py"
LOG_FILE = "run.log"

# SET PROCESS PRIORITY TO IDLE/BELOW NORMAL
# This ensures the computer stays smooth for the user while training.
p = psutil.Process(os.getpid())
if os.name == 'nt':
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
else:
    p.nice(19) 

def chat(prompt):
    """Reliable CLI-based chat."""
    full_prompt = f"System: AI Researcher. Output ONLY JSON.\nUser: {prompt}"
    try:
        result = subprocess.run(
            ['ollama', 'run', MODEL, full_prompt],
            capture_output=True, text=True, timeout=300
        )
        return result.stdout
    except:
        return "{}"

def run_command(cmd, timeout=600):
    """Run training with low priority inherited."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except:
        return "ERROR"

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except:
        return "unknown"

def parse_results(output):
    results = {}
    match = re.search(r"val_bpb:\s+([\d\.]+)", output)
    if match: results["val_bpb"] = float(match.group(1))
    return results

def log_result(commit, bpb, status, desc):
    header = "commit\tval_bpb\tstatus\tdescription\n"
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if not file_exists: f.write(header)
        f.write(f"{commit}\t{bpb:.6f}\t{status}\t{desc}\n")

def main():
    print("=== AutoResearch: FOLDING EDITION (Always-On Background Mode) ===")
    print("Priority set to BELOW NORMAL. Your computer will stay responsive.")
    
    best_bpb = 999.0
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] == "keep":
                    try:
                        val = float(parts[1])
                        if 0 < val < best_bpb: best_bpb = val
                    except: continue

    if best_bpb == 999.0: best_bpb = 2.15525
    print(f"Current best: {best_bpb}. Starting infinite research loop...")

    while True:
        try:
            with open(TRAIN_FILE, "r", encoding="utf-8") as f:
                current_code = f.read()
            
            prompt = f"Best BPB: {best_bpb}. Suggest ONE change for MATRIX_LR, EMBEDDING_LR, SCALAR_LR, or WEIGHT_DECAY. Output ONLY JSON. Example: {{\"MATRIX_LR\": 0.041}}"
            
            ai_response = chat(prompt)
            match = re.search(r"\{.*\}", ai_response.replace("\n", ""))
            if not match: 
                time.sleep(10)
                continue
            
            changes = json.loads(match.group())
            new_code = current_code
            desc = ""
            for var, val in changes.items():
                if var in current_code:
                    new_code = re.sub(fr"({var}\s*=\s*)[\d\.\*e\-]+", fr"\g<1>{val}", new_code)
                    desc += f"{var}={val} "
            
            if new_code == current_code:
                time.sleep(5)
                continue

            with open(TRAIN_FILE, "w", encoding="utf-8") as f:
                f.write(new_code)
            
            print(f"[{time.strftime('%H:%M:%S')}] Testing: {desc}")
            output = run_command("uv run train.py")
            results = parse_results(output)
            
            if "val_bpb" in results:
                new_bpb = results["val_bpb"]
                if new_bpb < best_bpb:
                    print(f"!!! IMPROVEMENT: {new_bpb}")
                    best_bpb = new_bpb
                    subprocess.run(f'git commit -am "Improve to {new_bpb} via {desc}"', shell=True)
                    log_result(get_git_hash(), new_bpb, "keep", desc)
                else:
                    log_result(get_git_hash(), new_bpb, "discard", desc)
                    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
                        f.write(current_code)
            else:
                log_result(get_git_hash(), 0.0, "crash", desc)
                with open(TRAIN_FILE, "w", encoding="utf-8") as f:
                    f.write(current_code)
                    
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(30)
        
        time.sleep(2)

if __name__ == "__main__":
    main()
