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

# SET PROCESS PRIORITY TO BELOW NORMAL
p = psutil.Process(os.getpid())
if os.name == 'nt':
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
else:
    p.nice(19) 

def unload_model():
    """Tells Ollama to unload the model to free up RAM."""
    try:
        requests.post("http://localhost:11434/api/generate", 
                      json={"model": MODEL, "keep_alive": 0}, timeout=5)
    except:
        pass

def chat(prompt):
    """Reliable CLI-based chat."""
    full_prompt = f"System: AI Researcher. Output ONLY JSON.\nUser: {prompt}"
    try:
        result = subprocess.run(
            ['ollama', 'run', MODEL, full_prompt],
            capture_output=True, text=True, timeout=300
        )
        # Immediately tell ollama we are done for now
        unload_model()
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
        return "latest"

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
    print("=== AutoResearch: FOLDING EDITION (Low Memory Mode) ===")
    
    # Ensure we start clean
    subprocess.run("git restore train.py", shell=True)
    
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
    print(f"Current best: {best_bpb}. Starting research loop...")

    while True:
        try:
            gc.collect() # Aggressive cleanup
            
            # 1. Suggest change
            prompt = f"Best BPB: {best_bpb}. Suggest ONE change for MATRIX_LR, EMBEDDING_LR, SCALAR_LR, or WEIGHT_DECAY. Output ONLY JSON. Example: {{\"MATRIX_LR\": 0.041}}"
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
                    new_code = re.sub(fr"({var}\s*=\s*)[\d\.\*e\-]+", fr"\g<1>{val}", new_code)
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
                    subprocess.run(f'git commit -am "Improve to {new_bpb} via {desc}"', shell=True)
                    subprocess.run('git push mine master', shell=True)
                    log_result("latest", new_bpb, "keep", desc)
                else:
                    print(f"Discarding {new_bpb}")
                    log_result("latest", new_bpb, "discard", desc)
                    subprocess.run("git restore train.py", shell=True)
            else:
                print("Crash detected.")
                log_result("latest", 0.0, "crash", desc)
                subprocess.run("git restore train.py", shell=True)
            
            # Cleanup after training run
            del output
            gc.collect()
                    
        except Exception as e:
            print(f"Loop error: {e}")
            subprocess.run("git restore train.py", shell=True)
            time.sleep(30)
        
        time.sleep(5)

if __name__ == "__main__":
    main()
