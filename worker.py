#!/usr/bin/env python3
"""
AutoResearch Worker - The 'Muscle' of the LAN Swarm.
Uses ZeroConf to find the coordinator (Exo style).
"""

import os
import sys
import time
import uuid
import socket
import subprocess
import re
import requests
from zeroconf import ServiceBrowser, Zeroconf, ServiceListener

# --- Config ---
TRAIN_FILE = "train.py"
WORKER_ID = str(uuid.uuid4())[:8]
HOSTNAME = socket.gethostname()

class CoordinatorListener(ServiceListener):
    def __init__(self):
        self.address = None
        self.port = None

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info:
            self.address = socket.inet_ntoa(info.addresses[0])
            self.port = info.port
            print(f"Found Coordinator: {self.address}:{self.port}")

    def update_service(self, zc, type_, name):
        pass

    def remove_service(self, zc, type_, name):
        pass

def find_coordinator():
    zeroconf = Zeroconf()
    listener = CoordinatorListener()
    browser = ServiceBrowser(zeroconf, "_autoresearch._tcp.local.", listener)
    
    print("Searching for coordinator on LAN...")
    timeout = 30
    start = time.time()
    while not listener.address and (time.time() - start) < timeout:
        time.sleep(0.5)
    
    zeroconf.close()
    if listener.address:
        return f"http://{listener.address}:{listener.port}"
    return None

def run_training(params):
    """Modify train.py and run it."""
    # 1. Apply changes to train.py
    with open(TRAIN_FILE, "r") as f:
        code = f.read()
    
    new_code = code
    for var, val in params.items():
        if var in code:
            new_code = re.sub(fr"({var}\s*=\s*)[\d\.\*e\-]+", fr"\g<1>{val}", new_code)
    
    with open(TRAIN_FILE, "w") as f:
        f.write(new_code)
    
    # 2. Run
    print(f"[{time.strftime('%H:%M:%S')}] Starting experiment: {params}")
    try:
        # We assume 'uv' is installed on the worker
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True, text=True, timeout=1200, errors='replace'
        )
        output = result.stdout + result.stderr
        
        # 3. Parse result
        match = re.search(r"val_bpb:\s+([\d\.]+)", output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Training error: {e}")
    
    return None

def main():
    print(f"=== AutoResearch Worker: {WORKER_ID} ({HOSTNAME}) ===")
    
    coordinator_url = find_coordinator()
    if not coordinator_url:
        print("Could not find coordinator. Check if coordinator.py is running on your LAN.")
        sys.exit(1)
    
    print(f"Connected to swarm at {coordinator_url}")
    
    while True:
        try:
            # 1. Get task
            resp = requests.get(f"{coordinator_url}/task", params={
                "worker_id": WORKER_ID,
                "hostname": HOSTNAME
            }, timeout=30) # Increased timeout
            
            if resp.status_code != 200:
                print("Coordinator busy or error. Sleeping...")
                time.sleep(10)
                continue
                
            params = resp.json()
            
            # 2. Do work
            bpb = run_training(params)
            
            # 3. Report
            if bpb is not None:
                print(f"Finished! Result: {bpb}")
                requests.post(f"{coordinator_url}/report", json={
                    "worker_id": WORKER_ID,
                    "bpb": bpb,
                    "params": params
                }, timeout=30) # Increased timeout
            else:
                print("Training failed or crashed.")
                # Maybe report a high BPB or error status?
                
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(10)
            # Try to rediscover if connection lost
            coordinator_url = find_coordinator() or coordinator_url
            
        time.sleep(2)

if __name__ == "__main__":
    main()
