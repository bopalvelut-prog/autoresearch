#!/usr/bin/env python3
"""
AutoResearch Coordinator - The 'Brain' of the LAN Swarm.
Mixture of Exo (auto-discovery) and prima.cpp (dashboard/efficiency).
"""

import os
import json
import time
import re
import socket
import threading
from typing import Dict, List, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from zeroconf import ServiceInfo, Zeroconf
import requests
import subprocess

# --- Config ---
MODEL = "qwen2.5:0.5b"
PORT = 8000
RESULTS_FILE = "cluster_results.tsv"
TRAIN_FILE = "train.py"

def get_lan_ip():
    """Robustly find the LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

app = FastAPI(title="AutoResearch Coordinator")

# --- Ensure Dashboard Template Exists ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoResearch Swarm</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }
        .card { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #444; }
        .best { font-size: 2em; color: #4caf50; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #444; }
        .status-keep { color: #4caf50; font-weight: bold; }
        .status-discard { color: #888; }
        .worker-tag { display: inline-block; background: #444; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
    </style>
</head>
<body>
    <h1>AutoResearch Swarm Dashboard</h1>
    
    <div class="card">
        <h2>Global Best BPB</h2>
        <div class="best">{{ best_bpb }}</div>
        <p>Total Experiments: {{ total_experiments }}</p>
    </div>

    <div class="card">
        <h2>Active Workers</h2>
        <table>
            <tr><th>Hostname</th><th>ID</th><th>Status</th><th>Last Seen</th></tr>
            {% for id, w in workers.items() %}
            <tr>
                <td>{{ w.hostname }}</td>
                <td><span class="worker-tag">{{ id }}</span></td>
                <td>{{ w.status }}</td>
                <td>{{ w.last_seen }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h2>Recent Experiments</h2>
        <table>
            <tr><th>Time</th><th>BPB</th><th>Status</th><th>Description</th></tr>
            {% for e in experiments %}
            <tr class="status-{{ e.status }}">
                <td>{{ e.time }}</td>
                <td>{{ e.bpb }}</td>
                <td>{{ e.status }}</td>
                <td>{{ e.desc }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

if not os.path.exists("dashboard.html"):
    with open("dashboard.html", "w") as f:
        f.write(DASHBOARD_HTML)

templates = Jinja2Templates(directory=".") 

# --- State ---
class ClusterState:
    def __init__(self):
        self.best_bpb = 999.0
        self.experiments: List[Dict] = []
        self.active_workers: Dict[str, Dict] = {}
        self.pending_tasks: List[Dict] = []
        self.lock = threading.Lock()
        
        # Load existing results if any
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE, "r") as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            bpb = float(parts[1])
                            self.best_bpb = min(self.best_bpb, bpb)
                            self.experiments.append({
                                "time": parts[0],
                                "bpb": bpb,
                                "status": parts[2],
                                "desc": parts[3] if len(parts)>3 else ""
                            })
            except: pass

state = ClusterState()

# --- LLM Brain (Background Task) ---
def task_generator_loop():
    """Continuously generate suggestions in the background to avoid blocking API."""
    while True:
        with state.lock:
            # Only generate if the queue is small
            if len(state.pending_tasks) >= 5:
                time.sleep(10)
                continue
        
        print("[Brain] Consulting LLM for new hyperparameters...")
        prompt = f"Best BPB: {state.best_bpb}. Suggest ONE change for MATRIX_LR, EMBEDDING_LR, SCALAR_LR, or WEIGHT_DECAY. Output ONLY JSON. Example: {{\"MATRIX_LR\": 0.041}}"
        new_task = None
        try:
            full_prompt = f"System: AI Researcher. Output ONLY JSON.\nUser: {prompt}"
            result = subprocess.run(
                ['ollama', 'run', MODEL, full_prompt],
                capture_output=True, text=True, timeout=120, errors='replace'
            )
            match = re.search(r"\{.*\}", result.stdout.replace("\n", ""))
            if match:
                new_task = json.loads(match.group())
        except Exception as e:
            print(f"Ollama error: {e}")
        
        if not new_task:
            new_task = {"MATRIX_LR": 0.04 + (time.time() % 0.01)}
            
        with state.lock:
            state.pending_tasks.append(new_task)
            print(f"[Brain] New task added to queue: {new_task}")
        
        time.sleep(2)

# --- mDNS Advertising ---
def start_mdns():
    zeroconf = Zeroconf()
    hostname = socket.gethostname()
    local_ip = get_lan_ip()
    info = ServiceInfo(
        "_autoresearch._tcp.local.",
        f"{hostname}._autoresearch._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=PORT,
        properties={"version": "1.0"},
    )
    print(f"Broadcasting {hostname}.local at {local_ip} on LAN...")
    zeroconf.register_service(info)
    return zeroconf

# --- API Endpoints ---
class TaskReport(BaseModel):
    worker_id: str
    bpb: float
    params: Dict
    hardware: Optional[Dict] = None

@app.get("/task")
async def get_task(worker_id: str, hostname: str = "unknown"):
    with state.lock:
        # Register/Update worker
        state.active_workers[worker_id] = {
            "hostname": hostname,
            "last_seen": datetime.now().strftime("%H:%M:%S"),
            "status": "training"
        }
        
        # Get task from pre-filled queue
        if not state.pending_tasks:
            print(f"Queue empty for {hostname}, providing default.")
            return {"MATRIX_LR": 0.041}
        
        task = state.pending_tasks.pop(0)
        print(f"Assigned task to {hostname}: {task}")
        return task

@app.post("/report")
async def report_result(report: TaskReport):
    with state.lock:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "keep" if report.bpb < state.best_bpb else "discard"
        if report.bpb < state.best_bpb:
            state.best_bpb = report.bpb
            print(f"!!! NEW BEST: {state.best_bpb} from {report.worker_id}")
        
        desc = ", ".join([f"{k}={v}" for k,v in report.params.items()])
        state.experiments.append({
            "time": ts,
            "bpb": report.bpb,
            "status": status,
            "desc": desc,
            "worker": report.worker_id
        })
        
        # Log to file
        file_exists = os.path.exists(RESULTS_FILE)
        with open(RESULTS_FILE, "a") as f:
            if not file_exists: f.write("time\tbpb\tstatus\tdesc\n")
            f.write(f"{ts}\t{report.bpb:.6f}\t{status}\t{desc}\n")
            
        state.active_workers[report.worker_id]["status"] = "idle"
        state.active_workers[report.worker_id]["last_seen"] = datetime.now().strftime("%H:%M:%S")
        
    return {"status": "ok", "best": state.best_bpb}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "best_bpb": state.best_bpb,
        "workers": state.active_workers,
        "experiments": state.experiments[-20:][::-1], # Last 20
        "total_experiments": len(state.experiments)
    })

if __name__ == "__main__":
    # Start task generator thread
    threading.Thread(target=task_generator_loop, daemon=True).start()
    
    zc = start_mdns()
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT)
    finally:
        zc.unregister_all_services()
        zc.close()
