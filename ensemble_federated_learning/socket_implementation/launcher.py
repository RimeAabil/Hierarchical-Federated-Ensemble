import subprocess
import time
import sys
import os

# Configuration
PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

META_SCRIPT = os.path.join(BASE_DIR, "meta_server.py")
AGG_SCRIPT = os.path.join(BASE_DIR, "aggregator_node.py")
CLIENT_SCRIPT = os.path.join(BASE_DIR, "client_node.py")

AGG_CONFIGS = [
    {"id": 0, "type": "gnb", "port": 6000},
    {"id": 1, "type": "log_reg", "port": 6001},
    {"id": 2, "type": "rf", "port": 6002}
]

CLIENTS_PER_AGG = 2

processes = []

def start_process(cmd, name):
    print(f"[Launcher] Starting {name}...")
    # Use Popen to run mostly in background, piping output could be messy if all shared.
    # For debugging, let them inherit stdout/stderr or pipe to files.
    # We will let them output to console for now so user can see activity.
    p = subprocess.Popen(cmd) 
    processes.append(p)
    return p

try:
    # 1. Start Meta Server
    start_process([PYTHON_EXE, META_SCRIPT], "Meta Server")
    time.sleep(2) # Give it time to bind

    # 2. Start Aggregators
    for agg in AGG_CONFIGS:
        cmd = [PYTHON_EXE, AGG_SCRIPT, str(agg["id"]), agg["type"], str(agg["port"]), str(CLIENTS_PER_AGG)]
        start_process(cmd, f"Aggregator {agg['id']}")
    
    time.sleep(2) # Wait for aggregators to connect to meta and bind own ports

    # 3. Start Clients
    # Client IDs 0-1 -> Agg 0
    # Client IDs 2-3 -> Agg 1
    # Client IDs 4-5 -> Agg 2
    client_id_counter = 0
    for agg in AGG_CONFIGS:
        for _ in range(CLIENTS_PER_AGG):
            cmd = [PYTHON_EXE, CLIENT_SCRIPT, str(client_id_counter), str(agg["port"]), agg["type"]]
            start_process(cmd, f"Client {client_id_counter} -> Agg {agg['id']}")
            client_id_counter += 1
            
    print("[Launcher] All processes started. Waiting for completion...")
    
    # Wait for Meta Server to finish (it shuts down everything else)
    # Actually Meta Server process is index 0
    processes[0].wait()
    print("[Launcher] Meta Server finished.")

except KeyboardInterrupt:
    print("[Launcher] Interrupted. Terminating all processes...")
finally:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    print("[Launcher] Clean exit.")
