import numpy as np
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from client import Client
from server import Server

# --- Data Loading Logic (Reused from HFEL for consistency) ---
def load_and_preprocess_data(random_seed=42):
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split: Train+Val (80%) and Test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # Split Train+Val into Train (60%) and Val (20%)
    # Matches HFEL split exactly
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_seed, stratify=y_train_val
    )
    
    # For Traditional FL, we can choose to use X_train only (strict compare)
    # or X_train + X_val (since we don't need meta-validation).
    # To keep the "local data" seen by clients identical to HFEL clients, we use X_train.
    return X_train, y_train, X_test, y_test

def partition_data_non_iid(X, y, num_clients, random_seed=42):
    np.random.seed(random_seed)
    indices = np.argsort(y)
    X_sorted = X[indices]
    y_sorted = y[indices]
    
    num_shards = num_clients * 2
    shard_size = len(y) // num_shards
    
    client_data = []
    shard_indices = list(range(num_shards))
    np.random.shuffle(shard_indices)
    
    for i in range(num_clients):
        s1, s2 = shard_indices[i*2], shard_indices[i*2+1]
        start1, end1 = s1 * shard_size, (s1 + 1) * shard_size
        start2, end2 = s2 * shard_size, (s2 + 1) * shard_size
        
        X_client = np.concatenate([X_sorted[start1:end1], X_sorted[start2:end2]], axis=0)
        y_client = np.concatenate([y_sorted[start1:end1], y_sorted[start2:end2]], axis=0)
        
        client_data.append((X_client, y_client))
        
    return client_data

# --- Main Training Loop ---
def main():
    RANDOM_SEED = 42
    NUM_CLIENTS = 6 # Matches HFEL (3 aggregators * 2 clients)
    ROUNDS = 50 # Standard FedAvg rounds
    
    # 1. Prepare Data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(RANDOM_SEED)
    client_shards = partition_data_non_iid(X_train, y_train, NUM_CLIENTS, RANDOM_SEED)
    
    # 2. Initialize Nodes
    server = Server(random_seed=RANDOM_SEED)
    clients = []
    for i in range(NUM_CLIENTS):
        c_X, c_y = client_shards[i]
        clients.append(Client(i, c_X, c_y, random_seed=RANDOM_SEED))
        
    print(f"Initialized {NUM_CLIENTS} clients. Starting FedAvg for {ROUNDS} rounds...")
    
    # 3. Training Loop
    best_acc = 0.0
    
    for r in range(ROUNDS):
        # a. Collect updates
        updates = []
        for client in clients:
            # Sync with global
            current_global = server.global_model.get_params()
            if current_global is not None:
                client.set_weights(current_global)
            
            # Train (1 epoch/step per round usually, or full local fit?)
            # FedAvg: usually E epochs.
            w, n = client.train()
            updates.append((w, n))
            
        # b. Aggregate
        aggregated_weights = server.aggregate(updates)
        server.update_global_model(aggregated_weights)
        
        # c. Evaluate
        acc = server.evaluate(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            
        if (r+1) % 5 == 0:
            print(f"Round {r+1}/{ROUNDS} - Global Accuracy: {acc:.4f}")
            
    print(f"Final Best Accuracy: {best_acc:.4f}")
    
    # Save Results
    results = {
        "method": "Traditional Federated Learning",
        "accuracy": best_acc
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
