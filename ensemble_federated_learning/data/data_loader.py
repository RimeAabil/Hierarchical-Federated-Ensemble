import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(random_seed=42):
    """
    Loads breast cancer dataset, scales features, and performs initial splits.
    """
    logger.info("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split: Train+Val (80%) and Test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # Split Train+Val into Train (80% of 80%) and Val (20% of 80%)
    # Val is used for Stacking Meta-Learner training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_seed, stratify=y_train_val
    )
    
    logger.info(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def partition_data_non_iid(X, y, num_clients, random_seed=42):
    """
    Partitions the training data into non-IID shards for clients.
    Approach: Sort by class and then divide into asymmetric shards.
    """
    np.random.seed(random_seed)
    
    # Sort data by labels to create non-IID partitions easily
    indices = np.argsort(y)
    X_sorted = X[indices]
    y_sorted = y[indices]
    
    # Divide indices into num_clients * 2 shards (each client gets 2 shards of different sizes)
    num_shards = num_clients * 2
    shard_size = len(y) // num_shards
    
    client_data = []
    shard_indices = list(range(num_shards))
    np.random.shuffle(shard_indices)
    
    for i in range(num_clients):
        # Pick two random shards
        s1, s2 = shard_indices[i*2], shard_indices[i*2+1]
        
        # Determine uneven split for non-IIDness
        start1, end1 = s1 * shard_size, (s1 + 1) * shard_size
        start2, end2 = s2 * shard_size, (s2 + 1) * shard_size
        
        X_client = np.concatenate([X_sorted[start1:end1], X_sorted[start2:end2]], axis=0)
        y_client = np.concatenate([y_sorted[start1:end1], y_sorted[start2:end2]], axis=0)
        
        # Log distribution
        unique, counts = np.unique(y_client, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"Client {i} distribution: {dist}")
        
        client_data.append((X_client, y_client))
        
    return client_data

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    partition_data_non_iid(X_train, y_train, num_clients=6)
