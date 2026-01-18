import logging
import random
import numpy as np
from data.data_loader import load_and_preprocess_data, partition_data_non_iid
from clients.client import Client
from aggregators.aggregator import Aggregator
from meta_learner.stacking_model import MetaLearner
from evaluation.metrics import compute_metrics, display_comparison
from inference.inference_pipeline import InferencePipeline

# Set random seeds for determinism
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Hierarchical Federated Ensemble Learning with Meta-Stacking ===")
    
    # 1. Data Preparation
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(RANDOM_SEED)
    
    num_aggregators = 3
    clients_per_aggregator = 2
    total_clients = num_aggregators * clients_per_aggregator
    
    client_data_shards = partition_data_non_iid(X_train, y_train, total_clients, RANDOM_SEED)
    
    # 2. Initialize Hierarchical Components
    aggregators = []
    
    # Define heterogeneous model types for each aggregator
    model_types = ['gnb', 'log_reg', 'rf']
    
    for agg_id in range(num_aggregators):
        m_type = model_types[agg_id]
        agg_clients = []
        for c_idx in range(clients_per_aggregator):
            client_id = agg_id * clients_per_aggregator + c_idx
            X_c, y_c = client_data_shards[client_id]
            client = Client(client_id, X_c, y_c, model_type=m_type)
            agg_clients.append(client)
        
        aggregator = Aggregator(agg_id, agg_clients, model_type=m_type, random_seed=RANDOM_SEED)
        aggregators.append(aggregator)
        
    meta_learner = MetaLearner(RANDOM_SEED)
    meta_learner.set_aggregators(aggregators)
    
    # 3. Execution Phase - Level 2: Federated Learning
    for agg in aggregators:
        # Pass X_val, y_val for calibration (specifically for GNB)
        agg.train_federated(rounds=10, local_epochs=5, X_val=X_val, y_val=y_val)
        
    # 4. Execution Phase - Level 3: Meta-Training (Stacking)
    meta_learner.train_meta_learner(X_val, y_val)
    
    # 5. Inference and Evaluation
    inference_pipeline = InferencePipeline(aggregators, meta_learner)
    
    comparison_results = []
    
    # Evaluate Individual Aggregators (Base Learners)
    for i, agg in enumerate(aggregators):
        probs = agg.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
        res = compute_metrics(y_test, preds, model_name=f"Aggregator_{i}")
        res['name'] = f"Aggregator_{i}"
        comparison_results.append(res)
        
    # Evaluate Final Stacked Model
    final_preds, final_probs = inference_pipeline.run_inference(X_test)
    final_res = compute_metrics(y_test, final_preds, model_name="Stacked_Meta_Learner")
    final_res['name'] = "Stacked_Meta_Learner"
    comparison_results.append(final_res)
    
    # Display final comparison table
    display_comparison(comparison_results)
    
    logger.info("Project executed successfully.")

if __name__ == "__main__":
    main()
