import socket
import sys
import threading
import logging
import pickle
import numpy as np

# Adjust path to import from parent directory
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from meta_learner.stacking_model import MetaLearner
from socket_implementation.comms import send_msg, recv_msg
from evaluation.metrics import compute_metrics, display_comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Meta] - %(message)s')
logger = logging.getLogger(__name__)

META_PORT = 5000
NUM_AGGREGATORS = 3

class AggregatorProxy:
    """Represents a connected Aggregator on the Meta Server side."""
    def __init__(self, conn, addr, agg_id):
        self.conn = conn
        self.addr = addr
        self.agg_id = agg_id
        
    def request_prediction(self, X):
        """Request prediction from this aggregator."""
        send_msg(self.conn, {"type": "PREDICT", "data": X})
        response = recv_msg(self.conn)
        return response["probs"]
        
    def request_train(self):
        """Trigger federated training on this aggregator and its clients."""
        logger.info(f"Triggering training on Aggregator {self.agg_id}")
        send_msg(self.conn, {"type": "TRAIN"})
        response = recv_msg(self.conn) # Wait for confirmation
        return response

class MetaServer:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', META_PORT))
        self.server_socket.listen(NUM_AGGREGATORS)
        self.aggregators = []
        self.meta_learner_model = MetaLearner(random_seed=42)
        
    def start(self):
        logger.info(f"Meta Server listening on port {META_PORT}...")
        
        # 1. Wait for Aggregators to Connect
        while len(self.aggregators) < NUM_AGGREGATORS:
            conn, addr = self.server_socket.accept()
            # Expect handshake
            msg = recv_msg(conn)
            if msg['type'] == 'HANDSHAKE':
                agg_id = msg['agg_id']
                logger.info(f"Aggregator {agg_id} connected from {addr}")
                self.aggregators.append(AggregatorProxy(conn, addr, agg_id))
            else:
                conn.close()
                
        # Sort aggregators by ID just to be deterministic
        self.aggregators.sort(key=lambda x: x.agg_id)
        
        logger.info("All aggregators connected. Starting Hierarchical Training.")
        
        # 2. Trigger Federated Training
        # In a real sync FL, we might iterate rounds. Here we trust Aggregators to do their FL loop.
        for agg in self.aggregators:
            agg.request_train()
            
        logger.info("Federated Training Phase Completed.")
        
        # 3. Meta-Training
        # We need validation data. For simulation, we'll assume we have it locally or 
        # normally we'd receive predictions on validation set from aggregators.
        # To make this simple and robust, let's load the data here again (or pass it in).
        # Ideally, we should receive the *predictions* on the validation set from aggregators.
        
        # Request validation predictions on held-out set
        # For simplicity in this demo, we load the same dataset and split.
        from data.data_loader import load_and_preprocess_data
        _, _, X_val, y_val, X_test, y_test = load_and_preprocess_data(42)
        
        logger.info("Collecting validation predictions for Stacking...")
        
        # Get predictions from all aggregators
        agg_val_preds = []
        for agg in self.aggregators:
            probs = agg.request_prediction(X_val)
            agg_val_preds.append(probs)
            
        # Combine for meta-learner (feature matrix: [n_samples, n_aggs * n_classes] or just n_aggs if binary prob)
        # Using the logic from stacking_model.py
        # We need to adapt MetaLearner to accept pre-computed predictions if possible,
        # or we manually construct the input matrix.
        
        # Let's see how stacking_model.py works.
        # It creates a StackingClassifier. We might abuse it or implement manual logic.
        # Actually, standard StackingClassifier expects estimators. 
        # We are mocking a simpler meta-learner here: LogisticRegression on top of Aggregator outputs.
        
        X_meta_train = np.hstack(agg_val_preds) # [Results from Agg1, Results from Agg2...]
        
        logger.info(f"Training Meta-Learner on stacked features shape: {X_meta_train.shape}")
        
        self.meta_learner_model.meta_model.fit(X_meta_train, y_val)
        
        # 4. Final Evaluation
        logger.info("Evaluating on Test Set...")
        agg_test_preds = []
        comparison_results = []
        
        for i, agg in enumerate(self.aggregators):
            probs = agg.request_prediction(X_test)
            agg_test_preds.append(probs)
            
            # Evaluate individual aggregator
            preds = np.argmax(probs, axis=1)
            res = compute_metrics(y_test, preds, model_name=f"Aggregator_{agg.agg_id}")
            res['name'] = f"Aggregator_{agg.agg_id}"
            comparison_results.append(res)
            
        # Stacked Inference
        X_meta_test = np.hstack(agg_test_preds)
        final_preds = self.meta_learner_model.meta_model.predict(X_meta_test)
        
        final_res = compute_metrics(y_test, final_preds, model_name="Stacked_Meta_Learner")
        final_res['name'] = "Stacked_Meta_Learner"
        comparison_results.append(final_res)
        
        display_comparison(comparison_results)
        
        # Shutdown
        logger.info("Training Complete. Shutting down.")
        for agg in self.aggregators:
            send_msg(agg.conn, {"type": "SHUTDOWN"})
            
if __name__ == "__main__":
    server = MetaServer()
    server.start()
