import socket
import sys
import threading
import logging
import pickle
import numpy as np

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aggregators.aggregator import Aggregator
from clients.client import Client
from socket_implementation.comms import send_msg, recv_msg
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Aggregator] - %(message)s')
logger = logging.getLogger(__name__)

META_HOST = 'localhost'
META_PORT = 5000

class ClientProxy:
    def __init__(self, conn, addr, client_id):
        self.conn = conn
        self.addr = addr
        self.client_id = client_id
        
    def set_weights(self, weights):
        send_msg(self.conn, {"type": "SET_WEIGHTS", "weights": weights})
        
    def train(self):
        send_msg(self.conn, {"type": "TRAIN"})
        resp = recv_msg(self.conn)
        return resp["weights"] # Assuming client returns this dictionary

class AggregatorNode:
    def __init__(self, agg_id, model_type, port, num_clients):
        self.agg_id = agg_id
        self.model_type = model_type
        self.port = port
        self.num_clients = num_clients
        self.clients = []
        
        # Logic component
        # We pass an empty list of clients initially.
        self.aggregator_logic = Aggregator(agg_id, [], model_type=model_type)
        
        # Initialize a shadow global model for inference
        if model_type == 'log_reg':
            self.global_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
            # Init dummy coef/intercept to avoid NotFittedError if needed, 
            # but ideally we get weights after first round.
            # Loading data to get shape would be best, but we'll wait for first aggregation.
            
        elif model_type == 'gnb':
            self.global_model = GaussianNB()
            
        elif model_type == 'rf':
            self.global_model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('localhost', self.port))
        self.server_sock.listen(self.num_clients)
        
        logger.info(f"Aggregator {self.agg_id} ({self.model_type}) listening for clients on {self.port}...")
        
        # 2. Connect to Meta Server
        self.meta_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.meta_conn.connect((META_HOST, META_PORT))
            send_msg(self.meta_conn, {"type": "HANDSHAKE", "agg_id": self.agg_id})
            logger.info(f"Connected to Meta Server.")
        except ConnectionRefusedError:
            logger.error("Could not connect to Meta Server.")
            return

        # 3. Accept Clients
        while len(self.clients) < self.num_clients:
            conn, addr = self.server_sock.accept()
            msg = recv_msg(conn)
            if msg['type'] == 'HANDSHAKE':
                c_id = msg['client_id']
                logger.info(f"Client {c_id} connected.")
                self.clients.append(ClientProxy(conn, addr, c_id))
        
        logger.info("All clients connected.")
        
        # 4. Command Loop
        running = True
        while running:
            msg = recv_msg(self.meta_conn)
            if not msg:
                break
                
                
            elif msg['type'] == 'PREDICT':
                X = msg['data']
                # Use local shadow model
                # We need to ensure it's fitted or has attributes set.
                try:
                    probs = self.global_model.predict_proba(X)
                    send_msg(self.meta_conn, {"type": "PREDICT_RESULT", "probs": probs})
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    # Return dummy zero probs if failed
                    send_msg(self.meta_conn, {"type": "PREDICT_RESULT", "probs": np.zeros((len(X), 2))})
                
            elif msg['type'] == 'SHUTDOWN':
                logger.info("Received SHUTDOWN command.")
                running = False
            
        # Cleanup
        for c in self.clients:
            send_msg(c.conn, {"type": "SHUTDOWN"})
        self.server_sock.close()
        self.meta_conn.close()
        
    def run_federated_round(self):
        rounds = 1 
        
        for r in range(rounds):
            logger.info(f" -- Round {r+1}/{rounds} --")
            
            # 1. Train Clients
            collected_weights = []
            for client in self.clients:
                # Send current global params if we have them? 
                # For first round, clients might start from scratch or own init.
                # Simplification: Clients train, we aggregate.
                
                w = client.train()
                collected_weights.append(w)
                
            # 2. Aggregate
            global_params = self.aggregator_logic.aggregate(collected_weights)
            
            # 3. Update Shadow Model (for inference) & Broadcast to Clients
            if self.model_type == 'log_reg':
                self.global_model.coef_ = global_params['coef']
                self.global_model.intercept_ = global_params['intercept']
                # Hack to fake "classes_" if not set, required for predict_proba
                if not hasattr(self.global_model, 'classes_'):
                    self.global_model.classes_ = np.array([0, 1])
                    
            elif self.model_type == 'gnb':
                 # GNB aggregation returns theta, var, class_prior usually
                 # We need to map these back to GaussianNB attributes
                 self.global_model.theta_ = global_params['theta']
                 self.global_model.var_ = global_params['var']
                 self.global_model.class_prior_ = global_params['class_prior']
                 self.global_model.class_count_ = global_params['class_count']
                 if not hasattr(self.global_model, 'classes_'):
                    self.global_model.classes_ = np.array([0, 1])
                 if not hasattr(self.global_model, 'epsilon_'):
                     self.global_model.epsilon_ = 1e-9

            elif self.model_type == 'rf':
                # RF aggregation returns list of estimators
                self.global_model.estimators_ = global_params['estimators']
                # Need to set n_outputs_ and classes_
                self.global_model.n_outputs_ = 1
                self.global_model.classes_ = np.array([0, 1])
                # Ensure all trees are fitted? They should be coming from clients.

            
            # 4. Broadcast back to clients
            for client in self.clients:
                client.set_weights(global_params)
                
            logger.info(f"Round {r+1} Aggregation complete.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python aggregator_node.py <ID> <TYPE> <PORT> <NUM_CLIENTS>")
        sys.exit(1)
        
    agg_id = int(sys.argv[1])
    m_type = sys.argv[2]
    port = int(sys.argv[3])
    n_clients = int(sys.argv[4])
    
    node = AggregatorNode(agg_id, m_type, port, n_clients)
    node.start()
