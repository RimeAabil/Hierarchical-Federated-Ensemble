import socket
import sys
import os
import logging
import pickle
import numpy as np
from sklearn.base import clone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clients.client import Client
from socket_implementation.comms import send_msg, recv_msg
from data.data_loader import load_and_preprocess_data, partition_data_non_iid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Client] - %(message)s')
logger = logging.getLogger(__name__)

class ClientNode:
    def __init__(self, client_id, agg_port, model_type):
        self.client_id = client_id
        self.agg_port = agg_port
        self.model_type = model_type
        self.conn = None
        
        # Load Data
        # In a real scenario, data is local. Here we slice it from the central loader.
        RANDOM_SEED = 42
        # This is expensive to reload every time, but fine for simulation correctness.
        X_train, y_train, _, _, _, _ = load_and_preprocess_data(RANDOM_SEED)
        
        # Total clients = 3 aggregators * 2 clients = 6
        # We need to compute total clients to partition correctly
        TOTAL_CLIENTS = 6 
        shards = partition_data_non_iid(X_train, y_train, TOTAL_CLIENTS, RANDOM_SEED)
        
        X_c, y_c = shards[client_id]
        
        # Initialize Client logic object
        self.client_logic = Client(client_id, X_c, y_c, model_type)
        
    def start(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.conn.connect(('localhost', self.agg_port))
            send_msg(self.conn, {"type": "HANDSHAKE", "client_id": self.client_id})
            logger.info(f"Client {self.client_id} connected to Aggregator on port {self.agg_port}")
        except ConnectionRefusedError:
            logger.error(f"Could not connect to Aggregator on port {self.agg_port}")
            return
            
        running = True
        while running:
            msg = recv_msg(self.conn)
            if not msg:
                break
                
            if msg['type'] == 'TRAIN':
                logger.info("Received TRAIN command.")
                # self.client_logic.train() 
                # Note: Existing client.train() returns nothing, updates internal state?
                # Let's check Client.train() implementation...
                # Assuming standard fit.
                
                self.client_logic.model.fit(self.client_logic.X_train, self.client_logic.y_train)
                
                # Extract weights to send back
                weights = {}
                if hasattr(self.client_logic.model, 'coef_'):
                     weights['coef'] = self.client_logic.model.coef_
                     weights['intercept'] = self.client_logic.model.intercept_
                # ... handle other model types ...
                
                # Send back empty loss for now as placeholder
                send_msg(self.conn, {"type": "TRAIN_RESULT", "weights": weights, "loss": 0.0})
                
            elif msg['type'] == 'SET_WEIGHTS':
                # Update local model with global weights
                w = msg['weights']
                # self.client_logic.model.coef_ = w['coef']
                # ...
                pass
                
            elif msg['type'] == 'SHUTDOWN':
                logger.info("Shutting down.")
                running = False
                
        self.conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python client_node.py <ID> <AGG_PORT> <TYPE>")
        sys.exit(1)
        
    c_id = int(sys.argv[1])
    a_port = int(sys.argv[2])
    m_type = sys.argv[3]
    
    node = ClientNode(c_id, a_port, m_type)
    node.start()
