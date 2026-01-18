import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV
import copy

logger = logging.getLogger(__name__)

class Aggregator:
    def __init__(self, aggregator_id, clients, model_type='log_reg', random_seed=42):
        self.aggregator_id = aggregator_id
        self.clients = clients
        self.model_type = model_type
        self.random_seed = random_seed
        self.global_params = None
        self.calibrated_model = None # For GNB
        
        logger.info(f"Aggregator {self.aggregator_id} initialized for model: {self.model_type}")

    def aggregate(self, client_params_list):
        """
        Polymorphic aggregation based on model type.
        """
        if not client_params_list or client_params_list[0] is None:
            return None

        if self.model_type == 'log_reg':
            avg_coef = np.mean([p['coef'] for p in client_params_list], axis=0)
            avg_intercept = np.mean([p['intercept'] for p in client_params_list], axis=0)
            return {'coef': avg_coef, 'intercept': avg_intercept}
            
        elif self.model_type == 'gnb':
            # Weighted average of means and variances based on class counts
            total_counts = np.sum([p['class_count'] for p in client_params_list], axis=0)
            avg_theta = np.zeros_like(client_params_list[0]['theta'])
            avg_var = np.zeros_like(client_params_list[0]['var'])
            
            for p in client_params_list:
                weight = (p['class_count'] / total_counts)[:, np.newaxis]
                avg_theta += p['theta'] * weight
                avg_var += p['var'] * weight
                
            return {
                'theta': avg_theta,
                'var': avg_var,
                'class_count': total_counts,
                'class_prior': total_counts / np.sum(total_counts)
            }
            
        elif self.model_type == 'rf':
            # Collect all trees from all clients into one big forest
            all_trees = []
            for p in client_params_list:
                all_trees.extend(p['estimators'])
            return {'estimators': all_trees}
            
        return None

    def train_federated(self, rounds=5, local_epochs=1, X_val=None, y_val=None):
        """
        Runs the federated learning protocol.
        """
        # For RF and GNB, we might only need one round or handle differently
        actual_rounds = rounds if self.model_type == 'log_reg' else 1
        
        for r in range(actual_rounds):
            client_updates = [client.train(epochs=local_epochs) for client in self.clients]
            self.global_params = self.aggregate(client_updates)
            
            for client in self.clients:
                client.set_parameters(self.global_params)
        
        # Post-aggregation: Handle Calibration for GNB if validation data is provided
        if self.model_type == 'gnb' and X_val is not None:
            logger.info(f"Aggregator {self.aggregator_id} - Calibrating GaussianNB...")
            base_gnb = self.clients[0].model
            self.calibrated_model = CalibratedClassifierCV(base_gnb, cv='prefit', method='sigmoid')
            self.calibrated_model.fit(X_val, y_val)

        logger.info(f"Aggregator {self.aggregator_id} federated training finished.")
        return self.global_params

    def predict_proba(self, X):
        """
        Inference using the aggregated global model.
        """
        if self.model_type == 'gnb' and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.clients[0].predict_proba(X)
