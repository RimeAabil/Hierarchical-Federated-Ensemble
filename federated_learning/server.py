import numpy as np
from model import FLModel

class Server:
    def __init__(self, random_seed=42):
        self.global_model = FLModel(random_seed=random_seed)
        
    def aggregate(self, updates):
        """
        FedAvg aggregation.
        updates: list of (weights, num_samples)
        """
        if not updates:
            return None
            
        # Filter invalid updates
        valid_updates = [u for u in updates if u[0] is not None]
        if not valid_updates:
            return None
            
        total_samples = sum([u[1] for u in valid_updates])
        
        # Initialize zero weights based on shape of first update
        first_weights = valid_updates[0][0]
        avg_coef = np.zeros_like(first_weights['coef'])
        avg_intercept = np.zeros_like(first_weights['intercept'])
        
        for weights, n_samples in valid_updates:
            weight_factor = n_samples / total_samples
            avg_coef += weights['coef'] * weight_factor
            avg_intercept += weights['intercept'] * weight_factor
            
        return {'coef': avg_coef, 'intercept': avg_intercept}
        
    def update_global_model(self, serialized_params):
        self.global_model.set_params(serialized_params)
        
    def evaluate(self, X_test, y_test):
        preds = self.global_model.predict(X_test)
        acc = np.mean(preds == y_test)
        return acc
