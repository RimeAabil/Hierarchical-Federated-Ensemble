from model import FLModel
import numpy as np

class Client:
    def __init__(self, client_id, X_train, y_train, random_seed=42):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = FLModel(random_seed=random_seed)
        
    def train(self, rounds=1):
        """Train local model."""
        # For LogisticRegression in sklearn, valid 'epochs' conceptually maps to max_iter or partial_fit.
        # Since we are doing FedAvg, we usually train for a few epochs locally.
        # Standard LR .fit() converges fully. We will just .fit() on local data.
        if len(np.unique(self.y_train)) < 2:
            # Cannot fit with 1 class
            return None, 0
            
        self.model.fit(self.X_train, self.y_train)
        return self.model.get_params(), len(self.X_train)
        
    def set_weights(self, weights):
        """Update local model with global weights."""
        self.model.set_params(weights)
        
    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = np.mean(preds == y_test)
        return acc
