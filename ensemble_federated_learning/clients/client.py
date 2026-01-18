from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, client_id, X_train, y_train, model_type='log_reg'):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        
        if model_type == 'log_reg':
            self.model = SGDClassifier(loss='log_loss', penalty='l2', random_state=42)
        elif model_type == 'gnb':
            self.model = GaussianNB()
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def get_parameters(self):
        """Returns parameters based on model type."""
        if self.model_type == 'log_reg':
            if not hasattr(self.model, "coef_"): return None
            return {'coef': self.model.coef_.copy(), 'intercept': self.model.intercept_.copy()}
        
        elif self.model_type == 'gnb':
            if not hasattr(self.model, "theta_"): return None
            # Handle non-IID: Ensure theta and var cover both classes [0, 1]
            num_features = self.model.theta_.shape[1]
            full_theta = np.zeros((2, num_features))
            full_var = np.ones((2, num_features)) # Var defaults to 1 for unseen classes
            full_counts = np.zeros(2)
            
            for i, cls in enumerate(self.model.classes_):
                full_theta[int(cls)] = self.model.theta_[i]
                full_var[int(cls)] = self.model.var_[i]
                full_counts[int(cls)] = self.model.class_count_[i]
                
            return {
                'theta': full_theta,
                'var': full_var,
                'class_count': full_counts,
                'class_prior': full_counts / (np.sum(full_counts) + 1e-6)
            }
            
        elif self.model_type == 'rf':
            if not hasattr(self.model, "estimators_"): return None
            return {'estimators': self.model.estimators_}
            
        return None

    def set_parameters(self, params):
        """Sets parameters based on model type."""
        if params is None: return
        
        if self.model_type == 'log_reg':
            if not hasattr(self.model, "classes_"): self.model.classes_ = np.array([0, 1])
            self.model.coef_ = params['coef'].copy()
            self.model.intercept_ = params['intercept'].copy()
            
        elif self.model_type == 'gnb':
            self.model.theta_ = params['theta'].copy()
            self.model.var_ = params['var'].copy()
            self.model.class_count_ = params['class_count'].copy()
            self.model.class_prior_ = params['class_prior'].copy()
            self.model.classes_ = np.array([0, 1])
            
        elif self.model_type == 'rf':
            self.model.estimators_ = params['estimators']
            self.model.n_estimators = len(params['estimators'])
            self.model.classes_ = np.array([0, 1])
            self.model.n_classes_ = 2
            self.model.n_outputs_ = 1

    def train(self, epochs=1):
        """Trains the model locally."""
        if self.model_type == 'log_reg':
            if not hasattr(self.model, "classes_"):
                self.model.fit(self.X_train, self.y_train)
            else:
                for _ in range(epochs):
                    self.model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))
        else:
            self.model.fit(self.X_train, self.y_train)
            
        return self.get_parameters()

    def predict_proba(self, X):
        """Generates probability predictions."""
        # For RF, if it was just loaded from estimators, it might need classes defined
        if self.model_type == 'rf' and not hasattr(self.model, "classes_"):
            self.model.classes_ = np.array([0, 1])
        return self.model.predict_proba(X)
