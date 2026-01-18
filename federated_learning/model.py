from sklearn.linear_model import LogisticRegression
import numpy as np

class FLModel:
    def __init__(self, random_seed=42):
        # We use standard Logistic Regression
        self.model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_seed)
        self.random_seed = random_seed

    def get_params(self):
        """Return model parameters (coef, intercept)."""
        if not hasattr(self.model, 'coef_'):
            return None
        return {
            'coef': self.model.coef_,
            'intercept': self.model.intercept_
        }

    def set_params(self, params):
        """Set model parameters."""
        self.model.coef_ = params['coef']
        self.model.intercept_ = params['intercept']
        # Helper to ensure sklearn sees it as fitted
        if not hasattr(self.model, 'classes_'):
            self.model.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
