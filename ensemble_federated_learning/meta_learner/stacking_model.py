from sklearn.linear_model import LogisticRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(self, random_seed=42):
        self.meta_model = LogisticRegression(random_state=random_seed)
        self.aggregators = []

    def set_aggregators(self, aggregators):
        """Sets the trained aggregators (ensemble base learners)."""
        self.aggregators = aggregators

    def prepare_meta_features(self, X):
        """
        Collects predictions from all aggregators to create the meta-features.
        Each aggregator provides class probabilities.
        """
        meta_features = []
        for agg in self.aggregators:
            # Get probability of class 1
            probs = agg.predict_proba(X)[:, 1].reshape(-1, 1)
            meta_features.append(probs)
        
        # Stack horizontally: [num_samples, num_aggregators]
        return np.concatenate(meta_features, axis=1)

    def train_meta_learner(self, X_val, y_val):
        """
        Trains the meta-model on the validation set predictions.
        """
        logger.info("Training Meta-Learner (Stacking) on validation data predictions...")
        
        meta_features = self.prepare_meta_features(X_val)
        self.meta_model.fit(meta_features, y_val)
        
        logger.info("Meta-Learner training complete.")

    def predict_proba(self, X):
        """
        Performs final ensemble prediction using the stacked model.
        """
        meta_features = self.prepare_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        """
        Performs final ensemble prediction (class labels).
        """
        meta_features = self.prepare_meta_features(X)
        return self.meta_model.predict(meta_features)
