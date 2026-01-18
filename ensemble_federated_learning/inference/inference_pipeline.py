import logging
import numpy as np

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, aggregators, meta_learner):
        """
        aggregators: List of trained Aggregator objects.
        meta_learner: Trained MetaLearner object.
        """
        self.aggregators = aggregators
        self.meta_learner = meta_learner

    def run_inference(self, X):
        """
        Executes the hierarchical inference protocol:
        1. X -> Aggregators (Base Learners)
        2. Predictions -> Meta-Learner
        3. Final Output
        """
        # logger.info("Running hierarchical inference...")
        
        # In this system, the meta_learner already knows how to 
        # collect predictions from its aggregators.
        final_probs = self.meta_learner.predict_proba(X)
        final_preds = self.meta_learner.predict(X)
        
        return final_preds, final_probs

    def evaluate_aggregators(self, X, y):
        """
        Helper to evaluate individual aggregators for comparison.
        """
        agg_results = []
        for i, agg in enumerate(self.aggregators):
            probs = agg.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            agg_results.append((f"Aggregator_{i}", preds))
        return agg_results
