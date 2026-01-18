from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Computes and logs classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
    
    logger.info(f"--- {model_name} Evaluation ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return results

def display_comparison(results_list):
    """
    Displays a comparison table of results.
    results_list: List of dictionaries with 'name' and metric values.
    """
    df = pd.DataFrame(results_list)
    df.set_index('name', inplace=True)
    logger.info("\n=== Comparative Performance ===")
    logger.info("\n" + df.to_string())
    return df
