import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import pandas as pd

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute a comprehensive set of metrics for classification.
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

    # Per-class F1
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(f1_per)):
        metrics[f"class_{i}_f1"] = f1_per[i]

    # ROC-AUC if probabilities are provided
    if y_probs is not None:
        try:
            # Handle binary vs multiclass
            if len(np.unique(y_true)) <= 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1] if y_probs.ndim > 1 else y_probs)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except Exception as e:
            metrics["roc_auc"] = None

    return metrics

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
