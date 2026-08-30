"""
Project : InsightCart

File : evaluate_models.py

Purpose :
Utility functions for metrics calculation and persistence.
"""

import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    return metrics


def save_metrics(metrics: dict, out_path: str = "artifacts/model_metrics.json") -> str:
    """Save metrics dict to a JSON file (creates parent dir if needed)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return out_path
