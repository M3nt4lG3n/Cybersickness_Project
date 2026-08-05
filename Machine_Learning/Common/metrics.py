"""Common/metrics.py - metric helpers shared by model_selection.py and
evaluation.py, so scoring definitions live in exactly one place."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
