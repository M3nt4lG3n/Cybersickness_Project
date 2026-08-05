"""Common/evaluation.py - held-out test-set evaluation, using the metric
definitions from metrics.py."""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from Common.metrics import classification_metrics, regression_metrics
from Common.utils import get_logger

logger = get_logger("evaluation")


def evaluate_regressor(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    results = regression_metrics(y_test, y_pred)
    logger.info(f"Regression evaluation: {results}")
    return {"metrics": results, "y_pred": y_pred}


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    results = classification_metrics(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Classification evaluation: {results}")
    return {"metrics": results, "y_pred": y_pred, "report": report, "confusion_matrix": cm}


def evaluate(model, X_test, y_test, task_type: str = "regression") -> dict:
    if task_type == "regression":
        return evaluate_regressor(model, X_test, y_test)
    return evaluate_classifier(model, X_test, y_test)


def results_to_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame([results["metrics"]])
