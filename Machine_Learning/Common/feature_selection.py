"""Common/feature_selection.py - simple, transparent feature-pruning steps
run before model comparison (per ML_PIPELINE.md: compare models before
hyperparameter tuning - keep feature selection equally lightweight/explicit)."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from Common.utils import get_logger

logger = get_logger("feature_selection")


def drop_zero_variance(X: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return X
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric.fillna(numeric.median()))
    keep = numeric.columns[selector.get_support()]
    drop = set(numeric.columns) - set(keep)
    if drop:
        logger.info(f"Dropping {len(drop)} zero/low-variance columns: {sorted(drop)}")
    non_numeric = [c for c in X.columns if c not in numeric.columns]
    return X[list(keep) + non_numeric]


def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return X
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if drop:
        logger.info(f"Dropping {len(drop)} highly-correlated columns (>{threshold}): {drop}")
    return X.drop(columns=drop)


def select_features(X: pd.DataFrame, variance_threshold: float = 0.0, corr_threshold: float = 0.95) -> pd.DataFrame:
    X = drop_zero_variance(X, variance_threshold)
    X = drop_highly_correlated(X, corr_threshold)
    return X
