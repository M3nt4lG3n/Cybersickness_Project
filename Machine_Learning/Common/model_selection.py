"""
Common/model_selection.py

Per ML_PIPELINE.md principle: "Compare models before hyperparameter tuning."

compare_models() runs several candidate models through GroupKFold cross
validation (grouped by PatientID) and returns a leaderboard. tune_model()
then hyperparameter-tunes only the winner.
"""

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from Config import config
from Common.utils import get_logger

logger = get_logger("model_selection")


def default_regressors() -> Dict[str, BaseEstimator]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    return {
        "Ridge": Ridge(random_state=config.RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        "SVR": SVR(),
    }


def default_classifiers() -> Dict[str, BaseEstimator]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        "SVC": SVC(probability=True, random_state=config.RANDOM_STATE),
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    models: Optional[Dict[str, BaseEstimator]] = None,
    task_type: str = "regression",
    scoring: Optional[str] = None,
) -> pd.DataFrame:
    """Cross-validate each candidate model and return a leaderboard sorted
    best-first. `cv` should be an iterable of (train_idx, test_idx), e.g.
    list(preprocessing.group_kfold_splits(X, y, groups))."""
    if models is None:
        models = default_regressors() if task_type == "regression" else default_classifiers()
    if scoring is None:
        scoring = "neg_root_mean_squared_error" if task_type == "regression" else "balanced_accuracy"

    cv = list(cv)
    rows = []
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            rows.append({
                "model": name,
                "scoring": scoring,
                "mean_score": scores.mean(),
                "std_score": scores.std(),
            })
            logger.info(f"{name}: {scoring}={scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as exc:
            logger.warning(f"Model '{name}' failed during CV: {exc}")

    leaderboard_cols = ["model", "scoring", "mean_score", "std_score"]
    if not rows:
        logger.error(
            "All candidate models failed during cross-validation - see the "
            "warnings above for the underlying error from each model."
        )
        return pd.DataFrame(columns=leaderboard_cols)

    leaderboard = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
    return leaderboard


def tune_model(
    model: BaseEstimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: Optional[str] = None,
) -> Tuple[BaseEstimator, dict]:
    """Hyperparameter-tune a single (already-chosen) model via GridSearchCV."""
    search = GridSearchCV(
        clone(model), param_grid=param_grid, cv=list(cv), scoring=scoring, n_jobs=-1, refit=True
    )
    search.fit(X, y)
    logger.info(f"Best params for {model.__class__.__name__}: {search.best_params_} "
                f"(score={search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_