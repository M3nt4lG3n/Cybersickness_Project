"""
Common/pipeline.py

Orchestrates one Experiment end-to-end:

    DataProvider -> preprocessing -> feature_selection -> model_selection
    -> (tune best model) -> evaluation -> plotting -> save results

Matches the architecture in ML_PIPELINE.md:
    Raw Data -> Existing Processing -> Processed CSVs -> DatasetLoader
    -> DataProviders -> Experiment -> Pipeline -> Results
"""

import joblib
import pandas as pd

from Config import config
from Common import evaluation, feature_selection, model_selection, plotting, preprocessing
from Common.experiment import Experiment
from Common.utils import get_logger

logger = get_logger("pipeline")


class Pipeline:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def run(self, tune: bool = True, make_plots: bool = True) -> Experiment:
        exp = self.experiment
        logger.info(f"=== Running experiment: {exp.name} ===")

        # 1. Load
        X, y, groups = exp.provider.get_data(target_column=exp.target_column)
        if X.empty or y is None:
            raise ValueError(
                f"No usable data for experiment '{exp.name}' "
                f"(check Patient_Data contents and target_column='{exp.target_column}')."
            )
        exp.raw_data = X.assign(**{exp.target_column: y, config.GROUP_COLUMN: groups})

        # Subjective/SSQ-style targets are typically recorded once per
        # session while sensor rows are per-timestamp, so the target column
        # is frequently NaN for most rows. Those rows can't be trained on or
        # scored against, so drop them here rather than imputing a label.
        target_missing = y.isna()
        if target_missing.any():
            logger.info(
                f"Dropping {int(target_missing.sum())} / {len(y)} rows with a "
                f"missing target_column ('{exp.target_column}')."
            )
            X, y, groups = X[~target_missing], y[~target_missing], groups[~target_missing]
        if X.empty:
            raise ValueError(
                f"No rows remain for experiment '{exp.name}' after dropping "
                f"missing targets - check that '{exp.target_column}' is "
                "actually populated somewhere in the data."
            )

        # 2. Preprocess: impute -> encode categoricals -> feature-select
        X = preprocessing.impute(X)
        X = preprocessing.encode_categoricals(X)
        X = feature_selection.select_features(X)

        # 3. Group split (by PatientID, never leaking a patient across sets)
        X_train, X_test, y_train, y_test, groups_train, _ = preprocessing.group_train_test_split(
            X, y, groups
        )
        exp.X_train, exp.X_test, exp.y_train, exp.y_test = X_train, X_test, y_train, y_test

        # 4. Compare candidate models (grouped CV on the training set)
        cv = list(preprocessing.group_kfold_splits(X_train, y_train, groups_train))
        leaderboard = model_selection.compare_models(
            X_train, y_train, cv=cv, models=exp.models, task_type=exp.task_type
        )
        exp.leaderboard = leaderboard
        if leaderboard.empty:
            raise RuntimeError("All candidate models failed during comparison.")
        exp.best_model_name = leaderboard.iloc[0]["model"]

        models = exp.models or (
            model_selection.default_regressors()
            if exp.task_type == "regression"
            else model_selection.default_classifiers()
        )
        best_model = models[exp.best_model_name]

        # 5. Fit (optionally tuned) best model on the full training set
        if tune:
            param_grid = _default_param_grid(exp.best_model_name)
            if param_grid:
                best_model, exp.best_params = model_selection.tune_model(
                    best_model, param_grid, X_train, y_train, cv=cv
                )
            else:
                best_model.fit(X_train, y_train)
        else:
            best_model.fit(X_train, y_train)
        exp.best_model = best_model

        # 6. Evaluate on held-out (unseen-patient) test set
        exp.test_results = evaluation.evaluate(best_model, X_test, y_test, task_type=exp.task_type)

        # 7. Plots + persistence
        if make_plots:
            plotting.plot_model_comparison(leaderboard, name=f"{exp.name}_model_comparison")
            plotting.plot_feature_importance(best_model, X_train.columns, name=f"{exp.name}_feature_importance")
            if exp.task_type == "classification":
                plotting.plot_confusion_matrix(
                    exp.test_results["confusion_matrix"], name=f"{exp.name}_confusion_matrix"
                )

        self._save_results()
        exp.save_summary()
        logger.info(f"=== Finished experiment: {exp.name} ===")
        return exp

    def _save_results(self):
        exp = self.experiment
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if exp.leaderboard is not None:
            exp.leaderboard.to_csv(config.RESULTS_DIR / f"{exp.name}_leaderboard.csv", index=False)
        if exp.test_results is not None:
            evaluation.results_to_frame(exp.test_results).to_csv(
                config.RESULTS_DIR / f"{exp.name}_test_metrics.csv", index=False
            )

        config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if exp.best_model is not None:
            model_path = config.SAVED_MODELS_DIR / f"{exp.name}_{exp.best_model_name}.joblib"
            joblib.dump(exp.best_model, model_path)
            logger.info(f"Saved model: {model_path}")


def _default_param_grid(model_name: str) -> dict:
    grids = {
        "RandomForestRegressor": {"n_estimators": [200, 400], "max_depth": [None, 8, 16]},
        "RandomForestClassifier": {"n_estimators": [200, 400], "max_depth": [None, 8, 16]},
        "GradientBoostingRegressor": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
        "GradientBoostingClassifier": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
        "SVR": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        "SVC": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    }
    return grids.get(model_name, {})