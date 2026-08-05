"""
Common/experiment.py

Per ML_PIPELINE.md: "Experiment is the central state object."

An Experiment bundles everything that defines one run - which data
provider/modality, which target column, which task type, which candidate
models - and accumulates results as pipeline.py executes each stage. This
keeps Models/*_ML.py scripts declarative (a few lines each) and keeps every
run's config + results reproducible and inspectable together.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from Config import config
from Common.data_provider import BaseDataProvider
from Common.utils import get_logger

logger = get_logger("experiment")


@dataclass
class Experiment:
    name: str
    provider: BaseDataProvider
    target_column: Optional[str] = None
    task_type: str = "regression"          # "regression" | "classification"
    models: Optional[Dict] = None           # None -> model_selection defaults
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # populated as the pipeline runs
    raw_data: Optional[pd.DataFrame] = None
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    leaderboard: Optional[pd.DataFrame] = None
    best_model_name: Optional[str] = None
    best_model = None
    best_params: Optional[dict] = None
    test_results: Optional[dict] = None

    def summary(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider.__class__.__name__,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "n_rows": None if self.raw_data is None else len(self.raw_data),
            "best_model": self.best_model_name,
            "best_params": self.best_params,
            "test_metrics": None if self.test_results is None else self.test_results.get("metrics"),
        }

    def save_summary(self):
        out_dir = config.RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}_summary.csv"
        pd.DataFrame([self.summary()]).to_csv(out_path, index=False)
        logger.info(f"Saved experiment summary: {out_path}")
        return out_path
