"""
experiment.py

Core data structures for the Cybersickness ML framework.

Every pipeline stage operates on an Experiment instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

@dataclass(slots=True)
class ExperimentMetadata:
    """General information describing one experiment."""

    name: str

    modality: str

    timestamp: datetime = field(default_factory=datetime.now)

    random_seed: int = 42

    notes: str = ""

    verbose: bool = True

    save_results: bool = True

    save_figures: bool = True

    save_models: bool = True

@dataclass(slots=True)
class DatasetState:
    """
    Stores the dataset as it moves through the pipeline.
    """

    raw: pd.DataFrame | None = None

    processed: pd.DataFrame | None = None

    features: pd.DataFrame | None = None

    target: pd.Series | None = None

    X_train: pd.DataFrame | None = None

    X_test: pd.DataFrame | None = None

    y_train: pd.Series | None = None

    y_test: pd.Series | None = None

    feature_names: list[str] = field(default_factory=list)

@dataclass(slots=True)
class FeatureState:

    selected_features: list[str] = field(default_factory=list)

    rankings: pd.DataFrame | None = None

    permutation_importance: pd.DataFrame | None = None

    shap_values: Any = None

@dataclass(slots=True)
class ModelState:

    candidate_models: dict[str, Any] = field(default_factory=dict)

    scores: dict[str, float] = field(default_factory=dict)

    best_model_name: str = ""

    best_model: Any = None

    best_parameters: dict = field(default_factory=dict)

@dataclass(slots=True)
class EvaluationState:

    metrics: dict[str, float] = field(default_factory=dict)

    predictions = None

    probabilities = None

    confusion_matrix = None

    roc_curve = None

@dataclass(slots=True)
class Experiment:
    """
    Master object passed between every pipeline stage.
    """

    metadata: ExperimentMetadata

    dataset: DatasetState = field(default_factory=DatasetState)

    features: FeatureState = field(default_factory=FeatureState)

    model: ModelState = field(default_factory=ModelState)

    evaluation: EvaluationState = field(default_factory=EvaluationState)

    output_directory: Path | None = None

    runtime: RuntimeState = field(default_factory=RuntimeState)

@dataclass(slots=True)
class RuntimeState:

    stage_times: dict[str, float] = field(default_factory=dict)

    total_runtime: float = 0.0