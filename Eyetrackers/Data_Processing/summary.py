"""
summary.py

Summary generation utilities for the LabScribe
cybersickness analysis pipeline.

Responsibilities:

    • Combine ECG and balance summaries
    • Generate experiment-level metrics
    • Export final summary CSV


This module does not handle:

    • ECG processing
    • Balance calculations
    • Video rendering
    • CSV loading


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd



# ============================================================
# Constants
# ============================================================

SUMMARY_VERSION = "1.0"



# ============================================================
# Dataclass
# ============================================================

@dataclass(slots=True)
class ExperimentSummary:
    """
    Combined experiment summary.

    Stores:

        ECG metrics

        Balance metrics

        Recording metadata
    """

    metrics: dict

    version: str = SUMMARY_VERSION



# ============================================================
# Validation Helpers
# ============================================================

def validate_summary_dict(
    summary: dict,
) -> None:
    """
    Validate summary dictionary.
    """

    if summary is None:

        raise ValueError(
            "Summary cannot be None."
        )


    if not isinstance(
        summary,
        dict,
    ):

        raise TypeError(
            "Summary must be a dictionary."
        )



def clean_summary_values(
    summary: dict,
) -> dict:
    """
    Convert unsupported values before CSV export.

    Converts:

        numpy values
        infinite values

    Keeps:

        NaN values
    """

    cleaned = {}


    for key, value in summary.items():

        if isinstance(
            value,
            np.generic,
        ):

            value = value.item()


        if isinstance(
            value,
            float,
        ):

            if np.isinf(value):

                value = np.nan


        cleaned[key] = value


    return cleaned

# ============================================================
# Summary Combination
# ============================================================

def merge_summaries(
    ecg_summary: dict | None = None,
    balance_summary: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Combine ECG, balance, and recording metadata.

    Priority order:

        metadata
        balance
        ECG

    Existing keys are overwritten by later sources.
    """

    combined = {}


    if metadata:

        validate_summary_dict(
            metadata
        )

        combined.update(
            metadata
        )


    if ecg_summary:

        validate_summary_dict(
            ecg_summary
        )

        combined.update(
            ecg_summary
        )


    if balance_summary:

        validate_summary_dict(
            balance_summary
        )

        combined.update(
            balance_summary
        )


    return clean_summary_values(
        combined
    )



# ============================================================
# Experiment Summary Creation
# ============================================================

def calculate_combined_summary(
    ecg_result=None,
    balance_result=None,
    metadata: dict | None = None,
) -> ExperimentSummary:
    """
    Create final experiment summary.

    Parameters
    ----------
    ecg_result

        ECGAnalysisResult from ecg.py.


    balance_result

        BalanceAnalysisResult from balance.py.


    metadata

        Additional recording information.

    """

    ecg_summary = None

    balance_summary = None


    # --------------------------------------------------------
    # Extract ECG summary
    # --------------------------------------------------------

    if ecg_result is not None:

        ecg_summary = (
            ecg_result.summary
        )


    # --------------------------------------------------------
    # Extract balance summary
    # --------------------------------------------------------

    if balance_result is not None:

        balance_summary = (
            balance_result.summary
        )


    metrics = merge_summaries(
        ecg_summary,
        balance_summary,
        metadata,
    )


    return ExperimentSummary(

        metrics=metrics,
    )



# ============================================================
# DataFrame Conversion
# ============================================================

def summary_to_dataframe(
    summary: ExperimentSummary | dict,
) -> pd.DataFrame:
    """
    Convert summary object into dataframe.

    Produces one-row dataframe suitable
    for CSV export.
    """

    if isinstance(
        summary,
        ExperimentSummary,
    ):

        values = summary.metrics.copy()

        values["SummaryVersion"] = (
            summary.version
        )

    else:

        values = summary.copy()


    values = clean_summary_values(
        values
    )


    return pd.DataFrame(
        [values]
    )



# ============================================================
# CSV Export
# ============================================================

def export_summary(
    summary: ExperimentSummary | dict,
    output_file: str | Path,
) -> None:
    """
    Export final summary CSV.

    Output:

        *_summary.csv
    """

    dataframe = summary_to_dataframe(
        summary
    )


    dataframe.to_csv(
        output_file,
        index=False,
    )

# ============================================================
# Required Summary Metrics
# ============================================================

REQUIRED_SUMMARY_FIELDS = [

    # Recording

    "RecordingLength",


    # ECG

    "AverageHeartRate",

    "MinimumHeartRate",

    "MaximumHeartRate",

    "HeartRateStd",

    "TotalHeartBeats",


    # Balance

    "AverageWeight",

    "WeightStd",

    "AverageCOPX",

    "AverageCOPY",

    "COPStdX",

    "COPStdY",

    "MaxSwayDistance",

    "MeanSwayVelocity",

    "TotalCOPPathLength",
]



# ============================================================
# Summary Completion
# ============================================================

def ensure_summary_fields(
    summary: dict,
    required_fields: list[str] | None = None,
) -> dict:
    """
    Ensure required metrics exist.

    Missing metrics are added as NaN.

    This prevents export failures when:
        - ECG detection fails
        - Balance data is missing
        - Partial recordings occur
    """

    validate_summary_dict(
        summary
    )


    if required_fields is None:

        required_fields = (
            REQUIRED_SUMMARY_FIELDS
        )


    completed = summary.copy()


    for field in required_fields:

        if field not in completed:

            completed[field] = np.nan


    return completed



# ============================================================
# Enhanced Summary Creation
# ============================================================

def finalize_summary(
    summary: ExperimentSummary,
) -> ExperimentSummary:
    """
    Apply final validation and cleanup.
    """

    metrics = ensure_summary_fields(
        summary.metrics
    )


    metrics = clean_summary_values(
        metrics
    )


    return ExperimentSummary(

        metrics=metrics,

        version=summary.version,
    )



# ============================================================
# Summary Comparison
# ============================================================

def compare_summaries(
    summaries: list[ExperimentSummary],
) -> pd.DataFrame:
    """
    Convert multiple experiment summaries into
    a comparison dataframe.

    Useful for:

        multiple patients

        baseline vs VR trials

        recovery comparisons
    """

    rows = []


    for summary in summaries:

        finalized = finalize_summary(
            summary
        )


        rows.append(
            finalized.metrics
        )


    return pd.DataFrame(
        rows
    )



# ============================================================
# Final Export Wrapper
# ============================================================

def export_final_summary(
    summary: ExperimentSummary,
    output_file: str | Path,
) -> None:
    """
    Final recommended summary exporter.

    Performs:

        1. Validation
        2. Missing field insertion
        3. CSV export
    """

    finalized = finalize_summary(
        summary
    )


    export_summary(
        finalized,
        output_file,
    )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Dataclass

    "ExperimentSummary",


    # Creation

    "calculate_combined_summary",

    "merge_summaries",

    "finalize_summary",


    # Conversion

    "summary_to_dataframe",


    # Export

    "export_summary",

    "export_final_summary",


    # Comparison

    "compare_summaries",


    # Validation

    "ensure_summary_fields",

    "validate_summary_dict",
]