"""
balance.py

Balance board processing utilities for the LabScribe
cybersickness analysis pipeline.

Responsibilities:

    • Force sensor calculations
    • Center of pressure calculations
    • COP smoothing
    • Sway metrics
    • Balance summary statistics

This module does not handle:

    • Video rendering
    • OpenCV
    • ECG processing


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd



# ============================================================
# Constants
# ============================================================

DEFAULT_SENSOR_DISTANCE_X = 1.0
DEFAULT_SENSOR_DISTANCE_Y = 1.0


COP_SMOOTH_WINDOW = 15



# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class BalanceFrame:
    """
    Represents one balance board sample.
    """

    timestamp_ms: int

    tl: float
    tr: float
    bl: float
    br: float

    total_weight: float

    cop_x: float
    cop_y: float



@dataclass(slots=True)
class BalanceAnalysisResult:
    """
    Complete balance processing output.
    """

    dataframe: pd.DataFrame

    summary: dict



# ============================================================
# Helper Functions
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe division.

    Returns NaN if denominator is zero.
    """

    if denominator == 0:
        return np.nan

    return numerator / denominator



def calculate_total_weight(
    tl,
    tr,
    bl,
    br,
):
    """
    Calculate total force.
    """

    return (
        tl +
        tr +
        bl +
        br
    )



def normalize_percentage(
    value,
    total,
):
    """
    Convert a force value into a percentage.
    """

    return (
        safe_divide(
            value,
            total,
        )
        * 100.0
    )



def rolling_smooth(
    values: pd.Series,
    window: int,
) -> pd.Series:
    """
    Centered rolling average.

    Used only for visualization.

    Raw values remain unchanged.
    """

    return (
        values
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
    )

# ============================================================
# Force and COP Calculations
# ============================================================

def calculate_cop(
    tl: float,
    tr: float,
    bl: float,
    br: float,
) -> tuple[float, float]:
    """
    Calculate center of pressure.

    Coordinate system:

        Y+
        ^
        |
    TL ------ TR
    |        |
    |        |
    BL ------ BR
        |
        v
        Y-


    Returns
    -------
    COP_X
    COP_Y

    """

    total = calculate_total_weight(
        tl,
        tr,
        bl,
        br,
    )


    if total == 0:
        return (
            np.nan,
            np.nan,
        )


    # Left/right coordinate

    cop_x = (
        (-tl - bl + tr + br)
        /
        total
    )


    # Front/back coordinate

    cop_y = (
        (tl + tr - bl - br)
        /
        total
    )


    return (
        cop_x,
        cop_y,
    )



def calculate_force_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate balance board force metrics.

    Required columns:

        TL
        TR
        BL
        BR


    Adds:

        TotalWeight

        LeftWeight
        RightWeight

        FrontWeight
        BackWeight

        LeftPercent
        RightPercent

        FrontPercent
        BackPercent

        COP_X
        COP_Y
    """

    required = [
        "TL",
        "TR",
        "BL",
        "BR",
    ]


    missing = [
        c for c in required
        if c not in df.columns
    ]


    if missing:

        raise ValueError(
            f"Missing balance columns: {missing}"
        )


    result = df.copy()


    # --------------------------------------------------------
    # Total force
    # --------------------------------------------------------

    result["TotalWeight"] = (
        result["TL"] +
        result["TR"] +
        result["BL"] +
        result["BR"]
    )


    # --------------------------------------------------------
    # Left / right
    # --------------------------------------------------------

    result["LeftWeight"] = (
        result["TL"] +
        result["BL"]
    )


    result["RightWeight"] = (
        result["TR"] +
        result["BR"]
    )


    result["LeftPercent"] = (
        result["LeftWeight"]
        /
        result["TotalWeight"]
        *
        100.0
    )


    result["RightPercent"] = (
        result["RightWeight"]
        /
        result["TotalWeight"]
        *
        100.0
    )


    # --------------------------------------------------------
    # Front / back
    # --------------------------------------------------------

    result["FrontWeight"] = (
        result["TL"] +
        result["TR"]
    )


    result["BackWeight"] = (
        result["BL"] +
        result["BR"]
    )


    result["FrontPercent"] = (
        result["FrontWeight"]
        /
        result["TotalWeight"]
        *
        100.0
    )


    result["BackPercent"] = (
        result["BackWeight"]
        /
        result["TotalWeight"]
        *
        100.0
    )


    # --------------------------------------------------------
    # COP
    # --------------------------------------------------------

    cop_values = result.apply(
        lambda row:
            calculate_cop(
                row["TL"],
                row["TR"],
                row["BL"],
                row["BR"],
            ),

        axis=1,
    )


    result["COP_X"] = [
        value[0]
        for value in cop_values
    ]


    result["COP_Y"] = [
        value[1]
        for value in cop_values
    ]


    return result



# ============================================================
# COP Smoothing
# ============================================================

def smooth_cop(
    df: pd.DataFrame,
    window: int = COP_SMOOTH_WINDOW,
) -> pd.DataFrame:
    """
    Add smoothed COP values.

    IMPORTANT:

        COP_X and COP_Y remain unchanged.

        Smoothed values are only used
        by visualization.

    Adds:

        COP_X_Smooth
        COP_Y_Smooth
    """

    result = df.copy()


    result["COP_X_Smooth"] = rolling_smooth(
        result["COP_X"],
        window,
    )


    result["COP_Y_Smooth"] = rolling_smooth(
        result["COP_Y"],
        window,
    )


    return result



# ============================================================
# Balance Processing Pipeline
# ============================================================

def process_balance_dataframe(
    df: pd.DataFrame,
    cop_window: int = COP_SMOOTH_WINDOW,
) -> pd.DataFrame:
    """
    Complete balance preprocessing pipeline.

    Performs:

        1. Force calculations
        2. COP calculations
        3. COP smoothing

    """

    result = calculate_force_metrics(
        df,
    )


    result = smooth_cop(
        result,
        cop_window,
    )


    return result

# ============================================================
# COP Sway Analysis
# ============================================================

def calculate_cop_path_length(
    cop_x: pd.Series,
    cop_y: pd.Series,
) -> float:
    """
    Calculate total COP travel distance.

    Uses raw COP values.

    Units are the same as the COP coordinate system.
    """

    x = np.asarray(
        cop_x,
        dtype=float,
    )

    y = np.asarray(
        cop_y,
        dtype=float,
    )


    if len(x) < 2:
        return 0.0


    dx = np.diff(x)

    dy = np.diff(y)


    distances = np.sqrt(
        dx ** 2 +
        dy ** 2
    )


    return float(
        np.nansum(distances)
    )



def calculate_cop_velocity(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate instantaneous COP velocity.

    Uses:

        distance / time

    Time is derived from UnixTime_ms.

    Returns
    -------
    pandas Series
        COP velocity per sample.
    """

    if "UnixTime_ms" not in df.columns:

        raise ValueError(
            "UnixTime_ms required for COP velocity."
        )


    dx = df["COP_X"].diff()

    dy = df["COP_Y"].diff()


    distance = np.sqrt(
        dx ** 2 +
        dy ** 2
    )


    dt = (
        df["UnixTime_ms"]
        .diff()
        /
        1000.0
    )


    velocity = (
        distance /
        dt
    )


    velocity.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )


    return velocity



def calculate_max_sway_distance(
    cop_x: pd.Series,
    cop_y: pd.Series,
) -> float:
    """
    Calculate maximum distance from the mean COP position.
    """

    mean_x = np.nanmean(
        cop_x
    )

    mean_y = np.nanmean(
        cop_y
    )


    distances = np.sqrt(
        (cop_x - mean_x) ** 2
        +
        (cop_y - mean_y) ** 2
    )


    return float(
        np.nanmax(distances)
    )



def calculate_sway_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add COP sway metrics.

    Adds:

        COP_Velocity
    """

    result = df.copy()


    result["COP_Velocity"] = (
        calculate_cop_velocity(
            result
        )
    )


    return result



# ============================================================
# Balance Summary Statistics
# ============================================================

def compute_balance_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate balance summary statistics.

    Returns:

        RecordingLength

        AverageWeight
        WeightStd

        AverageCOPX
        AverageCOPY

        COPStdX
        COPStdY

        MaxSwayDistance

        MeanSwayVelocity

        TotalCOPPathLength
    """

    summary = {}


    # --------------------------------------------------------
    # Recording duration
    # --------------------------------------------------------

    if "UnixTime_ms" in df.columns:

        duration = (
            df["UnixTime_ms"].iloc[-1]
            -
            df["UnixTime_ms"].iloc[0]
        )

        summary["RecordingLength"] = (
            float(duration)
            /
            1000.0
        )

    else:

        summary["RecordingLength"] = np.nan



    # --------------------------------------------------------
    # Weight statistics
    # --------------------------------------------------------

    summary["AverageWeight"] = float(
        df["TotalWeight"]
        .mean()
    )


    summary["WeightStd"] = float(
        df["TotalWeight"]
        .std()
    )


    # --------------------------------------------------------
    # COP position statistics
    # --------------------------------------------------------

    summary["AverageCOPX"] = float(
        df["COP_X"]
        .mean()
    )


    summary["AverageCOPY"] = float(
        df["COP_Y"]
        .mean()
    )


    summary["COPStdX"] = float(
        df["COP_X"]
        .std()
    )


    summary["COPStdY"] = float(
        df["COP_Y"]
        .std()
    )


    # --------------------------------------------------------
    # Sway metrics
    # --------------------------------------------------------

    summary["MaxSwayDistance"] = (
        calculate_max_sway_distance(
            df["COP_X"],
            df["COP_Y"],
        )
    )


    if "COP_Velocity" in df.columns:

        summary["MeanSwayVelocity"] = float(
            df["COP_Velocity"]
            .mean()
        )

    else:

        summary["MeanSwayVelocity"] = np.nan



    summary["TotalCOPPathLength"] = (
        calculate_cop_path_length(
            df["COP_X"],
            df["COP_Y"],
        )
    )


    return summary



# ============================================================
# Public Balance Analysis Function
# ============================================================

def analyze_balance(
    df: pd.DataFrame,
    cop_window: int = COP_SMOOTH_WINDOW,
) -> BalanceAnalysisResult:
    """
    Complete balance processing pipeline.

    Returns:

        BalanceAnalysisResult
    """

    processed = process_balance_dataframe(
        df,
        cop_window,
    )


    processed = calculate_sway_metrics(
        processed,
    )


    summary = compute_balance_summary(
        processed,
    )


    return BalanceAnalysisResult(

        dataframe=processed,

        summary=summary,
    )

# ============================================================
# COP Sway Analysis
# ============================================================

def calculate_cop_path_length(
    cop_x: pd.Series,
    cop_y: pd.Series,
) -> float:
    """
    Calculate total COP travel distance.

    Uses raw COP values.

    Units are the same as the COP coordinate system.
    """

    x = np.asarray(
        cop_x,
        dtype=float,
    )

    y = np.asarray(
        cop_y,
        dtype=float,
    )


    if len(x) < 2:
        return 0.0


    dx = np.diff(x)

    dy = np.diff(y)


    distances = np.sqrt(
        dx ** 2 +
        dy ** 2
    )


    return float(
        np.nansum(distances)
    )



def calculate_cop_velocity(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate instantaneous COP velocity.

    Uses:

        distance / time

    Time is derived from UnixTime_ms.

    Returns
    -------
    pandas Series
        COP velocity per sample.
    """

    if "UnixTime_ms" not in df.columns:

        raise ValueError(
            "UnixTime_ms required for COP velocity."
        )


    dx = df["COP_X"].diff()

    dy = df["COP_Y"].diff()


    distance = np.sqrt(
        dx ** 2 +
        dy ** 2
    )


    dt = (
        df["UnixTime_ms"]
        .diff()
        /
        1000.0
    )


    velocity = (
        distance /
        dt
    )


    velocity.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )


    return velocity



def calculate_max_sway_distance(
    cop_x: pd.Series,
    cop_y: pd.Series,
) -> float:
    """
    Calculate maximum distance from the mean COP position.
    """

    mean_x = np.nanmean(
        cop_x
    )

    mean_y = np.nanmean(
        cop_y
    )


    distances = np.sqrt(
        (cop_x - mean_x) ** 2
        +
        (cop_y - mean_y) ** 2
    )


    return float(
        np.nanmax(distances)
    )



def calculate_sway_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add COP sway metrics.

    Adds:

        COP_Velocity
    """

    result = df.copy()


    result["COP_Velocity"] = (
        calculate_cop_velocity(
            result
        )
    )


    return result



# ============================================================
# Balance Summary Statistics
# ============================================================

def compute_balance_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate balance summary statistics.

    Returns:

        RecordingLength

        AverageWeight
        WeightStd

        AverageCOPX
        AverageCOPY

        COPStdX
        COPStdY

        MaxSwayDistance

        MeanSwayVelocity

        TotalCOPPathLength
    """

    summary = {}


    # --------------------------------------------------------
    # Recording duration
    # --------------------------------------------------------

    if "UnixTime_ms" in df.columns:

        duration = (
            df["UnixTime_ms"].iloc[-1]
            -
            df["UnixTime_ms"].iloc[0]
        )

        summary["RecordingLength"] = (
            float(duration)
            /
            1000.0
        )

    else:

        summary["RecordingLength"] = np.nan



    # --------------------------------------------------------
    # Weight statistics
    # --------------------------------------------------------

    summary["AverageWeight"] = float(
        df["TotalWeight"]
        .mean()
    )


    summary["WeightStd"] = float(
        df["TotalWeight"]
        .std()
    )


    # --------------------------------------------------------
    # COP position statistics
    # --------------------------------------------------------

    summary["AverageCOPX"] = float(
        df["COP_X"]
        .mean()
    )


    summary["AverageCOPY"] = float(
        df["COP_Y"]
        .mean()
    )


    summary["COPStdX"] = float(
        df["COP_X"]
        .std()
    )


    summary["COPStdY"] = float(
        df["COP_Y"]
        .std()
    )


    # --------------------------------------------------------
    # Sway metrics
    # --------------------------------------------------------

    summary["MaxSwayDistance"] = (
        calculate_max_sway_distance(
            df["COP_X"],
            df["COP_Y"],
        )
    )


    if "COP_Velocity" in df.columns:

        summary["MeanSwayVelocity"] = float(
            df["COP_Velocity"]
            .mean()
        )

    else:

        summary["MeanSwayVelocity"] = np.nan



    summary["TotalCOPPathLength"] = (
        calculate_cop_path_length(
            df["COP_X"],
            df["COP_Y"],
        )
    )


    return summary



# ============================================================
# Public Balance Analysis Function
# ============================================================

def analyze_balance(
    df: pd.DataFrame,
    cop_window: int = COP_SMOOTH_WINDOW,
) -> BalanceAnalysisResult:
    """
    Complete balance processing pipeline.

    Returns:

        BalanceAnalysisResult
    """

    processed = process_balance_dataframe(
        df,
        cop_window,
    )


    processed = calculate_sway_metrics(
        processed,
    )


    summary = compute_balance_summary(
        processed,
    )


    return BalanceAnalysisResult(

        dataframe=processed,

        summary=summary,
    )

# ============================================================
# Visualization Force Normalization
# ============================================================

def calculate_visualization_force_scale(
    df: pd.DataFrame,
) -> float:
    """
    Calculate a consistent force scaling value.

    Uses:

        average_total_weight = median(TotalWeight)

        expected_corner =
            average_total_weight / 4

        maximum_force =
            expected_corner * 1.8


    This makes visualization comparable between recordings.

    Returns
    -------
    maximum_force
    """

    if "TotalWeight" not in df.columns:

        raise ValueError(
            "TotalWeight column required."
        )


    average_total_weight = np.median(
        df["TotalWeight"]
    )


    expected_corner = (
        average_total_weight /
        4.0
    )


    maximum_force = (
        expected_corner *
        1.8
    )


    return float(
        maximum_force
    )



def add_visualization_force_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized force values for rendering.

    Adds:

        MaximumForceScale

        TL_Normalized
        TR_Normalized
        BL_Normalized
        BR_Normalized

    Values are between 0 and 1.
    """

    result = df.copy()


    maximum_force = (
        calculate_visualization_force_scale(
            result
        )
    )


    result["MaximumForceScale"] = (
        maximum_force
    )


    for sensor in [
        "TL",
        "TR",
        "BL",
        "BR",
    ]:

        result[
            f"{sensor}_Normalized"
        ] = (
            result[sensor]
            /
            maximum_force
        )


        result[
            f"{sensor}_Normalized"
        ] = (
            result[
                f"{sensor}_Normalized"
            ]
            .clip(
                lower=0,
                upper=1,
            )
        )


    return result



# ============================================================
# Export Helpers
# ============================================================

def export_balance_analysis(
    result: BalanceAnalysisResult,
    output_file: str,
) -> None:
    """
    Export processed balance dataframe.

    Creates:

        *_balance_analysis.csv
    """

    result.dataframe.to_csv(
        output_file,
        index=False,
    )



def export_balance_summary(
    result: BalanceAnalysisResult,
    output_file: str,
) -> None:
    """
    Export balance summary.

    Creates:

        *_balance_summary.csv
    """

    summary_df = pd.DataFrame(
        [
            result.summary
        ]
    )


    summary_df.to_csv(
        output_file,
        index=False,
    )



# ============================================================
# Updated Analysis Wrapper
# ============================================================

def analyze_balance(
    df: pd.DataFrame,
    cop_window: int = COP_SMOOTH_WINDOW,
) -> BalanceAnalysisResult:
    """
    Complete balance analysis pipeline.

    Processing order:

        1. Calculate force metrics
        2. Calculate raw COP
        3. Add COP smoothing
        4. Calculate sway metrics
        5. Add visualization scaling

    """

    processed = process_balance_dataframe(
        df,
        cop_window,
    )


    processed = calculate_sway_metrics(
        processed,
    )


    processed = add_visualization_force_columns(
        processed,
    )


    summary = compute_balance_summary(
        processed,
    )


    return BalanceAnalysisResult(

        dataframe=processed,

        summary=summary,
    )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Dataclasses

    "BalanceFrame",
    "BalanceAnalysisResult",


    # Main pipeline

    "analyze_balance",
    "process_balance_dataframe",


    # Calculations

    "calculate_force_metrics",
    "calculate_cop",
    "smooth_cop",

    "calculate_cop_path_length",
    "calculate_cop_velocity",
    "calculate_sway_metrics",

    "compute_balance_summary",


    # Visualization helpers

    "calculate_visualization_force_scale",
    "add_visualization_force_columns",


    # Export

    "export_balance_analysis",
    "export_balance_summary",
]