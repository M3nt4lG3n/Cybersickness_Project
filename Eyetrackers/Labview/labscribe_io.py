"""
labscribe_io.py

LabScribe file input/output utilities for the
cybersickness analysis pipeline.

Responsibilities:

    • Load LabScribe CSV files
    • Extract metadata
    • Identify ECG and balance channels
    • Prepare dataframe for analysis
    • Export processed CSV files


This module does not handle:

    • ECG processing
    • Balance calculations
    • Video rendering
    • Summary generation


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from .timestamps import generate_unix_timestamps

# ============================================================
# Constants
# ============================================================

DEFAULT_ECG_CHANNEL = "i1 2"

DEFAULT_BALANCE_CHANNELS = [
    "TL",
    "TR",
    "BL",
    "BR",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class LabScribeMetadata:
    """
    Metadata associated with a LabScribe recording.
    """

    file_path: Path

    file_name: str

    creation_date: object | None

    sample_count: int

    columns: list[str]



@dataclass(slots=True)
class LabScribeData:
    """
    Complete loaded LabScribe dataset.

    Contains:

        dataframe
        metadata
    """

    dataframe: pd.DataFrame

    metadata: LabScribeMetadata



# ============================================================
# Validation Helpers
# ============================================================

def validate_file_exists(
    file_path: str | Path,
) -> Path:
    """
    Verify that a CSV exists.
    """

    path = Path(
        file_path
    )


    if not path.exists():

        raise FileNotFoundError(
            f"LabScribe file not found: {path}"
        )


    if not path.is_file():

        raise ValueError(
            f"Path is not a file: {path}"
        )


    return path



def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Validate loaded LabScribe dataframe.
    """

    if df.empty:

        raise ValueError(
            "LabScribe dataframe is empty."
        )


    if len(df.columns) == 0:

        raise ValueError(
            "No columns found."
        )



def validate_required_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Verify required LabScribe columns exist.
    """

    missing = [

        column
        for column in columns
        if column not in df.columns

    ]


    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )
    
# ============================================================
# CSV Loading
# ============================================================

def load_labscribe_csv(
    file_path: str | Path,
    **kwargs,
) -> LabScribeData:
    """
    Load a LabScribe CSV file.

    Parameters
    ----------
    file_path:
        Path to LabScribe CSV.

    kwargs:
        Additional pandas.read_csv arguments.

    Returns
    -------
    LabScribeData
    """

    path = validate_file_exists(
        file_path
    )


    dataframe = pd.read_csv(
        path,
        **kwargs,
    )


    validate_dataframe(
        dataframe
    )


    metadata = extract_metadata(
        dataframe,
        path,
    )


    return LabScribeData(

        dataframe=dataframe,

        metadata=metadata,
    )



# ============================================================
# Metadata Extraction
# ============================================================

def extract_metadata(
    df: pd.DataFrame,
    file_path: str | Path,
    creation_date=None,
) -> LabScribeMetadata:
    """
    Create LabScribe metadata object.

    The creation date is intentionally optional because
    LabScribe exports vary depending on settings.

    It can be supplied manually later.
    """

    path = Path(
        file_path
    )


    if creation_date is None:

        creation_date = (
            path.stat()
            .st_mtime
        )


    return LabScribeMetadata(

        file_path=path,

        file_name=path.name,

        creation_date=creation_date,

        sample_count=len(df),

        columns=list(df.columns),
    )



# ============================================================
# Channel Detection
# ============================================================

def detect_ecg_channel(
    df: pd.DataFrame,
    preferred: str = DEFAULT_ECG_CHANNEL,
) -> str:
    """
    Find ECG channel.

    Uses the preferred LabScribe channel first.
    """

    if preferred in df.columns:

        return preferred


    # Fallback search

    possible = [

        column
        for column in df.columns
        if "ecg" in column.lower()

    ]


    if possible:

        return possible[0]


    raise ValueError(
        "Unable to locate ECG channel."
    )



def detect_balance_channels(
    df: pd.DataFrame,
    required=None,
) -> list[str]:
    """
    Locate balance board channels.

    Expected:

        TL
        TR
        BL
        BR
    """

    if required is None:

        required = DEFAULT_BALANCE_CHANNELS


    validate_required_columns(
        df,
        required,
    )


    return required



def detect_channels(
    df: pd.DataFrame,
) -> dict:
    """
    Detect all required LabScribe channels.

    Returns
    -------

    dict containing:

        ECG
        BALANCE
    """

    return {

        "ECG":
            detect_ecg_channel(df),

        "BALANCE":
            detect_balance_channels(df),

    }



# ============================================================
# Data Preparation
# ============================================================

def prepare_labscribe_dataframe(
    data: LabScribeData,
) -> pd.DataFrame:
    """
    Prepare dataframe for analysis.

    Performs:

        • Copy dataframe
        • Normalize column names
        • Remove completely empty rows

    Does not:

        • Add timestamps
        • Process ECG
        • Process balance
    """

    df = data.dataframe.copy()


    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    # --------------------------------------------------------
    # Remove empty rows
    # --------------------------------------------------------

    df.dropna(
        how="all",
        inplace=True,
    )


    df.reset_index(
        drop=True,
        inplace=True,
    )


    validate_dataframe(
        df
    )


    return df

# ============================================================
# Analysis Data Preparation
# ============================================================

def add_recording_timestamps(
    df: pd.DataFrame,
    csv_creation_date,
    recording_hour: int,
    am_pm: str,
    recording_minute: int = 0,
    recording_second: int = 0,
) -> pd.DataFrame:
    """
    Add UnixTime_ms to a LabScribe dataframe.

    Uses timestamps.py.
    """

    return generate_unix_timestamps(
        df,

        csv_creation_date,

        recording_hour,

        am_pm,

        recording_minute,

        recording_second,
    )



def prepare_analysis_dataframe(
    data: LabScribeData,
    csv_creation_date,
    recording_hour: int,
    am_pm: str,
    recording_minute: int = 0,
    recording_second: int = 0,
) -> pd.DataFrame:
    """
    Complete LabScribe preparation pipeline.

    Performs:

        1. Data cleanup
        2. Timestamp generation
        3. Channel validation


    Returns dataframe ready for:

        ecg.py

        balance.py
    """

    df = prepare_labscribe_dataframe(
        data
    )


    detect_channels(
        df
    )


    df = add_recording_timestamps(
        df,

        csv_creation_date,

        recording_hour,

        am_pm,

        recording_minute,

        recording_second,
    )


    return df



# ============================================================
# Data Extraction Helpers
# ============================================================

def get_ecg_signal(
    df: pd.DataFrame,
    channel: str = DEFAULT_ECG_CHANNEL,
) -> pd.Series:
    """
    Extract ECG signal column.
    """

    channel = detect_ecg_channel(
        df,
        channel,
    )


    return df[channel]



def get_balance_data(
    df: pd.DataFrame,
    channels=None,
) -> pd.DataFrame:
    """
    Extract balance board dataframe.

    Returns:

        UnixTime_ms

        TL
        TR
        BL
        BR
    """

    if channels is None:

        channels = DEFAULT_BALANCE_CHANNELS


    validate_required_columns(
        df,
        channels,
    )


    columns = list(channels)


    if "UnixTime_ms" in df.columns:

        columns.insert(
            0,
            "UnixTime_ms",
        )


    return df[columns].copy()



# ============================================================
# Sampling Information
# ============================================================

def estimate_sampling_rate(
    df: pd.DataFrame,
    timestamp_column: str = "UnixTime_ms",
) -> float:
    """
    Estimate sample rate.

    Delegates to timestamps.py.
    """

    from timestamps import estimate_sampling_rate as _estimate


    return _estimate(
        df,
        timestamp_column,
    )



# ============================================================
# CSV Export
# ============================================================

def export_dataframe(
    df: pd.DataFrame,
    output_file: str | Path,
) -> None:
    """
    Export dataframe to CSV.
    """

    path = Path(
        output_file
    )


    df.to_csv(
        path,
        index=False,
    )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Dataclasses

    "LabScribeMetadata",

    "LabScribeData",


    # Loading

    "load_labscribe_csv",

    "extract_metadata",

    "prepare_labscribe_dataframe",

    "prepare_analysis_dataframe",


    # Channels

    "detect_channels",

    "detect_ecg_channel",

    "detect_balance_channels",

    "get_ecg_signal",

    "get_balance_data",


    # Timestamp

    "add_recording_timestamps",

    "estimate_sampling_rate",


    # Export

    "export_dataframe",
]