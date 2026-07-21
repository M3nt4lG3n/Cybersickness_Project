"""
timestamps.py

Timestamp conversion utilities for LabScribe analysis.

Responsibilities:

    • Convert LabScribe relative Time values into UnixTime_ms
    • Combine CSV creation date with user-provided recording time
    • Validate timestamp generation

This module does not handle:

    • CSV loading
    • ECG processing
    • Balance processing
    • Video rendering


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd



# ============================================================
# Constants
# ============================================================

MILLISECONDS_PER_SECOND = 1000



# ============================================================
# Dataclass
# ============================================================

@dataclass(slots=True)
class RecordingStartTime:
    """
    Represents the absolute recording start time.

    Attributes
    ----------
    date
        Calendar date of recording.

    hour
        Recording hour (1-12).

    minute
        Recording minute.

    second
        Recording second.

    am_pm
        AM or PM.
    """

    date: datetime

    hour: int

    minute: int = 0

    second: int = 0

    am_pm: str = "AM"



# ============================================================
# Validation Helpers
# ============================================================

def validate_ampm(
    value: str,
) -> str:
    """
    Validate AM/PM input.

    Returns uppercase version.
    """

    value = value.strip().upper()


    if value not in (
        "AM",
        "PM",
    ):

        raise ValueError(
            "AM/PM must be either 'AM' or 'PM'."
        )


    return value



def validate_hour(
    hour: int,
) -> None:
    """
    Validate 12-hour clock hour.
    """

    if hour < 1 or hour > 12:

        raise ValueError(
            "Hour must be between 1 and 12."
        )



def validate_minute_second(
    value: int,
    name: str,
) -> None:
    """
    Validate minute or second.
    """

    if value < 0 or value > 59:

        raise ValueError(
            f"{name} must be between 0 and 59."
        )



# ============================================================
# Time Parsing
# ============================================================

def parse_recording_hour(
    hour: int,
    am_pm: str,
    minute: int = 0,
    second: int = 0,
) -> tuple[int, int, int]:
    """
    Convert 12-hour clock input into 24-hour time.

    Examples:

        12 AM -> 0

        12 PM -> 12

        5 PM -> 17
    """

    am_pm = validate_ampm(
        am_pm
    )


    validate_hour(
        hour
    )


    validate_minute_second(
        minute,
        "Minute",
    )


    validate_minute_second(
        second,
        "Second",
    )


    if hour == 12:

        if am_pm == "AM":

            hour_24 = 0

        else:

            hour_24 = 12

    else:

        if am_pm == "PM":

            hour_24 = hour + 12

        else:

            hour_24 = hour


    return (
        hour_24,
        minute,
        second,
    )

# ============================================================
# Recording Start Time Construction
# ============================================================

def combine_date_and_time(
    date: datetime,
    hour: int,
    am_pm: str,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """
    Combine a CSV creation date with a user-provided
    recording start time.

    Example:

        CSV date:
            2026-07-20

        User input:
            3:30 PM

        Result:
            2026-07-20 15:30:00
    """

    hour_24, minute, second = parse_recording_hour(
        hour,
        am_pm,
        minute,
        second,
    )


    return datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour_24,
        minute=minute,
        second=second,
        microsecond=0,
        tzinfo=date.tzinfo,
    )



def create_recording_start(
    csv_creation_date: datetime,
    recording_hour: int,
    am_pm: str,
    recording_minute: int = 0,
    recording_second: int = 0,
) -> RecordingStartTime:
    """
    Create a RecordingStartTime object.

    This is the recommended interface for main.py.
    """

    start_time = combine_date_and_time(
        csv_creation_date,
        recording_hour,
        am_pm,
        recording_minute,
        recording_second,
    )


    return RecordingStartTime(

        date=start_time,

        hour=start_time.hour,

        minute=start_time.minute,

        second=start_time.second,

        am_pm=(
            "AM"
            if start_time.hour < 12
            else "PM"
        ),
    )



# ============================================================
# Unix Timestamp Conversion
# ============================================================

def datetime_to_unix_ms(
    timestamp: datetime,
) -> int:
    """
    Convert datetime to Unix milliseconds.
    """

    return int(
        timestamp.timestamp()
        *
        MILLISECONDS_PER_SECOND
    )



def calculate_unix_timestamp(
    start_time: datetime,
    relative_time_seconds,
) -> np.ndarray:
    """
    Convert LabScribe relative Time values into UnixTime_ms.

    Parameters
    ----------
    start_time:
        Absolute recording start time.

    relative_time_seconds:
        LabScribe Time column.

        Expected units:
            seconds

    Returns
    -------
    numpy.ndarray

        Unix timestamps in milliseconds.
    """

    relative_time_seconds = np.asarray(
        relative_time_seconds,
        dtype=float,
    )


    start_unix_ms = datetime_to_unix_ms(
        start_time
    )


    return (
        start_unix_ms
        +
        relative_time_seconds * 1000
    ).astype(
        np.int64
    )



def add_unix_time_column(
    df: pd.DataFrame,
    start_time: datetime,
    time_column: str = "Time",
) -> pd.DataFrame:
    """
    Add UnixTime_ms column to LabScribe dataframe.

    Original columns remain unchanged.
    """

    if time_column not in df.columns:

        raise ValueError(
            f"Missing time column: {time_column}"
        )


    result = df.copy()


    result["UnixTime_ms"] = (
        calculate_unix_timestamp(
            start_time,
            result[time_column],
        )
    )


    return result



# ============================================================
# Timestamp Validation
# ============================================================

def validate_timestamp_column(
    df: pd.DataFrame,
    column: str = "UnixTime_ms",
) -> None:
    """
    Validate generated timestamps.
    """

    if column not in df.columns:

        raise ValueError(
            f"Missing timestamp column: {column}"
        )


    values = df[column]


    if values.isna().any():

        raise ValueError(
            "Timestamp column contains missing values."
        )


    if len(values) > 1:

        differences = (
            values.diff()
            .dropna()
        )


        if (differences < 0).any():

            raise ValueError(
                "Timestamp column is not increasing."
            )
        
# ============================================================
# Date Parsing Helpers
# ============================================================

def parse_csv_creation_date(
    date_value,
) -> datetime:
    """
    Convert a LabScribe creation date into datetime.

    Supports:

        datetime objects

        pandas timestamps

        common string formats
    """

    if isinstance(
        date_value,
        datetime,
    ):
        return date_value


    if isinstance(
        date_value,
        pd.Timestamp,
    ):
        return date_value.to_pydatetime()



    if isinstance(
        date_value,
        str,
    ):

        formats = [

            "%Y-%m-%d",

            "%m/%d/%Y",

            "%m/%d/%y",

            "%Y/%m/%d",

            "%B %d %Y",

            "%b %d %Y",

        ]


        for fmt in formats:

            try:

                return datetime.strptime(
                    date_value,
                    fmt,
                )

            except ValueError:

                continue



    raise ValueError(
        f"Unable to parse creation date: {date_value}"
    )



# ============================================================
# High Level Timestamp Generation
# ============================================================

def generate_unix_timestamps(
    df: pd.DataFrame,
    csv_creation_date,
    recording_hour: int,
    am_pm: str,
    recording_minute: int = 0,
    recording_second: int = 0,
    time_column: str = "Time",
) -> pd.DataFrame:
    """
    Complete LabScribe timestamp pipeline.

    Steps:

        1. Parse CSV creation date

        2. Combine with recording start time

        3. Convert relative LabScribe time

        4. Add UnixTime_ms column


    Parameters
    ----------

    df:
        LabScribe dataframe.


    csv_creation_date:
        Date extracted from CSV metadata.


    recording_hour:
        Recording start hour (1-12).


    am_pm:
        AM or PM.


    Returns
    -------

    DataFrame containing UnixTime_ms.
    """

    parsed_date = parse_csv_creation_date(
        csv_creation_date,
    )


    start = create_recording_start(
        parsed_date,
        recording_hour,
        am_pm,
        recording_minute,
        recording_second,
    )


    result = add_unix_time_column(
        df,
        start.date,
        time_column,
    )


    validate_timestamp_column(
        result,
    )


    return result



# ============================================================
# Convenience Functions
# ============================================================

def get_recording_duration(
    df: pd.DataFrame,
    timestamp_column: str = "UnixTime_ms",
) -> float:
    """
    Return recording duration in seconds.
    """

    if timestamp_column not in df.columns:

        raise ValueError(
            f"Missing column: {timestamp_column}"
        )


    if len(df) < 2:

        return 0.0


    duration_ms = (
        df[timestamp_column].iloc[-1]
        -
        df[timestamp_column].iloc[0]
    )


    return float(
        duration_ms / 1000.0
    )



def estimate_sampling_rate(
    df: pd.DataFrame,
    timestamp_column: str = "UnixTime_ms",
) -> float:
    """
    Estimate sampling rate from timestamps.

    Useful when LabScribe exports do not provide
    an explicit sample rate.
    """

    if len(df) < 2:

        return np.nan


    dt = (
        df[timestamp_column]
        .diff()
        .dropna()
    )


    median_ms = np.median(
        dt
    )


    if median_ms <= 0:

        return np.nan


    return (
        1000.0 /
        median_ms
    )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Dataclasses

    "RecordingStartTime",


    # Main functions

    "generate_unix_timestamps",

    "create_recording_start",

    "add_unix_time_column",


    # Conversion

    "calculate_unix_timestamp",

    "datetime_to_unix_ms",

    "parse_csv_creation_date",


    # Validation

    "validate_timestamp_column",


    # Utilities

    "get_recording_duration",

    "estimate_sampling_rate",
]