"""
timestamps.py

Utilities for generating high-resolution Unix timestamps for
LabScribe recordings.

Design
------
LabScribe exports two time columns:

    TimeOfDay
        Minute:Second.Tenth
        Example:
            33:57.1

        Resolution:
            0.1 second

    Time
        Elapsed recording time

        Resolution:
            0.01 second

The Time column is treated as the authoritative timing
information because it has ten times the precision of
TimeOfDay.

TimeOfDay is only used to anchor the recording to an
absolute clock time.

Author
------
Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import math
import re

import numpy as np
import pandas as pd


# ==========================================================
# Constants
# ==========================================================

TIME_COLUMN = "Time"

TIME_OF_DAY_COLUMN = "TimeOfDay"

UNIX_COLUMN = "UnixTime_ms"

MILLISECONDS_PER_SECOND = 1000

# Standardized patient folder naming convention:
#     Patient_YearMonthDay_HourMinuteSecond
# Example:
#     Patient_20260720_163422 -> 2026-07-20 16:34:22
PATIENT_FOLDER_PATTERN = re.compile(
    r"^Patient_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$"
)


# ==========================================================
# Dataclasses
# ==========================================================

@dataclass(slots=True)
class RecordingStart:
    """
    Represents the estimated recording start.

    Attributes
    ----------
    datetime
        Estimated absolute start time.

    unix_ms
        Unix timestamp in milliseconds.
    """

    datetime: datetime

    unix_ms: int


# ==========================================================
# Validation
# ==========================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Validate the LabScribe dataframe.
    """

    if df.empty:

        raise ValueError(
            "Dataframe is empty."
        )

    required = (
        TIME_COLUMN,
        TIME_OF_DAY_COLUMN,
    )

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column '{column}'."
            )


def validate_hour(
    hour: int,
) -> None:
    """
    Validate the recording hour.
    """

    if not 1 <= hour <= 12:

        raise ValueError(
            "Hour must be between 1 and 12."
        )


def validate_ampm(
    am_pm: str,
) -> str:
    """
    Validate AM/PM input.
    """

    value = am_pm.strip().upper()

    if value not in ("AM", "PM"):

        raise ValueError(
            "AM/PM must be either 'AM' or 'PM'."
        )

    return value


def convert_hour_to_24(
    hour: int,
    am_pm: str,
) -> int:
    """
    Convert a 12-hour clock into 24-hour time.
    """

    validate_hour(hour)

    am_pm = validate_ampm(am_pm)

    if hour == 12:

        return 0 if am_pm == "AM" else 12

    if am_pm == "PM":

        return hour + 12

    return hour


# ==========================================================
# Datetime Helpers
# ==========================================================

def datetime_to_unix_ms(
    value: datetime,
) -> int:
    """
    Convert a datetime into Unix milliseconds.
    """

    return int(
        round(
            value.timestamp()
            * MILLISECONDS_PER_SECOND
        )
    )


def unix_ms_to_datetime(
    unix_ms: int,
) -> datetime:
    """
    Convert Unix milliseconds back into datetime.
    """

    return datetime.fromtimestamp(
        unix_ms / MILLISECONDS_PER_SECOND
    )

# ==========================================================
# Time Parsing
# ==========================================================

def parse_time_of_day(
    value: str,
) -> float:
    """
    Convert a LabScribe TimeOfDay value into seconds.

    Example
    -------
    33:57.1

    Returns
    -------
    2037.1
    """

    value = str(value).strip()

    try:

        minute_str, second_str = value.split(":")

        minute = int(minute_str)

        second = float(second_str)

    except Exception as error:

        raise ValueError(
            f"Invalid TimeOfDay value '{value}'."
        ) from error

    return (
        minute * 60.0
        + second
    )


# ==========================================================
# Recording Start Estimation
# ==========================================================

def estimate_start_offset(
    df: pd.DataFrame,
) -> float:
    """
    Estimate the recording start (seconds into the hour).

    Each row provides

        TimeOfDay - Time

    Because TimeOfDay has only 0.1-second precision,
    averaging every sample greatly reduces the
    quantization error.
    """

    validate_dataframe(df)

    time_of_day = np.array(

        [
            parse_time_of_day(value)
            for value in df[TIME_OF_DAY_COLUMN]
        ],

        dtype=float,

    )

    elapsed = df[TIME_COLUMN].to_numpy(
        dtype=float,
    )

    offsets = time_of_day - elapsed

    return float(
        np.mean(offsets)
    )


# ==========================================================
# Patient Folder Parsing
# ==========================================================

def parse_patient_folder_datetime(
    folder_name: str,
) -> datetime:
    """
    Parse the standardized patient folder name into a datetime.

    Expected format
    ----------------
        Patient_YearMonthDay_HourMinuteSecond

    Example
    -------
        Patient_20260720_163422
            -> datetime(2026, 7, 20, 16, 34, 22)

    This folder name is treated as authoritative for the
    whole-second wall-clock start of the recording (year,
    month, day, hour, minute, second). It has no sub-second
    resolution, so millisecond precision is estimated
    separately from the LabScribe TimeOfDay/Time columns
    (see build_recording_start).
    """

    match = PATIENT_FOLDER_PATTERN.match(
        folder_name.strip()
    )

    if not match:

        raise ValueError(
            f"Folder name '{folder_name}' does not match the "
            "expected 'Patient_YYYYMMDD_HHMMSS' format."
        )

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
    ) = (
        int(part)
        for part in match.groups()
    )

    try:

        return datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        )

    except ValueError as error:

        raise ValueError(
            f"Folder name '{folder_name}' contains an invalid "
            f"date/time: {error}"
        ) from error


def get_patient_folder_datetime(
    file_path,
) -> datetime:
    """
    Convenience wrapper: parse the standardized patient
    folder datetime from the parent folder of a given file.
    """

    from pathlib import Path

    folder_name = Path(file_path).parent.name

    return parse_patient_folder_datetime(
        folder_name
    )


# ==========================================================
# Recording Start Construction
# ==========================================================

def build_recording_start(
    df: pd.DataFrame,
    folder_datetime: datetime,
) -> RecordingStart:
    """
    Build absolute recording start timestamp.

    The whole-second wall-clock start (year, month, day,
    hour, minute, second) comes from the standardized
    Patient folder name (see parse_patient_folder_datetime)
    and is treated as authoritative.

    The folder name has no sub-second resolution, so the
    millisecond offset within that second is estimated from
    the LabScribe TimeOfDay/Time columns using
    estimate_start_offset, which averages across every
    sample to reduce the 0.1-second quantization error in
    TimeOfDay. Only the fractional (sub-second) part of that
    estimate is used here -- the minute/second themselves
    always come from folder_datetime.
    """

    start_seconds = estimate_start_offset(
        df,
    )

    fractional_second = (
        start_seconds
        -
        math.floor(start_seconds)
    )

    microsecond = int(
        round(
            fractional_second
            * 1_000_000
        )
    )

    # Guard against rounding up to a full second.
    if microsecond >= 1_000_000:

        microsecond = 0

        folder_datetime = (
            folder_datetime
            + timedelta(seconds=1)
        )

    start_datetime = folder_datetime.replace(
        microsecond=microsecond,
    )

    return RecordingStart(

        datetime=start_datetime,

        unix_ms=datetime_to_unix_ms(
            start_datetime
        ),

    )

# ==========================================================
# Unix Timestamp Generation
# ==========================================================

def calculate_unix_timestamps(
    recording_start: RecordingStart,
    elapsed_seconds,
) -> np.ndarray:
    """
    Generate Unix timestamps from elapsed recording time.

    Parameters
    ----------
    recording_start
        Estimated absolute recording start.

    elapsed_seconds
        LabScribe Time column.

    Returns
    -------
    ndarray[int64]
        Unix timestamps in milliseconds.
    """

    elapsed_seconds = np.asarray(
        elapsed_seconds,
        dtype=float,
    )

    elapsed_ms = np.round(
        elapsed_seconds
        * MILLISECONDS_PER_SECOND
    ).astype(np.int64)

    return (
        recording_start.unix_ms
        + elapsed_ms
    )


def add_unix_timestamp_column(
    df: pd.DataFrame,
    recording_start: RecordingStart,
) -> pd.DataFrame:
    """
    Return a dataframe containing a UnixTime_ms column.

    The original dataframe is left unchanged.
    """

    validate_dataframe(df)

    result = df.copy()

    result[UNIX_COLUMN] = calculate_unix_timestamps(

        recording_start,

        result[TIME_COLUMN],

    )

    return result

# ==========================================================
# Timestamp Validation
# ==========================================================

def validate_unix_timestamps(
    df: pd.DataFrame,
) -> None:
    """
    Validate the generated Unix timestamps.
    """

    if UNIX_COLUMN not in df.columns:

        raise ValueError(
            f"Missing '{UNIX_COLUMN}' column."
        )

    timestamps = df[UNIX_COLUMN]

    if timestamps.isna().any():

        raise ValueError(
            "UnixTime_ms contains missing values."
        )

    if len(timestamps) < 2:

        return

    differences = np.diff(
        timestamps.to_numpy()
    )

    if np.any(differences <= 0):

        raise ValueError(
            "Unix timestamps are not strictly increasing."
        )

    expected = np.round(

        np.diff(
            df[TIME_COLUMN].to_numpy()
        )

        * MILLISECONDS_PER_SECOND

    ).astype(np.int64)

    if not np.array_equal(
        differences,
        expected,
    ):

        raise ValueError(
            "UnixTime_ms increments do not match the "
            "LabScribe Time column."
        )
    
# ==========================================================
# Segment 3: Generate absolute timestamps from LabScribe Time
# ==========================================================

def generate_unix_timestamps(
    elapsed_seconds: np.ndarray,
    recording_start: RecordingStart,
) -> np.ndarray:
    """
    Convert LabScribe elapsed Time values into Unix timestamps.

    Parameters
    ----------
    elapsed_seconds:
        LabScribe Time column values in seconds.

    recording_start:
        Absolute recording start time.

    Returns
    -------
    np.ndarray
        Unix timestamps in milliseconds.

    Notes
    -----
    The LabScribe Time column is considered authoritative.

    Example:
        Time:
            0.00
            0.01
            0.02

        Start:
            2026-07-21 14:33:57.100

        Output:
            1784644437100
            1784644437110
            1784644437120

    No rollover handling is required here. If LabScribe displays:

        33:57.1

    followed by:

        34:00.0

    the elapsed Time column already contains the continuous time.
    """

    if elapsed_seconds is None:
        raise ValueError(
            "Elapsed time data cannot be None."
        )

    elapsed_seconds = np.asarray(
        elapsed_seconds,
        dtype=float,
    )

    if elapsed_seconds.ndim != 1:
        raise ValueError(
            "Elapsed time must be a one-dimensional array."
        )

    if len(elapsed_seconds) == 0:
        return np.array([], dtype=np.int64)

    if np.any(~np.isfinite(elapsed_seconds)):
        raise ValueError(
            "Elapsed time contains invalid values."
        )

    if np.any(elapsed_seconds < 0):
        raise ValueError(
            "Elapsed time cannot contain negative values."
        )

    unix_ms = (
        recording_start.unix_ms
        +
        (elapsed_seconds * 1000)
    )

    return np.round(
        unix_ms
    ).astype(np.int64)

# ==========================================================
# Segment 4: Pandas DataFrame timestamp integration
# ==========================================================

def add_unix_timestamp_column(
    df,
    recording_start: RecordingStart,
    time_column: str = "Time",
    output_column: str = "UnixTime_ms",
):
    """
    Add an absolute Unix timestamp column to a LabScribe DataFrame.

    Parameters
    ----------
    df:
        Pandas DataFrame containing LabScribe data.

    recording_start:
        RecordingStart object containing absolute start time.

    time_column:
        Name of LabScribe elapsed time column.

    output_column:
        Name of generated timestamp column.

    Returns
    -------
    pandas.DataFrame
        Copy of dataframe with UnixTime_ms added.

    Notes
    -----
    The original LabScribe columns are preserved.

    The generated timestamps are based only on:

        UnixTime_ms =
            recording_start
            +
            Time * 1000

    The Time column should contain elapsed seconds.
    """

    if time_column not in df.columns:
        raise KeyError(
            f"Missing required time column: {time_column}"
        )

    if df.empty:
        raise ValueError(
            "Cannot generate timestamps for empty dataframe."
        )

    result = df.copy()

    elapsed = result[time_column].to_numpy(
        dtype=float
    )

    timestamps = generate_unix_timestamps(
        elapsed,
        recording_start,
    )

    if len(timestamps) != len(result):
        raise RuntimeError(
            "Generated timestamp count does not match dataframe rows."
        )

    result[output_column] = timestamps

    return result


# ==========================================================
# Validation helpers
# ==========================================================

def validate_time_column(
    df,
    time_column: str = "Time",
) -> None:
    """
    Validate that a LabScribe Time column is usable.

    Checks:
        - exists
        - numeric
        - no NaN
        - monotonically increasing
        - starts at or near zero

    Raises
    ------
    ValueError
        If validation fails.
    """

    if time_column not in df.columns:
        raise ValueError(
            f"Missing Time column: {time_column}"
        )

    values = df[time_column].to_numpy(
        dtype=float
    )

    if len(values) == 0:
        raise ValueError(
            "Time column is empty."
        )

    if np.any(~np.isfinite(values)):
        raise ValueError(
            "Time column contains invalid values."
        )

    if np.any(np.diff(values) < 0):
        raise ValueError(
            "Time column is not monotonically increasing."
        )

    if values[0] > 0.1:
        raise ValueError(
            f"Time column does not start near zero. "
            f"First value: {values[0]}"
        )
    
# ==========================================================
# Segment 5: Public exports and standalone testing
# ==========================================================

__all__ = [
    "RecordingStart",
    "parse_time_of_day",
    "parse_patient_folder_datetime",
    "get_patient_folder_datetime",
    "build_recording_start",
    "get_recording_start",
    "generate_unix_timestamps",
    "add_unix_timestamp_column",
    "validate_time_column",
]


# ==========================================================
# Standalone verification
# ==========================================================

if __name__ == "__main__":

    import pandas as pd

    print(
        "Running timestamps.py self-test..."
    )

    # ------------------------------------------------------
    # Simulated LabScribe data
    # ------------------------------------------------------

    test_df = pd.DataFrame(
        {
            "TimeOfDay": [
                "33:57.1",
                "33:57.1",
                "33:57.1",
                "33:57.2",
                "33:57.2",
            ],
            "Time": [
                0.00,
                0.01,
                0.02,
                0.03,
                0.04,
            ],
            "i1 2": [
                0.1,
                0.2,
                0.15,
                0.3,
                0.2,
            ],
        }
    )


    # ------------------------------------------------------
    # Validate Time column
    # ------------------------------------------------------

    validate_time_column(
        test_df
    )

    print(
        "Time column validation passed."
    )


    # ------------------------------------------------------
    # Create recording start
    # ------------------------------------------------------

    folder_datetime = parse_patient_folder_datetime(
        "Patient_20260721_143357"
    )

    start = build_recording_start(
        test_df,
        folder_datetime,
    )

    print(
        "Recording start:"
    )

    print(
        start
    )


    # ------------------------------------------------------
    # Generate timestamps
    # ------------------------------------------------------

    result = add_unix_timestamp_column(
        test_df,
        start,
    )


    print(
        "\nGenerated dataframe:"
    )

    print(
        result
    )


    print(
        "\nTimestamp self-test complete."
    )