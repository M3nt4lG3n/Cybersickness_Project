"""
ecg.py

ECG processing utilities for the LabScribe analysis pipeline.

This module is responsible for:

    • ECG cleaning
    • R peak detection
    • Robust P/Q/R/S/T fiducial pairing
    • Heart-rate calculation
    • Beat interval calculation
    • ECG plotting
    • Summary statistics

Nothing in this module knows anything about balance board data or
video rendering.

Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================

DEFAULT_SAMPLING_RATE = 1000


# Physiological search windows relative to the R peak.
# Values are in seconds and converted to samples when needed.

P_SEARCH_WINDOW = (-0.25, -0.05)
Q_SEARCH_WINDOW = (-0.08, 0.00)
S_SEARCH_WINDOW = (0.00, 0.08)
T_SEARCH_WINDOW = (0.08, 0.45)


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class ECGBeat:
    """
    Represents one heartbeat.
    """

    beat_number: int

    r_index: int

    p_index: Optional[int]
    q_index: Optional[int]
    s_index: Optional[int]
    t_index: Optional[int]

    rr_ms: float
    heart_rate_bpm: float

    pr_ms: float
    qrs_ms: float
    qt_ms: float


@dataclass(slots=True)
class ECGAnalysisResult:
    """
    Returned by analyze_ecg().

    Attributes
    ----------
    clean_signal
        Filtered ECG.

    analysis_df
        Sample-by-sample dataframe.

    beats_df
        One row per heartbeat.

    summary
        Dictionary of summary metrics.

    signals
        Raw NeuroKit signal dataframe.

    info
        NeuroKit info dictionary.
    """

    clean_signal: np.ndarray

    analysis_df: pd.DataFrame

    beats_df: pd.DataFrame

    summary: dict

    signals: pd.DataFrame

    info: dict


# ============================================================
# Helper Functions
# ============================================================

def nearest_wave(
    r_index: int,
    candidates,
    min_offset: int,
    max_offset: int,
) -> Optional[int]:
    """
    Find the closest candidate wave inside a physiological window.

    Parameters
    ----------
    r_index
        Sample index of the R peak.

    candidates
        Array of detected fiducial locations.

    min_offset
        Minimum allowed offset from the R peak.

    max_offset
        Maximum allowed offset from the R peak.

    Returns
    -------
    int or None
    """

    if candidates is None:
        return None

    if len(candidates) == 0:
        return None

    candidates = np.asarray(candidates)

    lower = r_index + min_offset
    upper = r_index + max_offset

    valid = candidates[
        (candidates >= lower) &
        (candidates <= upper)
    ]

    if len(valid) == 0:
        return None

    nearest = valid[
        np.argmin(np.abs(valid - r_index))
    ]

    return int(nearest)


def sample_value(
    signal: np.ndarray,
    index: Optional[int],
) -> float:
    """
    Safely return the ECG value at an index.

    Missing fiducials return NaN.
    """

    if index is None:
        return np.nan

    if index < 0:
        return np.nan

    if index >= len(signal):
        return np.nan

    return float(signal[index])


def milliseconds(
    sample_difference: Optional[int],
    sampling_rate: int,
) -> float:
    """
    Convert a sample difference to milliseconds.
    """

    if sample_difference is None:
        return np.nan

    return (
        sample_difference /
        sampling_rate *
        1000.0
    )


def safe_interval(
    start_index: Optional[int],
    end_index: Optional[int],
    sampling_rate: int,
) -> float:
    """
    Compute an interval between two fiducials.

    Missing fiducials return NaN.
    """

    if start_index is None:
        return np.nan

    if end_index is None:
        return np.nan

    if end_index <= start_index:
        return np.nan

    return milliseconds(
        end_index - start_index,
        sampling_rate,
    )


# ============================================================
# Internal Utilities
# ============================================================

def _window_to_samples(
    window_seconds: tuple[float, float],
    sampling_rate: int,
) -> tuple[int, int]:
    """
    Convert a search window from seconds to samples.
    """

    return (
        int(window_seconds[0] * sampling_rate),
        int(window_seconds[1] * sampling_rate),
    )


def _heart_rate_from_rr(
    rr_ms: float,
) -> float:
    """
    Convert an RR interval to BPM.
    """

    if np.isnan(rr_ms):
        return np.nan

    if rr_ms <= 0:
        return np.nan

    return 60000.0 / rr_ms


def _empty_summary() -> dict:
    """
    Placeholder summary dictionary.

    Populated later by compute_hr_statistics().
    """

    return {
        "AverageHeartRate": np.nan,
        "MinimumHeartRate": np.nan,
        "MaximumHeartRate": np.nan,
        "HeartRateStd": np.nan,
        "TotalHeartBeats": 0,
    }

# ============================================================
# ECG Feature Detection
# ============================================================

def detect_ecg_features(
    signal: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> tuple[list[ECGBeat], np.ndarray, pd.DataFrame, dict]:
    """
    Detect ECG fiducials using NeuroKit2.

    This function performs:

        1. ECG cleaning
        2. R peak detection
        3. Delineation of P/Q/S/T waves
        4. Robust fiducial pairing
        5. Beat interval calculation

    Unlike the previous implementation, this does NOT assume that
    NeuroKit returns equal-length arrays for each fiducial. Each beat
    is matched independently around its R peak.

    Parameters
    ----------
    signal
        Raw ECG signal.

    timestamps
        Unix timestamps (currently retained for future use).

    sampling_rate
        ECG sampling frequency (Hz).

    Returns
    -------
    beats
        List[ECGBeat]

    clean_signal
        Filtered ECG signal.

    signals
        NeuroKit signals dataframe.

    info
        NeuroKit information dictionary.
    """

    # --------------------------------------------------------
    # Convert inputs
    # --------------------------------------------------------

    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps)

    # --------------------------------------------------------
    # Clean ECG
    # --------------------------------------------------------

    clean_signal = nk.ecg_clean(
        signal,
        sampling_rate=sampling_rate,
    )

    print("ECG debug:")
    print("Samples:", len(clean_signal))
    print("Min:", np.min(clean_signal))
    print("Max:", np.max(clean_signal))
    print("Mean:", np.mean(clean_signal))
    print("Std:", np.std(clean_signal))
    print("Sampling rate:", sampling_rate)

    # --------------------------------------------------------
    # NeuroKit processing
    # --------------------------------------------------------

    try:
        signals, info = nk.ecg_peaks(
            clean_signal,
            sampling_rate=sampling_rate,
            method="neurokit",
        )

    except Exception as error:

        print(
            f"NeuroKit ECG processing failed: {error}"
        )

        signals = pd.DataFrame(
            {
                "ECG_Clean": clean_signal
            }
        )

        info = {
            "ECG_R_Peaks": []
        }
    
    # --------------------------------------------------------
    # Validate detected R peaks
    # --------------------------------------------------------

    r_peaks = np.asarray(
        info["ECG_R_Peaks"],
        dtype=int,
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 4))
    plt.plot(clean_signal, label="Clean ECG")

    if len(r_peaks):
        plt.scatter(
            r_peaks,
            clean_signal[r_peaks],
            color="red",
            label="Detected R Peaks",
            zorder=5,
        )

    plt.legend()
    plt.title("ECG with Detected R Peaks")
    plt.show()

    print("Detected R peaks:", len(r_peaks))

    if len(r_peaks) > 0:
        print("First 10:", r_peaks[:10])

    if len(r_peaks) < 2:

        print(
            "Insufficient R peaks detected. "
            "Skipping ECG delineation."
        )

        return (
            [],
            clean_signal,
            signals,
            info,
        )

    # --------------------------------------------------------
    # Delineate ECG waveform
    # --------------------------------------------------------

    try:

        _, waves = nk.ecg_delineate(
            clean_signal,
            info["ECG_R_Peaks"],
            sampling_rate=sampling_rate,
            method="dwt",
        )

    except Exception:

        # Fallback if DWT fails
        _, waves = nk.ecg_delineate(
            clean_signal,
            info["ECG_R_Peaks"],
            sampling_rate=sampling_rate,
            method="peak",
        )

    # --------------------------------------------------------
    # Fiducials
    # --------------------------------------------------------

    r_peaks = np.asarray(
        info.get("ECG_R_Peaks", []),
        dtype=int,
    )

    print("Detected R peaks:", len(r_peaks))

    if len(r_peaks):
        print("First 10:", r_peaks[:10])

    p_peaks = np.asarray(
        waves.get("ECG_P_Peaks", []),
        dtype=int,
    )

    print("Detected P peaks:", len(p_peaks))

    if len(p_peaks):
        print("First 10:", p_peaks[:10])

    q_peaks = np.asarray(
        waves.get("ECG_Q_Peaks", []),
        dtype=int,
    )

    print("Detected Q peaks:", len(q_peaks))

    if len(q_peaks):
        print("First 10:", q_peaks[:10])

    s_peaks = np.asarray(
        waves.get("ECG_S_Peaks", []),
        dtype=int,
    )

    print("Detected S peaks:", len(s_peaks))
    if len(s_peaks):
        print("First 10:", s_peaks[:10])

    t_peaks = np.asarray(
        waves.get("ECG_T_Peaks", []),
        dtype=int,
    )

    print("Detected T peaks:", len(t_peaks))

    if len(t_peaks):
        print("First 10:", t_peaks[:10])

    # --------------------------------------------------------
    # Convert physiological windows to samples
    # --------------------------------------------------------

    p_window = _window_to_samples(
        P_SEARCH_WINDOW,
        sampling_rate,
    )

    q_window = _window_to_samples(
        Q_SEARCH_WINDOW,
        sampling_rate,
    )

    s_window = _window_to_samples(
        S_SEARCH_WINDOW,
        sampling_rate,
    )

    t_window = _window_to_samples(
        T_SEARCH_WINDOW,
        sampling_rate,
    )

    # --------------------------------------------------------
    # Build heartbeat list
    # --------------------------------------------------------

    beats: list[ECGBeat] = []

    previous_r: int | None = None

    for beat_number, r in enumerate(r_peaks, start=1):

        # --------------------------------------------
        # Robust wave pairing
        # --------------------------------------------

        p = nearest_wave(
            r,
            p_peaks,
            *p_window,
        )

        q = nearest_wave(
            r,
            q_peaks,
            *q_window,
        )

        s = nearest_wave(
            r,
            s_peaks,
            *s_window,
        )

        t = nearest_wave(
            r,
            t_peaks,
            *t_window,
        )

        # --------------------------------------------
        # RR interval
        # --------------------------------------------

        if previous_r is None:

            rr_ms = np.nan
            heart_rate = np.nan

        else:

            rr_ms = milliseconds(
                r - previous_r,
                sampling_rate,
            )

            heart_rate = _heart_rate_from_rr(
                rr_ms,
            )

        previous_r = r

        # --------------------------------------------
        # ECG intervals
        # --------------------------------------------

        pr_ms = safe_interval(
            p,
            q,
            sampling_rate,
        )

        qrs_ms = safe_interval(
            q,
            s,
            sampling_rate,
        )

        qt_ms = safe_interval(
            q,
            t,
            sampling_rate,
        )

        # --------------------------------------------
        # Store beat
        # --------------------------------------------

        beats.append(
            ECGBeat(
                beat_number=beat_number,
                r_index=int(r),
                p_index=p,
                q_index=q,
                s_index=s,
                t_index=t,
                rr_ms=rr_ms,
                heart_rate_bpm=heart_rate,
                pr_ms=pr_ms,
                qrs_ms=qrs_ms,
                qt_ms=qt_ms,
            )
        )

    return (
        beats,
        clean_signal,
        signals,
        info,
    )

# ============================================================
# Beat DataFrame Construction
# ============================================================

def beats_to_dataframe(
    beats: list[ECGBeat],
    clean_signal: np.ndarray,
) -> pd.DataFrame:
    """
    Convert ECGBeat objects into a dataframe.

    Adds ECG amplitude values for each fiducial point.
    Missing waves remain NaN.
    """

    rows = []

    for beat in beats:

        rows.append(
            {
                "Beat": beat.beat_number,

                "R_Index": beat.r_index,
                "P_Index": beat.p_index,
                "Q_Index": beat.q_index,
                "S_Index": beat.s_index,
                "T_Index": beat.t_index,

                "P_Value": sample_value(
                    clean_signal,
                    beat.p_index,
                ),

                "Q_Value": sample_value(
                    clean_signal,
                    beat.q_index,
                ),

                "R_Value": sample_value(
                    clean_signal,
                    beat.r_index,
                ),

                "S_Value": sample_value(
                    clean_signal,
                    beat.s_index,
                ),

                "T_Value": sample_value(
                    clean_signal,
                    beat.t_index,
                ),

                "RR_ms": beat.rr_ms,
                "HeartRate_BPM": beat.heart_rate_bpm,

                "PR_ms": beat.pr_ms,
                "QRS_ms": beat.qrs_ms,
                "QT_ms": beat.qt_ms,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Analysis DataFrame Construction
# ============================================================

def create_ecg_analysis_dataframe(
    signal: np.ndarray,
    clean_signal: np.ndarray,
    timestamps: np.ndarray,
    beats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create sample-level ECG dataframe.

    This dataframe is written to *_analysis.csv.

    Contains:

        UnixTime_ms
        Raw ECG
        Clean ECG
        Heart Rate
        Beat number
        Fiducial markers
    """

    df = pd.DataFrame(
        {
            "UnixTime_ms": timestamps,

            "ECG_Raw": signal,

            "ECG_Clean": clean_signal,
        }
    )

    df["HeartRate_BPM"] = np.nan

    df["Beat"] = np.nan


    # --------------------------------------------------------
    # Insert beat information
    # --------------------------------------------------------

    for _, row in beats_df.iterrows():

        index = int(row["R_Index"])

        if index >= len(df):
            continue

        df.loc[
            index,
            "HeartRate_BPM"
        ] = row["HeartRate_BPM"]

        df.loc[
            index,
            "Beat"
        ] = row["Beat"]


    # --------------------------------------------------------
    # Interpolate heart rate between beats
    # --------------------------------------------------------

    df["HeartRate_BPM"] = (
        df["HeartRate_BPM"]
        .interpolate()
        .bfill()
        .ffill()
    )


    return df


# ============================================================
# High Level ECG Analysis Wrapper
# ============================================================

def analyze_ecg(
    signal: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> ECGAnalysisResult:
    """
    Complete ECG analysis pipeline.

    This is the main function that should be called by main.py.

    Returns
    -------
    ECGAnalysisResult
    """

    (
        beats,
        clean_signal,
        signals,
        info,

    ) = detect_ecg_features(
        signal,
        timestamps,
        sampling_rate,
    )


    beats_df = beats_to_dataframe(
        beats,
        clean_signal,
    )


    analysis_df = create_ecg_analysis_dataframe(
        np.asarray(signal),
        clean_signal,
        np.asarray(timestamps),
        beats_df,
    )


    summary = _empty_summary()

    if len(beats_df) > 0:

        hr = (
            beats_df["HeartRate_BPM"]
            .dropna()
        )

        summary.update(
            {
                "AverageHeartRate":
                    float(hr.mean())
                    if len(hr)
                    else np.nan,

                "MinimumHeartRate":
                    float(hr.min())
                    if len(hr)
                    else np.nan,

                "MaximumHeartRate":
                    float(hr.max())
                    if len(hr)
                    else np.nan,

                "HeartRateStd":
                    float(hr.std())
                    if len(hr)
                    else np.nan,

                "TotalHeartBeats":
                    int(len(beats_df)),
            }
        )


    return ECGAnalysisResult(
        clean_signal=clean_signal,

        analysis_df=analysis_df,

        beats_df=beats_df,

        summary=summary,

        signals=signals,

        info=info,
    )

# ============================================================
# ECG Summary Statistics
# ============================================================

def compute_ecg_summary(
    beats_df: pd.DataFrame,
    clean_signal: np.ndarray,
) -> dict:
    """
    Calculate ECG summary metrics.

    Metrics included:

        AverageHeartRate
        MinimumHeartRate
        MaximumHeartRate
        HeartRateStd
        AverageRR
        AveragePR
        AverageQRS
        AverageQT
        TotalHeartBeats
    """

    summary = {}

    # --------------------------------------------------------
    # Heart rate statistics
    # --------------------------------------------------------

    required = [
        "HeartRate_BPM",
        "RR_ms",
        "PR_ms",
        "QRS_ms",
        "QT_ms",
    ]
    
    if not all(col in beats_df.columns for col in required):
        return {
            "AverageHeartRate": np.nan,
            "MinimumHeartRate": np.nan,
            "MaximumHeartRate": np.nan,
            "HeartRateStd": np.nan,
            "AverageRR": np.nan,
            "AveragePR": np.nan,
            "AverageQRS": np.nan,
            "AverageQT": np.nan,
            "TotalHeartBeats": 0,
        }

    heart_rate = (
        beats_df["HeartRate_BPM"]
        .dropna()
    )

    if len(heart_rate):

        summary["AverageHeartRate"] = float(
            heart_rate.mean()
        )

        summary["MinimumHeartRate"] = float(
            heart_rate.min()
        )

        summary["MaximumHeartRate"] = float(
            heart_rate.max()
        )

        summary["HeartRateStd"] = float(
            heart_rate.std()
        )

    else:

        summary["AverageHeartRate"] = np.nan
        summary["MinimumHeartRate"] = np.nan
        summary["MaximumHeartRate"] = np.nan
        summary["HeartRateStd"] = np.nan


    # --------------------------------------------------------
    # Interval statistics
    # --------------------------------------------------------

    for column in [
        "RR_ms",
        "PR_ms",
        "QRS_ms",
        "QT_ms",
    ]:

        values = (
            beats_df[column]
            .dropna()
        )

        if len(values):

            summary[
                f"Average{column}"
            ] = float(
                values.mean()
            )

        else:

            summary[
                f"Average{column}"
            ] = np.nan


    # --------------------------------------------------------
    # Beat count
    # --------------------------------------------------------

    summary["TotalHeartBeats"] = int(
        len(beats_df)
    )


    # --------------------------------------------------------
    # ECG signal statistics
    # --------------------------------------------------------

    if len(clean_signal):

        summary["ECG_Min"] = float(
            np.min(clean_signal)
        )

        summary["ECG_Max"] = float(
            np.max(clean_signal)
        )

        summary["ECG_Mean"] = float(
            np.mean(clean_signal)
        )

        summary["ECG_STD"] = float(
            np.std(clean_signal)
        )

    else:

        summary["ECG_Min"] = np.nan
        summary["ECG_Max"] = np.nan
        summary["ECG_Mean"] = np.nan
        summary["ECG_STD"] = np.nan


    return summary



# ============================================================
# ECG Plotting
# ============================================================

def plot_ecg(
    clean_signal: np.ndarray,
    timestamps: np.ndarray,
    beats_df: pd.DataFrame,
    output_file: str,
    title: str = "ECG Analysis",
):
    """
    Generate ECG analysis figure.

    Uses robust scaling:

        baseline = median(signal)

        amplitude =
            max(abs(signal-baseline))

        ymin =
            baseline - amplitude*1.2

        ymax =
            baseline + amplitude*1.2


    This prevents isolated noise spikes from compressing
    the useful ECG waveform.
    """


    if len(clean_signal) == 0:
        return


    # --------------------------------------------------------
    # Convert time axis
    # --------------------------------------------------------

    time_seconds = (
        timestamps -
        timestamps[0]
    ) / 1000.0


    # --------------------------------------------------------
    # Robust scaling
    # --------------------------------------------------------

    baseline = np.median(
        clean_signal
    )

    amplitude = np.max(
        np.abs(
            clean_signal - baseline
        )
    )


    if amplitude == 0:
        amplitude = 1


    ymin = (
        baseline -
        amplitude * 1.2
    )

    ymax = (
        baseline +
        amplitude * 1.2
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 5)
    )


    plt.plot(
        time_seconds,
        clean_signal,
        linewidth=1,
        label="ECG",
    )


    # --------------------------------------------------------
    # Fiducial markers
    # --------------------------------------------------------

    marker_config = [

        (
            "P_Index",
            "P",
        ),

        (
            "Q_Index",
            "Q",
        ),

        (
            "R_Index",
            "R",
        ),

        (
            "S_Index",
            "S",
        ),

        (
            "T_Index",
            "T",
        ),
    ]


    for column, label in marker_config:

        if column not in beats_df:
            continue


        indices = (
            beats_df[column]
            .dropna()
            .astype(int)
        )


        valid = indices[
            indices < len(clean_signal)
        ]


        if len(valid) == 0:
            continue


        plt.scatter(
            time_seconds[valid],
            clean_signal[valid],
            s=25,
            label=label,
        )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.ylim(
        ymin,
        ymax,
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )


    plt.title(
        title
    )


    plt.grid(
        True
    )


    plt.legend(
        loc="upper right"
    )


    plt.tight_layout()


    plt.savefig(
        output_file,
        dpi=300,
    )


    plt.close()

# ============================================================
# Validation Helpers
# ============================================================

def validate_ecg_input(
    signal,
    timestamps,
) -> None:
    """
    Validate ECG input arrays.

    Raises
    ------
    ValueError
        If the input data cannot be processed.
    """

    if signal is None:
        raise ValueError(
            "ECG signal is missing."
        )

    if timestamps is None:
        raise ValueError(
            "Timestamp data is missing."
        )


    if len(signal) == 0:
        raise ValueError(
            "ECG signal contains no samples."
        )


    if len(signal) != len(timestamps):

        raise ValueError(
            "ECG signal and timestamps must have equal length."
        )



# ============================================================
# CSV Export Helpers
# ============================================================

def export_ecg_analysis(
    result: ECGAnalysisResult,
    output_file: str,
) -> None:
    """
    Export sample-by-sample ECG analysis.

    Creates:

        *_analysis.csv
    """

    result.analysis_df.to_csv(
        output_file,
        index=False,
    )



def export_ecg_beats(
    result: ECGAnalysisResult,
    output_file: str,
) -> None:
    """
    Export heartbeat table.

    Creates:

        *_beats.csv
    """

    result.beats_df.to_csv(
        output_file,
        index=False,
    )



def export_ecg_summary(
    result: ECGAnalysisResult,
    output_file: str,
) -> None:
    """
    Export ECG summary.

    Creates:

        *_summary.csv
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
# Final ECG Analysis Pipeline
# ============================================================

def analyze_ecg(
    signal: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> ECGAnalysisResult:
    """
    Complete ECG processing pipeline.

    This is the main entry point for the LabScribe
    analysis program.

    Parameters
    ----------
    signal:
        Raw ECG channel.

    timestamps:
        UnixTime_ms values.

    sampling_rate:
        ECG sampling frequency.

    Returns
    -------
    ECGAnalysisResult
    """


    validate_ecg_input(
        signal,
        timestamps,
    )


    signal_array = np.asarray(
        signal,
        dtype=float,
    )

    timestamp_array = np.asarray(
        timestamps,
    )


    (
        beats,
        clean_signal,
        signals,
        info,

    ) = detect_ecg_features(
        signal_array,
        timestamp_array,
        sampling_rate,
    )


    beats_df = beats_to_dataframe(
        beats,
        clean_signal,
    )


    analysis_df = create_ecg_analysis_dataframe(
        signal_array,
        clean_signal,
        timestamp_array,
        beats_df,
    )


    summary = compute_ecg_summary(
        beats_df,
        clean_signal,
    )


    return ECGAnalysisResult(

        clean_signal=clean_signal,

        analysis_df=analysis_df,

        beats_df=beats_df,

        summary=summary,

        signals=signals,

        info=info,
    )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Dataclasses

    "ECGBeat",
    "ECGAnalysisResult",


    # Main analysis

    "analyze_ecg",
    "detect_ecg_features",


    # Statistics

    "compute_ecg_summary",


    # Visualization

    "plot_ecg",


    # Export

    "export_ecg_analysis",
    "export_ecg_beats",
    "export_ecg_summary",


    # Helpers

    "nearest_wave",
    "sample_value",
]