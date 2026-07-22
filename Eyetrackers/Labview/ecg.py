"""
ecg.py

Modern ECG processing pipeline for LabScribe recordings.

Design goals
------------
• Preserve the raw ECG waveform exactly as recorded.
• Use NeuroKit's recommended ECG pipeline.
• Detect reliable R peaks.
• Delineate P/Q/R/S/T waves.
• Compute beat intervals and heart-rate statistics.
• Export analysis CSVs.
• Plot the original ECG waveform with detected fiducials.

Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import neurokit2 as nk
import numpy as np
import pandas as pd


# ==========================================================
# Constants
# ==========================================================

DEFAULT_SAMPLING_RATE = 100


# ==========================================================
# ECG Strip-Chart Plot Layout
# ==========================================================
#
# LabScribe's live acquisition display scrolls the waveform
# across a fine time grid, similar to a chart recorder /
# clinical ECG strip. To reproduce that look in a static
# image, the figure is made very wide -- proportional to the
# recording's duration -- with:
#
#   • minor gridlines every ECG_MINOR_GRID_SECONDS
#     ("small squares", millisecond-level resolution)
#
#   • major gridlines + labeled ticks every
#     ECG_MAJOR_GRID_SECONDS ("big squares")
#
# A safety cap keeps the image from becoming unrenderable /
# unusably large on very long recordings.

ECG_MINOR_GRID_SECONDS = 0.04   # 40 ms "small square"
ECG_MAJOR_GRID_SECONDS = 1.0    # 1 s labeled gridline

ECG_PLOT_PIXELS_PER_SECOND = 200
ECG_PLOT_DPI = 150

ECG_PLOT_MIN_WIDTH_INCHES = 15
ECG_PLOT_MAX_WIDTH_INCHES = 300

ECG_PLOT_HEIGHT_INCHES = 6


# ==========================================================
# Dataclasses
# ==========================================================

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

    p_value: float
    q_value: float
    r_value: float
    s_value: float
    t_value: float


@dataclass(slots=True)
class ECGAnalysisResult:
    """
    Complete ECG processing result.
    """

    raw_signal: np.ndarray

    clean_signal: np.ndarray

    timestamps: np.ndarray

    analysis_df: pd.DataFrame

    beats_df: pd.DataFrame

    summary: dict

    signals: pd.DataFrame

    info: dict

# ==========================================================
# Validation
# ==========================================================

def validate_ecg_input(
    signal,
    timestamps,
) -> None:
    """
    Validate ECG inputs before processing.
    """

    if signal is None:
        raise ValueError("ECG signal is None.")

    if timestamps is None:
        raise ValueError("Timestamp array is None.")

    if len(signal) == 0:
        raise ValueError("ECG signal is empty.")

    if len(signal) != len(timestamps):
        raise ValueError(
            "Signal and timestamps must have identical lengths."
        )


# ==========================================================
# Utility Functions
# ==========================================================

def sample_value(
    signal: np.ndarray,
    index: Optional[int],
) -> float:
    """
    Safely return the ECG amplitude at a sample index.
    """

    if index is None:
        return np.nan

    if index < 0:
        return np.nan

    if index >= len(signal):
        return np.nan

    return float(signal[index])


def interval_ms(
    start: Optional[int],
    end: Optional[int],
    sampling_rate: int,
) -> float:
    """
    Convert two sample indices into milliseconds.
    """

    if start is None:
        return np.nan

    if end is None:
        return np.nan

    if end <= start:
        return np.nan

    return (
        (end - start)
        / sampling_rate
        * 1000.0
    )


def rr_to_bpm(
    rr_ms: float,
) -> float:
    """
    Convert an RR interval into beats per minute.
    """

    if np.isnan(rr_ms):
        return np.nan

    if rr_ms <= 0:
        return np.nan

    return 60000.0 / rr_ms


# ==========================================================
# Fiducial Helpers
# ==========================================================

def _sanitize_indices(values) -> np.ndarray:
    """
    Convert NeuroKit outputs into a clean integer array.

    NeuroKit may return lists containing None or NaN.
    """

    if values is None:
        return np.array([], dtype=int)

    cleaned = []

    for value in values:

        if value is None:
            continue

        try:

            if np.isnan(value):
                continue

        except TypeError:
            pass

        cleaned.append(int(value))

    return np.asarray(cleaned, dtype=int)


def _match_wave(
    r_index: int,
    candidates: np.ndarray,
    before: int,
    after: int,
) -> Optional[int]:
    """
    Match the nearest fiducial to an R peak.

    Parameters
    ----------
    before
        Maximum samples before R.

    after
        Maximum samples after R.
    """

    if len(candidates) == 0:
        return None

    lower = r_index - before
    upper = r_index + after

    valid = candidates[
        (candidates >= lower)
        & (candidates <= upper)
    ]

    if len(valid) == 0:
        return None

    return int(
        valid[
            np.argmin(
                np.abs(valid - r_index)
            )
        ]
    )


# ==========================================================
# Physiological Search Windows
# ==========================================================

def search_windows(
    sampling_rate: int,
):
    """
    Convert physiological windows into samples.

    Returns
    -------
    dict
    """

    return {

        "P": (
            int(0.25 * sampling_rate),
            int(0.05 * sampling_rate),
        ),

        "Q": (
            int(0.08 * sampling_rate),
            0,
        ),

        "S": (
            0,
            int(0.08 * sampling_rate),
        ),

        "T": (
            int(0.02 * sampling_rate),
            int(0.45 * sampling_rate),
        ),
    }

# ==========================================================
# ECG Feature Detection
# ==========================================================

def detect_ecg_features(
    signal: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> tuple[list[ECGBeat], np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """
    Detect ECG fiducials using NeuroKit's complete processing
    pipeline.

    Parameters
    ----------
    signal
        Raw ECG waveform from LabScribe.

    timestamps
        Unix timestamps corresponding to each sample.

    sampling_rate
        ECG sampling frequency.

    Returns
    -------
    beats
        List of ECGBeat objects.

    raw_signal
        Original ECG waveform.

    clean_signal
        NeuroKit cleaned waveform.

    signals
        NeuroKit processed dataframe.

    info
        NeuroKit information dictionary.
    """

    validate_ecg_input(signal, timestamps)

    raw_signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps)

    # ------------------------------------------------------
    # Complete NeuroKit pipeline
    # ------------------------------------------------------

    try:

        signals, info = nk.ecg_process(
            raw_signal,
            sampling_rate=sampling_rate,
        )

    except Exception as error:

        raise RuntimeError(
            f"NeuroKit ECG processing failed:\n{error}"
        )

    clean_signal = signals["ECG_Clean"].to_numpy()

    r_peaks = np.asarray(
        info.get("ECG_R_Peaks", []),
        dtype=int,
    )

    print(f"Detected {len(r_peaks)} R peaks")

    if len(r_peaks):

        print(
            "First R peaks:",
            r_peaks[:10]
        )

    if len(r_peaks) < 2:

        return (
            [],
            raw_signal,
            clean_signal,
            signals,
            info,
        )

    # ------------------------------------------------------
    # ECG Delineation
    # ------------------------------------------------------

    try:

        _, waves = nk.ecg_delineate(
            clean_signal,
            r_peaks,
            sampling_rate=sampling_rate,
            method="dwt",
        )

    except Exception:

        print(
            "DWT delineation failed."
            " Falling back to peak method."
        )

        _, waves = nk.ecg_delineate(
            clean_signal,
            r_peaks,
            sampling_rate=sampling_rate,
            method="peak",
        )

    # ------------------------------------------------------
    # Fiducial arrays
    # ------------------------------------------------------

    p_peaks = _sanitize_indices(
        waves.get("ECG_P_Peaks")
    )

    q_peaks = _sanitize_indices(
        waves.get("ECG_Q_Peaks")
    )

    s_peaks = _sanitize_indices(
        waves.get("ECG_S_Peaks")
    )

    t_peaks = _sanitize_indices(
        waves.get("ECG_T_Peaks")
    )

    windows = search_windows(
        sampling_rate
    )

    beats: list[ECGBeat] = []

    previous_r = None

    # ------------------------------------------------------
    # Build beat list
    # ------------------------------------------------------

    for beat_number, r in enumerate(
        r_peaks,
        start=1,
    ):

        p = _match_wave(
            r,
            p_peaks,
            *windows["P"],
        )

        q = _match_wave(
            r,
            q_peaks,
            *windows["Q"],
        )

        s = _match_wave(
            r,
            s_peaks,
            *windows["S"],
        )

        t = _match_wave(
            r,
            t_peaks,
            *windows["T"],
        )

        # ----------------------------------------------
        # RR interval
        # ----------------------------------------------

        if previous_r is None:

            rr = np.nan
            bpm = np.nan

        else:

            rr = (
                (r - previous_r)
                / sampling_rate
                * 1000.0
            )

            bpm = rr_to_bpm(rr)

        previous_r = r

        beats.append(

            ECGBeat(

                beat_number=beat_number,

                r_index=int(r),

                p_index=p,
                q_index=q,
                s_index=s,
                t_index=t,

                rr_ms=rr,
                heart_rate_bpm=bpm,

                pr_ms=interval_ms(
                    p,
                    q,
                    sampling_rate,
                ),

                qrs_ms=interval_ms(
                    q,
                    s,
                    sampling_rate,
                ),

                qt_ms=interval_ms(
                    q,
                    t,
                    sampling_rate,
                ),

                # IMPORTANT:
                # Store amplitudes from the RAW waveform,
                # not the filtered waveform.

                p_value=sample_value(
                    raw_signal,
                    p,
                ),

                q_value=sample_value(
                    raw_signal,
                    q,
                ),

                r_value=sample_value(
                    raw_signal,
                    r,
                ),

                s_value=sample_value(
                    raw_signal,
                    s,
                ),

                t_value=sample_value(
                    raw_signal,
                    t,
                ),
            )
        )

    return (
        beats,
        raw_signal,
        clean_signal,
        signals,
        info,
    )

# ==========================================================
# Beat DataFrame
# ==========================================================

def beats_to_dataframe(
    beats: list[ECGBeat],
) -> pd.DataFrame:
    """
    Convert ECGBeat objects into a dataframe.

    Every amplitude is taken directly from the RAW ECG
    waveform stored inside each ECGBeat.
    """

    rows = []

    for beat in beats:

        rows.append({

            "Beat": beat.beat_number,

            "R_Index": beat.r_index,

            "P_Index": beat.p_index,
            "Q_Index": beat.q_index,
            "S_Index": beat.s_index,
            "T_Index": beat.t_index,

            "P_Value": beat.p_value,
            "Q_Value": beat.q_value,
            "R_Value": beat.r_value,
            "S_Value": beat.s_value,
            "T_Value": beat.t_value,

            "RR_ms": beat.rr_ms,
            "HeartRate_BPM": beat.heart_rate_bpm,

            "PR_ms": beat.pr_ms,
            "QRS_ms": beat.qrs_ms,
            "QT_ms": beat.qt_ms,
        })

    return pd.DataFrame(rows)


# ==========================================================
# Sample-by-Sample Analysis DataFrame
# ==========================================================

def create_ecg_analysis_dataframe(
    raw_signal: np.ndarray,
    timestamps: np.ndarray,
    beats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the sample-level ECG dataframe.

    Unlike the previous implementation, this stores the
    ORIGINAL LabScribe waveform instead of the filtered
    NeuroKit waveform.

    Heart-rate values are interpolated between beats.
    """

    df = pd.DataFrame({

        "UnixTime_ms": timestamps,

        "ECG": raw_signal,

    })

    df["HeartRate_BPM"] = np.nan
    df["Beat"] = np.nan

    # Fiducial markers
    df["P_Wave"] = 0
    df["Q_Wave"] = 0
    df["R_Wave"] = 0
    df["S_Wave"] = 0
    df["T_Wave"] = 0

    # Fiducial amplitudes
    df["P_Amplitude"] = np.nan
    df["Q_Amplitude"] = np.nan
    df["R_Amplitude"] = np.nan
    df["S_Amplitude"] = np.nan
    df["T_Amplitude"] = np.nan

    # ------------------------------------------------------
    # Insert beat information
    # ------------------------------------------------------

    for _, beat in beats_df.iterrows():

        r = int(beat["R_Index"])

        if r >= len(df):
            continue

        df.loc[r, "Beat"] = beat["Beat"]
        df.loc[r, "HeartRate_BPM"] = beat["HeartRate_BPM"]

        # ---------------------------
        # P
        # ---------------------------

        if not pd.isna(beat["P_Index"]):

            p = int(beat["P_Index"])

            if p < len(df):

                df.loc[p, "P_Wave"] = 1
                df.loc[p, "P_Amplitude"] = beat["P_Value"]

        # ---------------------------
        # Q
        # ---------------------------

        if not pd.isna(beat["Q_Index"]):

            q = int(beat["Q_Index"])

            if q < len(df):

                df.loc[q, "Q_Wave"] = 1
                df.loc[q, "Q_Amplitude"] = beat["Q_Value"]

        # ---------------------------
        # R
        # ---------------------------

        df.loc[r, "R_Wave"] = 1
        df.loc[r, "R_Amplitude"] = beat["R_Value"]

        # ---------------------------
        # S
        # ---------------------------

        if not pd.isna(beat["S_Index"]):

            s = int(beat["S_Index"])

            if s < len(df):

                df.loc[s, "S_Wave"] = 1
                df.loc[s, "S_Amplitude"] = beat["S_Value"]

        # ---------------------------
        # T
        # ---------------------------

        if not pd.isna(beat["T_Index"]):

            t = int(beat["T_Index"])

            if t < len(df):

                df.loc[t, "T_Wave"] = 1
                df.loc[t, "T_Amplitude"] = beat["T_Value"]

    # ------------------------------------------------------
    # Interpolate Heart Rate
    # ------------------------------------------------------

    if df["HeartRate_BPM"].notna().sum() > 1:

        df["HeartRate_BPM"] = (

            df["HeartRate_BPM"]

            .interpolate()

            .bfill()

            .ffill()

        )

    return df

# ==========================================================
# ECG Summary Statistics
# ==========================================================

def compute_ecg_summary(
    beats_df: pd.DataFrame,
    raw_signal: np.ndarray,
) -> dict:
    """
    Compute summary statistics for an ECG recording.
    """

    summary = {}

    # ------------------------------------------------------
    # Heart Rate
    # ------------------------------------------------------

    if len(beats_df):

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

        # ----------------------------------------------
        # ECG intervals
        # ----------------------------------------------

        for column in (
            "RR_ms",
            "PR_ms",
            "QRS_ms",
            "QT_ms",
        ):

            values = (
                beats_df[column]
                .dropna()
            )

            summary[f"Average{column}"] = (

                float(values.mean())

                if len(values)

                else np.nan

            )

    else:

        summary["AverageHeartRate"] = np.nan
        summary["MinimumHeartRate"] = np.nan
        summary["MaximumHeartRate"] = np.nan
        summary["HeartRateStd"] = np.nan

        summary["AverageRR_ms"] = np.nan
        summary["AveragePR_ms"] = np.nan
        summary["AverageQRS_ms"] = np.nan
        summary["AverageQT_ms"] = np.nan

    # ------------------------------------------------------
    # Recording statistics
    # ------------------------------------------------------

    summary["TotalHeartBeats"] = int(
        len(beats_df)
    )

    summary["ECG_Min"] = float(
        np.min(raw_signal)
    )

    summary["ECG_Max"] = float(
        np.max(raw_signal)
    )

    summary["ECG_Mean"] = float(
        np.mean(raw_signal)
    )

    summary["ECG_STD"] = float(
        np.std(raw_signal)
    )

    return summary


# ==========================================================
# Complete ECG Pipeline
# ==========================================================

def analyze_ecg(
    signal: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> ECGAnalysisResult:
    """
    Complete ECG analysis pipeline.

    This is the primary public entry point.
    """

    (
        beats,
        raw_signal,
        clean_signal,
        signals,
        info,

    ) = detect_ecg_features(

        signal,
        timestamps,
        sampling_rate,

    )

    beats_df = beats_to_dataframe(
        beats
    )

    analysis_df = create_ecg_analysis_dataframe(

        raw_signal,

        np.asarray(timestamps),

        beats_df,

    )

    summary = compute_ecg_summary(

        beats_df,

        raw_signal,

    )

    return ECGAnalysisResult(

        raw_signal=raw_signal,

        clean_signal=clean_signal,

        timestamps=np.asarray(timestamps),

        analysis_df=analysis_df,

        beats_df=beats_df,

        summary=summary,

        signals=signals,

        info=info,

    )

# ==========================================================
# ECG Plotting
# ==========================================================

def plot_ecg(
    raw_signal: np.ndarray,
    timestamps: np.ndarray,
    beats_df: pd.DataFrame,
    output_file: str,
    title: str = "ECG Analysis",
):
    """
    Plot the ORIGINAL LabScribe ECG waveform.

    Unlike the previous implementation, this function never
    plots the filtered NeuroKit signal.

    Rendered as a wide strip-chart, similar to LabScribe's
    live scrolling display: a fine millisecond-resolution
    grid (minor gridlines every ECG_MINOR_GRID_SECONDS,
    labeled major gridlines every ECG_MAJOR_GRID_SECONDS)
    and a figure width proportional to the recording length.
    """

    if len(raw_signal) == 0:
        return

    # ------------------------------------------------------
    # Time axis
    # ------------------------------------------------------

    time_seconds = (
        timestamps - timestamps[0]
    ) / 1000.0

    duration_seconds = float(
        time_seconds[-1]
    )

    # ------------------------------------------------------
    # Figure sizing
    #
    # Width scales with recording duration so the fine
    # millisecond grid stays legible, capped to keep the
    # image renderable/usable for very long recordings.
    # ------------------------------------------------------

    width_inches = (
        duration_seconds
        * ECG_PLOT_PIXELS_PER_SECOND
        / ECG_PLOT_DPI
    )

    width_inches = max(
        width_inches,
        ECG_PLOT_MIN_WIDTH_INCHES,
    )

    width_inches = min(
        width_inches,
        ECG_PLOT_MAX_WIDTH_INCHES,
    )

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            width_inches,
            ECG_PLOT_HEIGHT_INCHES,
        ),
        dpi=ECG_PLOT_DPI,
    )

    ax.plot(
        time_seconds,
        raw_signal,
        color="black",
        linewidth=0.8,
        label="Raw ECG",
    )

    # ------------------------------------------------------
    # Plot fiducials
    # ------------------------------------------------------

    marker_info = [

        ("P_Index", "P", "tab:blue"),
        ("Q_Index", "Q", "tab:orange"),
        ("R_Index", "R", "red"),
        ("S_Index", "S", "green"),
        ("T_Index", "T", "purple"),

    ]

    for column, label, color in marker_info:

        if column not in beats_df:
            continue

        indices = (

            beats_df[column]

            .dropna()

            .astype(int)

        )

        if len(indices) == 0:
            continue

        valid = indices[
            indices < len(raw_signal)
        ]

        if len(valid) == 0:
            continue

        ax.scatter(

            time_seconds[valid],

            raw_signal[valid],

            s=28,

            color=color,

            label=label,

            zorder=5,

        )

    # ------------------------------------------------------
    # LabScribe-style strip-chart grid
    #
    # Minor gridlines: fine millisecond-resolution "small
    # squares". Major gridlines: labeled, once per second.
    # ------------------------------------------------------

    major_locator = MultipleLocator(
        ECG_MAJOR_GRID_SECONDS
    )

    minor_locator = MultipleLocator(
        ECG_MINOR_GRID_SECONDS
    )

    # A fine millisecond-resolution grid over a multi-minute
    # recording easily produces more ticks than matplotlib's
    # default safety limit (Locator.MAXTICKS = 1000).
    major_locator.MAXTICKS = 50_000
    minor_locator.MAXTICKS = 50_000

    ax.xaxis.set_major_locator(
        major_locator
    )

    ax.xaxis.set_minor_locator(
        minor_locator
    )

    ax.grid(
        which="major",
        axis="x",
        color="firebrick",
        alpha=0.5,
        linewidth=0.7,
    )

    ax.grid(
        which="minor",
        axis="x",
        color="lightcoral",
        alpha=0.3,
        linewidth=0.4,
    )

    ax.grid(
        which="major",
        axis="y",
        color="lightgray",
        alpha=0.6,
        linewidth=0.6,
    )

    ax.set_xlim(
        time_seconds[0],
        time_seconds[-1],
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("ECG")
    ax.set_title(title)

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=ECG_PLOT_DPI,
    )

    plt.close(fig)

# ==========================================================
# CSV Export
# ==========================================================

def export_ecg_analysis(
    result: ECGAnalysisResult,
    output_file: str,
) -> None:
    """
    Export the sample-by-sample ECG dataframe.
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
    Export the heartbeat table.
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
    Export summary statistics.
    """

    pd.DataFrame(
        [result.summary]
    ).to_csv(
        output_file,
        index=False,
    )

# ==========================================================
# Public API
# ==========================================================

__all__ = [

    "ECGBeat",

    "ECGAnalysisResult",

    "analyze_ecg",

    "detect_ecg_features",

    "plot_ecg",

    "compute_ecg_summary",

    "export_ecg_analysis",

    "export_ecg_beats",

    "export_ecg_summary",

]