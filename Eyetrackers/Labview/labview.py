import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from tkinter import Tk, filedialog, simpledialog

import cv2
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


# ==========================================================
# Data Classes
# ==========================================================

@dataclass(slots=True)
class Beat:
    """
    Stores measurements for a single heartbeat.
    """

    beat_number: int

    unix_time_ms: int

    heart_rate_bpm: float

    rr_ms: float

    p_time_ms: float
    q_time_ms: float
    r_time_ms: float
    s_time_ms: float
    t_time_ms: float

    p_amp: float
    q_amp: float
    r_amp: float
    s_amp: float
    t_amp: float

    pr_ms: float
    qrs_ms: float
    qt_ms: float


@dataclass(slots=True)
class BalanceSample:
    """
    One balance-board sample.
    """

    tl: float
    tr: float
    bl: float
    br: float

    total_weight: float

    cop_x: float
    cop_y: float

    left_percent: float
    right_percent: float

    front_percent: float
    back_percent: float


# ==========================================================
# GUI
# ==========================================================

def select_csv() -> str:
    """
    Ask the user for the LabScribe CSV.
    """

    root = Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select LabScribe CSV",
        filetypes=[("CSV files", "*.csv")]
    )

    if not filename:
        raise SystemExit("No CSV selected.")

    return filename


def get_recording_start():
    """
    Ask the user what hour the recording began.

    The recording date comes from the CSV creation date.
    """

    hour = simpledialog.askinteger(
        "Recording Start",
        "Enter the recording start hour (1-12):",
        minvalue=1,
        maxvalue=12
    )

    if hour is None:
        raise SystemExit("Cancelled.")

    am_pm = simpledialog.askstring(
        "AM / PM",
        "Enter AM or PM:"
    )

    if am_pm is None:
        raise SystemExit("Cancelled.")

    am_pm = am_pm.strip().upper()

    if am_pm not in ("AM", "PM"):
        raise ValueError("Please enter AM or PM.")

    return hour, am_pm


# ==========================================================
# Loading
# ==========================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the LabScribe CSV.
    """

    df = pd.read_csv(csv_path)

    required_columns = [
        "Time",
        "TimeOfDay",
        "i1 2",
        "TL",
        "TR",
        "BL",
        "BR"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    return df


# ==========================================================
# Unix Time
# ==========================================================

def compute_unix_time(
    df: pd.DataFrame,
    csv_path: str,
    start_hour: int,
    am_pm: str,
) -> pd.DataFrame:
    """
    Compute UnixTime_ms using:

        file creation DATE
        +
        user supplied HOUR
        +
        elapsed Time column

    This avoids issues with TimeOfDay rollovers.
    """

    creation = datetime.fromtimestamp(
        os.path.getctime(csv_path)
    )

    hour = start_hour

    if am_pm == "PM" and hour != 12:
        hour += 12

    if am_pm == "AM" and hour == 12:
        hour = 0

    recording_start = datetime(
        creation.year,
        creation.month,
        creation.day,
        hour,
        0,
        0,
        0,
    )

    recording_start_ms = int(
        recording_start.timestamp() * 1000
    )

    elapsed_ms = (
        pd.to_numeric(
            df["Time"],
            errors="coerce",
        )
        * 1000
    )

    df["UnixTime_ms"] = (
        recording_start_ms
        + elapsed_ms
    ).astype("Int64")

    return df

# ==========================================================
# Balance Processing
# ==========================================================

def compute_balance_metrics(df: pd.DataFrame):
    """
    Compute balance-board metrics.

    New columns added:

        TotalWeight
        COP_X
        COP_Y

        LeftWeight
        RightWeight
        FrontWeight
        BackWeight

        LeftPercent
        RightPercent
        FrontPercent
        BackPercent

        COP_Distance
        COP_PathLength
        COP_Velocity
    """

    # ------------------------------------------------------
    # Convert to numeric
    # ------------------------------------------------------

    sensors = ["TL", "TR", "BL", "BR"]

    for column in sensors:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # ------------------------------------------------------
    # Total weight
    # ------------------------------------------------------

    df["TotalWeight"] = (
        df["TL"]
        + df["TR"]
        + df["BL"]
        + df["BR"]
    )

    # Prevent divide-by-zero
    total = df["TotalWeight"].replace(0, np.nan)

    # ------------------------------------------------------
    # Center of Pressure
    #
    #          Front
    #
    #   TL               TR
    #
    #
    #   BL               BR
    #
    #          Back
    # ------------------------------------------------------

    df["COP_X"] = (
        (
            df["TR"]
            + df["BR"]
            - df["TL"]
            - df["BL"]
        )
        / total
    )

    df["COP_Y"] = (
        (
            df["TL"]
            + df["TR"]
            - df["BL"]
            - df["BR"]
        )
        / total
    )

    df["COP_X"] = df["COP_X"].fillna(0)
    df["COP_Y"] = df["COP_Y"].fillna(0)

    # ------------------------------------------------------
    # Side weights
    # ------------------------------------------------------

    df["LeftWeight"] = (
        df["TL"]
        + df["BL"]
    )

    df["RightWeight"] = (
        df["TR"]
        + df["BR"]
    )

    df["FrontWeight"] = (
        df["TL"]
        + df["TR"]
    )

    df["BackWeight"] = (
        df["BL"]
        + df["BR"]
    )

    # ------------------------------------------------------
    # Percentages
    # ------------------------------------------------------

    df["LeftPercent"] = (
        100
        * df["LeftWeight"]
        / total
    )

    df["RightPercent"] = (
        100
        * df["RightWeight"]
        / total
    )

    df["FrontPercent"] = (
        100
        * df["FrontWeight"]
        / total
    )

    df["BackPercent"] = (
        100
        * df["BackWeight"]
        / total
    )

    percentage_columns = [
        "LeftPercent",
        "RightPercent",
        "FrontPercent",
        "BackPercent",
    ]

    for column in percentage_columns:
        df[column] = (
            df[column]
            .fillna(0)
            .clip(0, 100)
        )

    # ------------------------------------------------------
    # COP movement between samples
    # ------------------------------------------------------

    dx = df["COP_X"].diff()
    dy = df["COP_Y"].diff()

    df["COP_Distance"] = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    df["COP_Distance"] = (
        df["COP_Distance"]
        .fillna(0)
    )

    # ------------------------------------------------------
    # Running sway path length
    # ------------------------------------------------------

    df["COP_PathLength"] = (
        df["COP_Distance"]
        .cumsum()
    )

    # ------------------------------------------------------
    # Instantaneous sway velocity
    # ------------------------------------------------------

    dt = (
        pd.to_numeric(
            df["Time"],
            errors="coerce"
        )
        .diff()
    )

    velocity = (
        df["COP_Distance"]
        / dt.replace(0, np.nan)
    )

    df["COP_Velocity"] = (
        velocity
        .fillna(0)
        .replace(
            [np.inf, -np.inf],
            0
        )
    )

    return df

def nearest_wave(
    r_index: int,
    wave_indices,
    minimum_offset: int,
    maximum_offset: int,
):
    """
    Returns the wave nearest an R peak
    within the supplied search window.

    Returns None if no wave exists.
    """

    if wave_indices is None:
        return None

    candidates = []

    for w in wave_indices:

        if np.isnan(w):
            continue

        w = int(w)

        delta = w - r_index

        if minimum_offset <= delta <= maximum_offset:
            candidates.append(w)

    if len(candidates) == 0:
        return None

    return min(
        candidates,
        key=lambda x: abs(x - r_index)
    )

# ==========================================================
# ECG Processing
# ==========================================================

def detect_ecg_features(df: pd.DataFrame):
    """
    Detect ECG waves and compute beat information.

    Returns
    -------
    df
        Original dataframe with ECG columns added.

    beats_df
        One row per heartbeat.
    """

    # ------------------------------------------------------
    # Prepare ECG signal
    # ------------------------------------------------------

    working = df.copy()

    working["Time"] = pd.to_numeric(
        working["Time"],
        errors="coerce"
    )

    working["i1 2"] = pd.to_numeric(
        working["i1 2"],
        errors="coerce"
    )

    working = working.dropna(
        subset=[
            "Time",
            "i1 2"
        ]
    ).reset_index()

    if len(working) < 20:
        raise RuntimeError(
            "Not enough ECG samples."
        )

    dt = np.median(
        np.diff(
            working["Time"]
        )
    )

    sampling_rate = round(1 / dt)

    print(
        f"Detected sampling rate: "
        f"{sampling_rate} Hz"
    )

    ecg = working["i1 2"].to_numpy()

    # ------------------------------------------------------
    # NeuroKit processing
    # ------------------------------------------------------

    signals, info = nk.ecg_process(
        ecg,
        sampling_rate=sampling_rate
    )

    # ------------------------------------------------------
    # ECG delineation
    # ------------------------------------------------------

    signals, waves = nk.ecg_delineate(
        ecg,
        info["ECG_R_Peaks"],
        sampling_rate=sampling_rate,
        method="dwt"
    )

    r_peaks = np.asarray(
        info["ECG_R_Peaks"],
        dtype=int
    )

    print(
        f"Detected "
        f"{len(r_peaks)} R peaks."
    )

    # ------------------------------------------------------
    # Prepare output columns
    # ------------------------------------------------------

    output_columns = [
        "P_Wave",
        "Q_Wave",
        "R_Wave",
        "S_Wave",
        "T_Wave",

        "P_Amplitude",
        "Q_Amplitude",
        "R_Amplitude",
        "S_Amplitude",
        "T_Amplitude",

        "HeartRate_BPM",
        "RR_ms",
        "PR_ms",
        "QRS_ms",
        "QT_ms"
    ]

    for column in output_columns:

        if column not in df.columns:

            df[column] = np.nan

    # ------------------------------------------------------
    # Build beat table
    # ------------------------------------------------------

    beats = []

    previous_r_time = None

    p_peaks = waves.get("ECG_P_Peaks", [])
    q_peaks = waves.get("ECG_Q_Peaks", [])
    s_peaks = waves.get("ECG_S_Peaks", [])
    t_peaks = waves.get("ECG_T_Peaks", [])

    n_beats = min(
        len(r_peaks),
        len(p_peaks),
        len(q_peaks),
        len(s_peaks),
        len(t_peaks)
    )

    print(
        f"Using {n_beats} complete beats."
    )

    # ------------------------------------------------------
    # Process each beat
    # ------------------------------------------------------

    for beat_number in range(n_beats):

        r = int(r_peaks[beat_number])
        p = nearest_wave(
            r,
            p_peaks,
            -int(0.30 * sampling_rate),
            -int(0.05 * sampling_rate),
        )

        q = nearest_wave(
            r,
            q_peaks,
            -int(0.08 * sampling_rate),
            0,
        )

        s = nearest_wave(
            r,
            s_peaks,
            0,
            int(0.08 * sampling_rate),
        )

        t = nearest_wave(
            r,
            t_peaks,
            int(0.05 * sampling_rate),
            int(0.45 * sampling_rate),
        )

        if (
            min(p, q, r, s, t) < 0
            or
            max(p, q, r, s, t) >= len(working)
        ):
            continue

        unix_time = int(
            working.loc[r, "UnixTime_ms"]
        )

        r_time = working.loc[r, "Time"]

        if previous_r_time is None:

            rr_ms = np.nan
            hr = np.nan

        else:

            rr_ms = (
                r_time
                - previous_r_time
            ) * 1000

            hr = (
                60000 / rr_ms
                if rr_ms > 0
                else np.nan
            )

        previous_r_time = r_time

        pr_ms = (
            working.loc[q, "Time"]
            -
            working.loc[p, "Time"]
        ) * 1000

        qrs_ms = (
            working.loc[s, "Time"]
            -
            working.loc[q, "Time"]
        ) * 1000

        qt_ms = (
            working.loc[t, "Time"]
            -
            working.loc[q, "Time"]
        ) * 1000

        beat = {

            "BeatNumber": beat_number + 1,

            "UnixTime_ms": unix_time,

            "HeartRate_BPM": hr,

            "RR_ms": rr_ms,

            "P_Time_ms":
                working.loc[p, "Time"] * 1000,

            "Q_Time_ms":
                working.loc[q, "Time"] * 1000,

            "R_Time_ms":
                working.loc[r, "Time"] * 1000,

            "S_Time_ms":
                working.loc[s, "Time"] * 1000,

            "T_Time_ms":
                working.loc[t, "Time"] * 1000,

            "P_Amplitude": ecg[p],
            "Q_Amplitude": ecg[q],
            "R_Amplitude": ecg[r],
            "S_Amplitude": ecg[s],
            "T_Amplitude": ecg[t],

            "PR_ms": pr_ms,
            "QRS_ms": qrs_ms,
            "QT_ms": qt_ms
        }

        beats.append(beat)

        # ----------------------------------------------
        # Copy values into main dataframe
        # ----------------------------------------------

        original_index = working.loc[r, "index"]

        df.loc[original_index, "P_Wave"] = 1
        df.loc[original_index, "Q_Wave"] = 1
        df.loc[original_index, "R_Wave"] = 1
        df.loc[original_index, "S_Wave"] = 1
        df.loc[original_index, "T_Wave"] = 1

        df.loc[original_index, "P_Amplitude"] = ecg[p]
        df.loc[original_index, "Q_Amplitude"] = ecg[q]
        df.loc[original_index, "R_Amplitude"] = ecg[r]
        df.loc[original_index, "S_Amplitude"] = ecg[s]
        df.loc[original_index, "T_Amplitude"] = ecg[t]

        df.loc[original_index, "HeartRate_BPM"] = hr
        df.loc[original_index, "RR_ms"] = rr_ms
        df.loc[original_index, "PR_ms"] = pr_ms
        df.loc[original_index, "QRS_ms"] = qrs_ms
        df.loc[original_index, "QT_ms"] = qt_ms

    beats_df = pd.DataFrame(beats)

    return df, beats_df

# ==========================================================
# Output
# ==========================================================

def finalize_ecg_dataframe(
    df: pd.DataFrame,
    beats_df: pd.DataFrame,
):
    """
    Fill continuous ECG values into the main dataframe.
    """

    if beats_df.empty:
        return df, beats_df

    # ------------------------------------------------------
    # Heart Rate interpolation
    # ------------------------------------------------------

    hr = df["HeartRate_BPM"].copy()

    # interpolate between beats
    hr = hr.interpolate(
        method="linear",
        limit_direction="both"
    )

    # fill beginning/end if necessary
    hr = hr.ffill().bfill()

    df["HeartRate_BPM"] = hr

    # ------------------------------------------------------
    # Ensure ECG columns exist
    # ------------------------------------------------------

    ecg_columns = [
        "P_Wave",
        "Q_Wave",
        "R_Wave",
        "S_Wave",
        "T_Wave",
        "P_Amplitude",
        "Q_Amplitude",
        "R_Amplitude",
        "S_Amplitude",
        "T_Amplitude",
        "RR_ms",
        "PR_ms",
        "QRS_ms",
        "QT_ms"
    ]

    for column in ecg_columns:

        if column not in df.columns:
            df[column] = np.nan

    return df, beats_df


# ==========================================================
# Saving
# ==========================================================

def save_outputs(
    csv_path: str,
    df: pd.DataFrame,
    beats_df: pd.DataFrame,
):
    """
    Save all CSV outputs.
    """

    base = os.path.splitext(csv_path)[0]

    combined_csv = base + "_analysis.csv"
    beats_csv = base + "_beats.csv"
    summary_csv = base + "_summary.csv"

    # ------------------------------------------------------
    # Combined dataframe
    # ------------------------------------------------------

    df.to_csv(
        combined_csv,
        index=False
    )

    print(
        f"Saved analysis CSV:\n"
        f"{combined_csv}"
    )

    # ------------------------------------------------------
    # Beat dataframe
    # ------------------------------------------------------

    beats_df.to_csv(
        beats_csv,
        index=False
    )

    print(
        f"Saved beat CSV:\n"
        f"{beats_csv}"
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {}

    summary["Samples"] = len(df)

    summary["RecordingLength_s"] = (
        df["Time"].max()
        - df["Time"].min()
    )

    summary["AverageHeartRate"] = (
        df["HeartRate_BPM"].mean()
    )

    summary["MinimumHeartRate"] = (
        df["HeartRate_BPM"].min()
    )

    summary["MaximumHeartRate"] = (
        df["HeartRate_BPM"].max()
    )

    summary["AverageWeight"] = (
        df["TotalWeight"].mean()
    )

    summary["AverageCOP_X"] = (
        df["COP_X"].mean()
    )

    summary["AverageCOP_Y"] = (
        df["COP_Y"].mean()
    )

    summary["TotalCOPPathLength"] = (
        df["COP_PathLength"].iloc[-1]
    )

    summary["AverageCOPVelocity"] = (
        df["COP_Velocity"].mean()
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_csv,
        index=False
    )

    print(
        f"Saved summary CSV:\n"
        f"{summary_csv}"
    )

    return (
        combined_csv,
        beats_csv,
        summary_csv,
    )

# ==========================================================
# ECG Visualization
# ==========================================================

def plot_ecg(
    csv_path: str,
    df: pd.DataFrame,
):
    """
    Generate a publication-quality ECG plot.
    """

    base = os.path.splitext(csv_path)[0]
    output = base + "_ecg.png"

    plot_df = df.copy()

    plot_df["Time_ms"] = (
        pd.to_numeric(
            plot_df["Time"],
            errors="coerce"
        ) * 1000
    )

    plot_df["i1 2"] = pd.to_numeric(
        plot_df["i1 2"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=[
            "Time_ms",
            "i1 2"
        ]
    )

    fig, ax1 = plt.subplots(
        figsize=(18, 8)
    )

    # --------------------------------------------------
    # ECG Signal
    # --------------------------------------------------

    ax1.plot(
        plot_df["Time_ms"],
        plot_df["i1 2"],
        linewidth=0.8,
        label="ECG"
    )

    ymin = plot_df["i1 2"].min()
    ymax = plot_df["i1 2"].max()

    margin = (ymax - ymin) * 0.05

    if margin == 0:
        margin = 1

    ax1.set_ylim(
        ymin - margin,
        ymax + margin
    )

    # --------------------------------------------------
    # Marker helper
    # --------------------------------------------------

    def draw_wave(column, amp_column, label, marker):

        subset = plot_df[
            plot_df[column] == 1
        ]

        if subset.empty:
            return

        ax1.scatter(
            subset["Time_ms"],
            subset[amp_column],
            s=35,
            marker=marker,
            label=label
        )

    draw_wave(
        "P_Wave",
        "P_Amplitude",
        "P",
        "o"
    )

    draw_wave(
        "Q_Wave",
        "Q_Amplitude",
        "Q",
        "v"
    )

    draw_wave(
        "R_Wave",
        "R_Amplitude",
        "R",
        "^"
    )

    draw_wave(
        "S_Wave",
        "S_Amplitude",
        "S",
        "s"
    )

    draw_wave(
        "T_Wave",
        "T_Amplitude",
        "T",
        "D"
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax1.set_xlabel(
        "Time (ms)"
    )

    ax1.set_ylabel(
        "ECG Amplitude"
    )

    ax1.grid(True)

    # --------------------------------------------------
    # Heart Rate Overlay
    # --------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        plot_df["Time_ms"],
        plot_df["HeartRate_BPM"],
        linewidth=1.0,
        alpha=0.8,
        label="Heart Rate"
    )

    ax2.set_ylabel(
        "Heart Rate (BPM)"
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right"
    )

    plt.title(
        "ECG Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved ECG graph:\n{output}"
    )

# ==========================================================
# ECG Window
# ==========================================================

def draw_ecg_window(
    frame,
    width,
    height,
    window_df,
):
    """
    Draw a scrolling ECG monitor.
    """

    if len(window_df) < 2:
        return

    #
    # ECG Grid
    #

    grid_minor = 20
    grid_major = grid_minor * 5

    minor_color = (235, 235, 235)
    major_color = (200, 200, 200)

    for x in range(0, width, grid_minor):
        cv2.line(
            frame,
            (x, 0),
            (x, height),
            minor_color,
            1,
        )

    for y in range(0, height, grid_minor):
        cv2.line(
            frame,
            (0, y),
            (width, y),
            minor_color,
            1,
        )

    for x in range(0, width, grid_major):
        cv2.line(
            frame,
            (x, 0),
            (x, height),
            major_color,
            1,
        )

    for y in range(0, height, grid_major):
        cv2.line(
            frame,
            (0, y),
            (width, y),
            major_color,
            1,
        )

    left = 20
    right = width - 20

    top = 20
    bottom = height - 20

    times = window_df["Time"].to_numpy()
    signal = window_df["i1 2"].to_numpy()

    t0 = times[0]
    t1 = times[-1]

    ymin = signal.min()
    ymax = signal.max()

    if ymax == ymin:
        ymax += 1

    pts = []

    for t, y in zip(times, signal):

        x = int(
            np.interp(
                t,
                [t0, t1],
                [left, right],
            )
        )

        yy = int(
            np.interp(
                y,
                [ymin, ymax],
                [bottom, top],
            )
        )

        pts.append((x, yy))

    #
    # ECG Trace
    #

    for i in range(len(pts) - 1):

        cv2.line(
            frame,
            pts[i],
            pts[i + 1],
            (0, 180, 0),
            2,
        )

    #
    # Wave markers
    #

    markers = [
        ("P_Wave", "P", (255, 0, 255)),
        ("Q_Wave", "Q", (255, 0, 0)),
        ("R_Wave", "R", (0, 0, 255)),
        ("S_Wave", "S", (255, 140, 0)),
        ("T_Wave", "T", (0, 170, 255)),
    ]

    for column, label, color in markers:

        subset = window_df[
            window_df[column] == 1
        ]

        for _, r in subset.iterrows():

            x = int(
                np.interp(
                    r["Time"],
                    [t0, t1],
                    [left, right],
                )
            )

            y = int(
                np.interp(
                    r["i1 2"],
                    [ymin, ymax],
                    [bottom, top],
                )
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x - 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    #
    # Current cursor
    #

    cv2.line(
        frame,
        (right, top),
        (right, bottom),
        (0, 0, 255),
        2,
    )

    #
    # Heart rate
    #

    hr = window_df["HeartRate_BPM"].iloc[-1]

    if not np.isnan(hr):

        cv2.putText(
            frame,
            f"{hr:.1f} BPM",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

# ==========================================================
# Balance / ECG Video
# ==========================================================

def create_balance_video(
    csv_path,
    df,
):

    output = (
        os.path.splitext(csv_path)[0]
        + "_balance.mp4"
    )

    width = 1280
    height = 720

    fps = round(
        1 /
        np.median(
            np.diff(df["Time"])
        )
    )

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ecg_window = 6.0

    for index, row in df.iterrows():

        frame = np.full(
            (height, width, 3),
            245,
            np.uint8,
        )

        #
        # Timestamp
        #

        cv2.putText(
            frame,
            f"Unix: {int(row['UnixTime_ms'])}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Time: {row['Time']:.3f} s",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

                # --------------------------------------------------
        # Balance Board Layout
        # --------------------------------------------------

        board_left = 80
        board_top = 120

        board_size = 360

        board_right = board_left + board_size
        board_bottom = board_top + board_size

        cv2.rectangle(
            frame,
            (board_left, board_top),
            (board_right, board_bottom),
            (40, 40, 40),
            3,
        )

        sensor_radius = 45

        tl_pos = (
            board_left,
            board_top,
        )

        tr_pos = (
            board_right,
            board_top,
        )

        bl_pos = (
            board_left,
            board_bottom,
        )

        br_pos = (
            board_right,
            board_bottom,
        )

        sensors = [
            ("TL", tl_pos),
            ("TR", tr_pos),
            ("BL", bl_pos),
            ("BR", br_pos),
        ]

        #
        # Draw sensor pads
        #

        maximum_force = max(
            df["TL"].max(),
            df["TR"].max(),
            df["BL"].max(),
            df["BR"].max(),
        )

        if maximum_force <= 0:
            maximum_force = 1

        for label, position in sensors:

            value = float(row[label])

            radius = int(
                np.interp(
                    value,
                    [0, maximum_force],
                    [15, sensor_radius],
                )
            )

            ratio = np.clip(
                value / maximum_force,
                0,
                1,
            )

            if ratio < 0.25:

                color = (255,120,0)

            elif ratio < 0.50:

                color = (0,220,0)

            elif ratio < 0.75:

                color = (0,255,255)

            else:

                color = (0,0,255)

            cv2.circle(
                frame,
                position,
                radius,
                color,
                -1,
            )

            cv2.circle(
                frame,
                position,
                sensor_radius,
                (40, 40, 40),
                2,
            )

            cv2.putText(
                frame,
                label,
                (
                    position[0] - 15,
                    position[1] - 60,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{value:.1f}",
                (
                    position[0] - 28,
                    position[1] + 75,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        # --------------------------------------------------
        # Center of Pressure
        # --------------------------------------------------

        cop_x = float(row["COP_X"])
        cop_y = float(row["COP_Y"])

        pixel_x = int(
            np.interp(
                cop_x,
                [-1, 1],
                [board_left, board_right],
            )
        )

        pixel_y = int(
            np.interp(
                cop_y,
                [-1, 1],
                [board_bottom, board_top],
            )
        )

        #
        # Trail
        #

        history = df.iloc[
            max(0, index - 180):index + 1
        ]

        trail = []

        for _, h in history.iterrows():

            tx = int(
                np.interp(
                    h["COP_X"],
                    [-1, 1],
                    [board_left, board_right],
                )
            )

            ty = int(
                np.interp(
                    h["COP_Y"],
                    [-1, 1],
                    [board_bottom, board_top],
                )
            )

            trail.append((tx, ty))

        for i in range(len(trail) - 1):

            alpha = (i + 1) / len(trail)

            color = (
                int(180 * alpha),
                int(255 * alpha),
                0,
            )

            cv2.line(
                frame,
                trail[i],
                trail[i + 1],
                color,
                2,
            )

        #
        # Current COP
        #

        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            10,
            (0, 0, 255),
            -1,
        )

        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            14,
            (255, 255, 255),
            2,
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        stats_x = 520
        stats_y = 150

        statistics = [

            f"Total Weight : {row['TotalWeight']:.2f}",

            "",

            f"Left  : {row['LeftPercent']:.1f} %",
            f"Right : {row['RightPercent']:.1f} %",

            "",

            f"Front : {row['FrontPercent']:.1f} %",
            f"Back  : {row['BackPercent']:.1f} %",

            "",

            f"COP X : {row['COP_X']:.3f}",
            f"COP Y : {row['COP_Y']:.3f}",

            "",

            f"Heart Rate : {row['HeartRate_BPM']:.1f} BPM",
        ]

        for line in statistics:

            if line == "":
                stats_y += 18
                continue

            cv2.putText(
                frame,
                line,
                (stats_x, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (20, 20, 20),
                2,
            )

            stats_y += 34

        # --------------------------------------------------
        # ECG Window
        # --------------------------------------------------

        current_time = row["Time"]

        window = df[
            (df["Time"] >= current_time - ecg_window)
            &
            (df["Time"] <= current_time)
        ]

        ecg_frame = frame[
            420:690,
            470:1240,
        ]

        draw_ecg_window(
            ecg_frame,
            ecg_frame.shape[1],
            ecg_frame.shape[0],
            window,
        )

        cv2.rectangle(
            frame,
            (470, 420),
            (1240, 690),
            (0, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            "ECG",
            (480, 410),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        writer.write(frame)
    
    writer.release()

    print(f"Saved visualization video:\n{output}")

# ==========================================================
# Main
# ==========================================================

def main():

    #
    # Load
    #

    csv_path = select_csv()

    start_hour, am_pm = get_recording_start()

    df = load_csv(csv_path)

    #
    # Time
    #

    df = compute_unix_time(
        df,
        csv_path,
        start_hour,
        am_pm,
    )

    #
    # Balance
    #

    df = compute_balance_metrics(df)

    #
    # ECG
    #

    df, beats = detect_ecg_features(df)

    df, beats = finalize_ecg_dataframe(
        df,
        beats,
    )

    #
    # Save
    #

    save_outputs(
        csv_path,
        df,
        beats,
    )

    #
    # Figures
    #

    plot_ecg(
        csv_path,
        df,
    )

    create_balance_video(
        csv_path,
        df,
    )

    print()

    print("======================================")
    print("Analysis Complete")
    print("======================================")



if __name__ == "__main__":
    main()