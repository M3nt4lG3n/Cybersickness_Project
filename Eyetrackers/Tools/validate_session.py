"""
validate_session.py

Validate a completed stereo eyetracker recording.

This script performs a post-recording quality check by examining:

    • Left video
    • Right video
    • Session CSV

It prints a summary to the console and also writes the same
report to a text file stored next to the CSV.

This is intended to be run AFTER recording an experimental session.
"""

from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import pandas as pd


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def find_file(folder, extensions, contains=None):
    """
    Finds the first file matching the requested extensions.

    Parameters
    ----------
    folder : str
    extensions : tuple[str]
    contains : str | None
    """

    for filename in os.listdir(folder):

        lower = filename.lower()

        if not lower.endswith(extensions):
            continue

        if contains is not None:

            if contains.lower() not in lower:
                continue

        return os.path.join(folder, filename)

    return None

def count_video(video_path):
    """
    Return information about a recorded video.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open:\n{video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return frame_count, fps, width, height


def duplicate_count(series):
    return int(series.duplicated().sum())


def missing_frames(series):

    values = sorted(series.tolist())

    missing = []

    expected = values[0]

    for value in values:

        while expected < value:
            missing.append(expected)
            expected += 1

        expected = value + 1

    return missing


def timestamp_wraps(series):
    """
    Count backwards jumps in timestamps.
    """

    return int(series.diff().lt(0).sum())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    root = tk.Tk()
    root.withdraw()

    print("Select Patient Recording Folder...")

    patient_folder = filedialog.askdirectory(
        title="Select Patient Recording Folder"
    )

    if not patient_folder:
        print("Validation cancelled.")
        return

    left_video = find_file(
        patient_folder,
        (".mp4",),
        "left"
    )

    right_video = find_file(
        patient_folder,
        (".mp4",),
        "right"
    )

    csv_file = find_file(
        patient_folder,
        (".csv",)
    )

    missing = []

    if left_video is None:
        missing.append("left video")

    if right_video is None:
        missing.append("right video")

    if csv_file is None:
        missing.append("CSV")

    if missing:

        print()

        print("Missing required files:")

        for item in missing:
            print(f"  - {item}")

        return

    report = []

    def log(message=""):
        print(message)
        report.append(message)

    validation_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log("=" * 70)
    log("Eyetrackers Session Validation Report")
    log("=" * 70)

    log(f"Validation Time : {validation_time}")
    log()

    #
    # -----------------------------------------------------
    # Video Information
    # -----------------------------------------------------
    #

    left_frames, left_fps, width, height = count_video(left_video)

    right_frames, right_fps, _, _ = count_video(right_video)

    #
    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------
    #

    df = pd.read_csv(csv_file)

    csv_rows = len(df)

    #
    # -----------------------------------------------------
    # Recording Duration
    # -----------------------------------------------------
    #

    duration_ms = (
        df["left_capture_ms"].iloc[-1]
        - df["left_capture_ms"].iloc[0]
    )

    duration_sec = duration_ms / 1000.0

    #
    # -----------------------------------------------------
    # Synchronization Statistics
    # -----------------------------------------------------
    #

    avg_delta = df["delta_ms"].mean()
    median_delta = df["delta_ms"].median()
    std_delta = df["delta_ms"].std()
    p95_delta = df["delta_ms"].quantile(0.95)
    max_delta = df["delta_ms"].max()

    #
    # -----------------------------------------------------
    # Latency Statistics
    # -----------------------------------------------------
    #

    avg_left_latency = df["left_latency_ms"].mean()
    avg_right_latency = df["right_latency_ms"].mean()

    max_left_latency = df["left_latency_ms"].max()
    max_right_latency = df["right_latency_ms"].max()

    #
    # -----------------------------------------------------
    # Continuity
    # -----------------------------------------------------
    #

    duplicate_left = duplicate_count(df["left_frame"])
    duplicate_right = duplicate_count(df["right_frame"])

    missing_left = missing_frames(df["left_frame"])
    missing_right = missing_frames(df["right_frame"])

    left_wraps = timestamp_wraps(df["left_capture_ms"])
    right_wraps = timestamp_wraps(df["right_capture_ms"])

    #
    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------
    #

    log("FILES")
    log("-" * 70)
    log(f"Left Video : {os.path.basename(left_video)}")
    log(f"Right Video: {os.path.basename(right_video)}")
    log(f"CSV        : {os.path.basename(csv_file)}")
    log()

    log("VIDEO")
    log("-" * 70)
    log(f"Resolution     : {width} x {height}")
    log(f"Left FPS       : {left_fps:.2f}")
    log(f"Right FPS      : {right_fps:.2f}")
    log(f"Duration       : {duration_sec:.2f} seconds")
    log()

    log("FRAME COUNTS")
    log("-" * 70)
    log(f"Left Frames    : {left_frames}")
    log(f"Right Frames   : {right_frames}")
    log(f"CSV Rows       : {csv_rows}")
    log()

    log("SYNCHRONIZATION")
    log("-" * 70)
    log(f"Average Delta  : {avg_delta:.2f} ms")
    log(f"Median Delta   : {median_delta:.2f} ms")
    log(f"95% Delta      : {p95_delta:.2f} ms")
    log(f"Maximum Delta  : {max_delta:.2f} ms")
    log(f"Std Dev Delta  : {std_delta:.2f} ms")
    log()

    log("LATENCY")
    log("-" * 70)
    log(f"Average Left   : {avg_left_latency:.2f} ms")
    log(f"Maximum Left   : {max_left_latency:.2f} ms")
    log(f"Average Right  : {avg_right_latency:.2f} ms")
    log(f"Maximum Right  : {max_right_latency:.2f} ms")
    log()

    log("CONTINUITY")
    log("-" * 70)
    log(f"Duplicate Left Frames : {duplicate_left}")
    log(f"Duplicate Right Frames: {duplicate_right}")
    log(f"Missing Left Frames   : {len(missing_left)}")
    log(f"Missing Right Frames  : {len(missing_right)}")
    log(f"Left Timestamp Wraps  : {left_wraps}")
    log(f"Right Timestamp Wraps : {right_wraps}")
    log()

    #
    # -----------------------------------------------------
    # Overall Assessment
    # -----------------------------------------------------
    #

    issues = []

    if left_frames != right_frames:
        issues.append("Left and right videos contain different numbers of frames.")

    if csv_rows != left_frames:
        issues.append("CSV row count does not match recorded video frames.")

    if duplicate_left:
        issues.append(f"{duplicate_left} duplicate left frame(s) detected.")

    if duplicate_right:
        issues.append(f"{duplicate_right} duplicate right frame(s) detected.")

    if len(missing_left):
        issues.append(f"{len(missing_left)} missing left frame(s).")

    if len(missing_right):
        issues.append(f"{len(missing_right)} missing right frame(s).")

    if left_wraps:
        issues.append("Left capture timestamps wrapped or moved backwards.")

    if right_wraps:
        issues.append("Right capture timestamps wrapped or moved backwards.")

    log("OVERALL ASSESSMENT")
    log("-" * 70)

    if not issues:

        log("No obvious recording issues were detected.")

    else:

        log("Potential issues detected:")

        for issue in issues:
            log(f"  • {issue}")

    log()

    log("Recording Quality Summary")
    log("-" * 70)
    log(f"Frame Count Agreement : {'Yes' if left_frames == right_frames else 'No'}")
    log(f"Timestamp Continuity  : {'Yes' if left_wraps == 0 and right_wraps == 0 else 'No'}")
    log(f"Duplicate Frames      : {'No' if duplicate_left == 0 and duplicate_right == 0 else 'Yes'}")
    log(f"Missing Frames        : {len(missing_left) + len(missing_right)}")
    log(f"Mean Sync Delta       : {avg_delta:.2f} ms")
    log(f"95th Percentile Delta : {p95_delta:.2f} ms")
    log()

    report_path = os.path.join(
        patient_folder,
        "validation.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        for line in report:
            f.write(line + "\n")

    print()
    print("=" * 70)
    print(f"Validation report saved to:\n{report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()