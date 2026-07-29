"""
Pupil Visualizer
================

Plays back <eye>_pupil_readings.csv (or the previously-saved
<eye>_pupil_readings_corrected.csv) recordings from 4 patient sub-folders
simultaneously, synced by frame number, in a 2-row x 4-column grid
(row 0 = LEFT eye, row 1 = RIGHT eye, one column per sub-folder/trial).

Playback is now paced using each frame's real CaptureTimestampMs delta
(like playback.py) instead of a fixed frame delay, so play speed reflects
the camera's actual capture cadence. All 8 streams (4 trials x 2 eyes)
advance on one shared frame index -- the inter-frame *wait time* between
that shared index and the next is derived from the CSV timestamps, but
the index itself stays common across streams (this keeps the grid,
slider, and marking system simple, and mirrors how the original tool's
frame_idx already worked). If your trials are not timestamp-aligned to
each other, the effective pacing follows the slowest stream at each step.

EXPECTED DIRECTORY LAYOUT
--------------------------
<Patient Parent Directory>/
    <subfolder_1>/
        left_pupil_readings.csv   (or left_pupil_readings_corrected.csv)
        right_pupil_readings.csv  (or right_pupil_readings_corrected.csv)
        left_eye_cropped.mp4
        right_eye_cropped.mp4
        Patient_<Number>_<Value.Value>.iwxdata
    <subfolder_2>/ ... (same files)
    <subfolder_3>/ ...
    <subfolder_4>/ ...

EXPECTED CSV COLUMNS (base / "uncorrected" data)
-------------------------------------------------
    Frame, Time_ms, CenterX, CenterY, MajorDiameter, MinorDiameter,
    Area, Angle, FrameNumber, CaptureTimestampMs, ReceiveTimestampMs

A "_corrected" CSV additionally carries:
    Corrected_CenterX, Corrected_CenterY, Corrected_MajorDiameter,
    Corrected_MinorDiameter, Corrected_Area, Corrected_Angle, Marked

CONTROLS
--------
    space        : play / pause
    left arrow   : step one frame backward (pauses playback)
    right arrow  : step one frame forward  (pauses playback)
    slider       : jump to any frame (works while playing or paused)
    u            : toggle underlay of original *_eye_cropped.mp4 video
    d            : toggle debug overlay (frame #, raw + corrected values)
    e            : toggle automatic jitter/error-handling correction
    /            : cycle playback speed through 1x -> 2x -> 3x -> 4x -> 1x
    click MARK   : per-video button (bottom of each cell) -- marks the
                   currently displayed frame for correction, or unmarks
                   it if already marked
    s            : save a corrected CSV for every loaded stream (into
                   each trial's own folder) and close the app
    q / ESC      : quit without saving

All toggles (u/d/e//) work identically whether playback is running or
paused.

COLOR KEY
---------
    green ellipse       : raw, unmodified detector output for this frame
    light-blue ellipse   : the "corrected" value for this frame -- shown
                           whenever automatic error-handling substituted
                           a reading, and/or the frame has been manually
                           marked. Both ellipses are hollow outlines,
                           matching pupils.py's drawing style.

REQUIREMENTS
------------
    pip install opencv-python pandas numpy
    (tkinter ships with most Python installs; on some Linux distros you
     may need `sudo apt-get install python3-tk`)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2

# =====================================================================
# CONFIG
# =====================================================================

# Starting state for automatic jitter / error-handling correction.
# When True, sudden implausible drops in pupil diameter (acquisition
# jitter) are detected at load time and the previous good reading is
# substituted for that frame. Toggle live in-app with 'e'.
DEFAULT_ENABLE_ERROR_HANDLING = True

# If a frame's average diameter falls below this fraction of the
# previous accepted diameter, it is treated as a jitter/dropout frame.
JITTER_SHRINK_RATIO = 0.55

# --- Layout -----------------------------------------------------------
GRID_ROWS = 2       # LEFT eye row, RIGHT eye row
GRID_COLS = 4        # one column per sub-folder/trial (always 4)
CELL_WIDTH = 320
CELL_HEIGHT = 240

# --- Mark button geometry (bottom-center of each cell) -----------------
MARK_BTN_W = 70
MARK_BTN_H = 18
MARK_BTN_X0 = (CELL_WIDTH - MARK_BTN_W) // 2
MARK_BTN_Y0 = CELL_HEIGHT - MARK_BTN_H - 4
MARK_BTN_X1 = MARK_BTN_X0 + MARK_BTN_W
MARK_BTN_Y1 = MARK_BTN_Y0 + MARK_BTN_H

# --- Playback -----------------------------------------------------------
DEFAULT_FRAME_DELAY_MS = 100  # fallback pacing when no timestamps are available
PLAYBACK_SPEEDS = [1.0, 2.0, 3.0, 4.0]

# --- Colors (BGR) -------------------------------------------------------
RAW_ELLIPSE_COLOR = (55, 255, 0)          # green, matches pupils.py
CORRECTED_ELLIPSE_COLOR = (255, 220, 120)  # light blue
MARK_BTN_ON_COLOR = (255, 220, 120)
MARK_BTN_OFF_COLOR = (90, 90, 90)

BASE_CSV_COLUMNS = [
    "Frame", "Time_ms", "CenterX", "CenterY", "MajorDiameter",
    "MinorDiameter", "Area", "Angle", "FrameNumber",
    "CaptureTimestampMs", "ReceiveTimestampMs",
]
CORRECTED_VALUE_COLUMNS = [
    "CenterX", "CenterY", "MajorDiameter", "MinorDiameter", "Area", "Angle",
]


# =====================================================================
# Directory / file helpers
# =====================================================================

def select_directory(title):
    """Prompt the user for a directory via a native folder-picker dialog,
    falling back to a console prompt if tkinter isn't available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception:
        path = input(f"{title}\nEnter path: ").strip()

    if not path:
        print("No directory selected. Exiting.")
        sys.exit(1)
    return path


def ask_use_corrected_csv():
    """Ask once, up front, whether previously-saved *_corrected.csv files
    should be loaded (when present) instead of the raw readings files."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        use_corrected = messagebox.askyesno(
            "Load corrected CSVs?",
            "Use previously saved *_corrected.csv files where they exist?\n\n"
            "Yes: load prior corrections/marks and continue from them.\n"
            "No: start fresh from the raw *_pupil_readings.csv files.",
        )
        root.destroy()
        return bool(use_corrected)
    except Exception:
        answer = input("Use corrected CSVs where available? [y/N]: ").strip().lower()
        return answer.startswith("y")


def find_iwxdata_value(folder):
    """Locate the *.iwxdata file in `folder` and extract the Value.Value
    segment from a filename formatted as Patient_Number_Value.Value"""
    matches = glob.glob(os.path.join(folder, "*.iwxdata"))
    if not matches:
        return "N/A"

    base = os.path.splitext(os.path.basename(matches[0]))[0]
    parts = base.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    return base


def raw_csv_path(folder, eye):
    return os.path.join(folder, f"{eye}_pupil_readings.csv")


def corrected_csv_path(folder, eye):
    return os.path.join(folder, f"{eye}_pupil_readings_corrected.csv")


def load_csv_for_stream(folder, eye, prefer_corrected):
    """Pick the raw or corrected CSV for this stream and load it."""
    corrected_path = corrected_csv_path(folder, eye)
    raw_path = raw_csv_path(folder, eye)

    path = None
    if prefer_corrected and os.path.exists(corrected_path):
        path = corrected_path
    elif os.path.exists(raw_path):
        path = raw_path
    elif os.path.exists(corrected_path):
        # Only a corrected file exists -- use it even if not explicitly requested.
        path = corrected_path

    if path is None:
        print(f"  [warning] no CSV found for {eye} in {folder}")
        return None, None

    missing = [c for c in BASE_CSV_COLUMNS if c not in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path)
    if missing:
        print(f"  [warning] {os.path.basename(path)} missing columns: {missing}")

    df = df.sort_values("Frame").reset_index(drop=True)
    df = df.set_index("Frame", drop=False)
    return df, path


# =====================================================================
# Automatic jitter/error-handling precompute
# =====================================================================

def precompute_corrections(df):
    """Runs the jitter/error-handling substitution once, sequentially,
    over the whole stream, and bakes the result into new columns so
    rendering/scrubbing never needs to re-walk history. Also ensures a
    'Marked' column exists, preserving any values already present (i.e.
    loaded from a previously-saved corrected CSV)."""

    n = len(df)
    auto_values = {c: np.full(n, np.nan) for c in CORRECTED_VALUE_COLUMNS}
    auto_flag = np.zeros(n, dtype=bool)
    diam_ratio = np.full(n, np.nan)  # new_diam / prev_good_diam, for diagnostics

    last_good = None
    for i in range(n):
        row = df.iloc[i]
        raw = {c: row[c] for c in CORRECTED_VALUE_COLUMNS}
        has_nan = any(pd.isna(v) for v in raw.values())

        if has_nan:
            if last_good is not None:
                for c in CORRECTED_VALUE_COLUMNS:
                    auto_values[c][i] = last_good[c]
                auto_flag[i] = True
            continue

        if last_good is not None:
            prev_diam = (last_good["MajorDiameter"] + last_good["MinorDiameter"]) / 2.0
            new_diam = (raw["MajorDiameter"] + raw["MinorDiameter"]) / 2.0
            if prev_diam > 0:
                diam_ratio[i] = new_diam / prev_diam
            if DEFAULT_ENABLE_ERROR_HANDLING and prev_diam > 0 and new_diam < prev_diam * JITTER_SHRINK_RATIO:
                for c in CORRECTED_VALUE_COLUMNS:
                    auto_values[c][i] = last_good[c]
                auto_flag[i] = True
                continue

        for c in CORRECTED_VALUE_COLUMNS:
            auto_values[c][i] = raw[c]
        last_good = raw

    for c in CORRECTED_VALUE_COLUMNS:
        df[f"AutoCorrected_{c}"] = auto_values[c]
    df["AutoCorrectedFlag"] = auto_flag
    df["_DiamRatio"] = diam_ratio  # internal-only, not written to output CSV

    if "Marked" not in df.columns:
        df["Marked"] = False
    else:
        df["Marked"] = df["Marked"].fillna(False).astype(bool)

    return df


# =====================================================================
# EyeStream: everything needed to render/save one eye of one sub-folder
# =====================================================================

class EyeStream:
    def __init__(self, folder, eye, prefer_corrected, stream_id):
        self.folder = folder
        self.eye = eye  # "left" or "right"
        self.stream_id = stream_id
        self.label = f"{os.path.basename(os.path.normpath(folder))} / {eye.upper()}"

        video_path = os.path.join(folder, f"{eye}_eye_cropped.mp4")

        self.df, self.source_path = load_csv_for_stream(folder, eye, prefer_corrected)
        if self.df is not None:
            self.df = precompute_corrections(self.df)

        self.cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
        if self.cap is not None and not self.cap.isOpened():
            print(f"  [warning] could not open video: {video_path}")
            self.cap = None

        self.iwx_value = find_iwxdata_value(folder)

        self.max_frame = int(self.df["Frame"].max()) if self.df is not None else 0
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else CELL_WIDTH
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else CELL_HEIGHT
        if self.video_w <= 0:
            self.video_w = CELL_WIDTH
        if self.video_h <= 0:
            self.video_h = CELL_HEIGHT

        self.last_video_idx = -1

    # ---- video frame retrieval (sequential-aware for speed) ----
    def get_video_frame(self, frame_idx):
        if self.cap is None:
            return None
        if frame_idx != self.last_video_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        self.last_video_idx = frame_idx if ok else self.last_video_idx
        return frame if ok else None

    # ---- csv row retrieval ----
    def get_row(self, frame_idx):
        if self.df is None or frame_idx not in self.df.index:
            return None
        return self.df.loc[frame_idx]

    # ---- resolved values for rendering / debug text ----
    def get_render_payload(self, frame_idx, error_handling_on):
        row = self.get_row(frame_idx)
        if row is None:
            return None

        raw_values = {c: row[c] for c in CORRECTED_VALUE_COLUMNS}
        raw_values["Time_ms"] = row["Time_ms"]
        if any(pd.isna(v) for k, v in raw_values.items() if k != "Time_ms"):
            raw_values = None

        auto_flag = bool(row["AutoCorrectedFlag"]) and error_handling_on
        marked = bool(row["Marked"])

        if auto_flag:
            corrected_values = {c: row[f"AutoCorrected_{c}"] for c in CORRECTED_VALUE_COLUMNS}
        elif raw_values is not None:
            corrected_values = {c: raw_values[c] for c in CORRECTED_VALUE_COLUMNS}
        else:
            corrected_values = None

        show_corrected = (auto_flag or marked) and corrected_values is not None

        return {
            "raw": raw_values,
            "corrected": corrected_values,
            "auto_flag": auto_flag,
            "marked": marked,
            "show_corrected": show_corrected,
        }

    # ---- manual marking ----
    def toggle_mark(self, frame_idx, interaction_log):
        if self.df is None or frame_idx not in self.df.index:
            return
        new_state = not bool(self.df.loc[frame_idx, "Marked"])
        self.df.loc[frame_idx, "Marked"] = new_state
        interaction_log.append((self.stream_id, frame_idx, new_state))
        print(f"[{self.label}] frame {frame_idx} {'MARKED' if new_state else 'unmarked'}")

    # ---- save corrected csv ----
    def save_corrected(self, error_handling_on):
        if self.df is None:
            return None

        out_df = self.df[BASE_CSV_COLUMNS].copy()

        for c in CORRECTED_VALUE_COLUMNS:
            if error_handling_on:
                use_auto = self.df["AutoCorrectedFlag"]
                out_df[f"Corrected_{c}"] = np.where(
                    use_auto, self.df[f"AutoCorrected_{c}"], self.df[c]
                )
            else:
                out_df[f"Corrected_{c}"] = self.df[c]

        out_df["Marked"] = self.df["Marked"].astype(bool)

        out_path = corrected_csv_path(self.folder, self.eye)
        out_df.to_csv(out_path, index=False)
        return out_path


# =====================================================================
# Jitter-ratio suggestion (advisory only, printed at close)
# =====================================================================

def suggest_jitter_ratio(streams, interaction_log, current_ratio):
    missed_ratios = []   # frames manually marked, but auto-detection did not flag them
    false_pos_ratios = []  # frames auto-flagged, then explicitly unmarked by the user

    unmark_events = {(sid, fidx) for sid, fidx, state in interaction_log if state is False}
    mark_events = {(sid, fidx) for sid, fidx, state in interaction_log if state is True}

    for stream in streams:
        if stream.df is None:
            continue
        for frame_idx, row in stream.df.iterrows():
            ratio = row["_DiamRatio"]
            if pd.isna(ratio):
                continue
            key = (stream.stream_id, frame_idx)
            if bool(row["Marked"]) and not bool(row["AutoCorrectedFlag"]) and key in mark_events:
                missed_ratios.append(ratio)
            if key in unmark_events and bool(row["AutoCorrectedFlag"]):
                false_pos_ratios.append(ratio)

    if not missed_ratios and not false_pos_ratios:
        return None, "No manual marks/unmarks were made this session -- no change suggested."

    required_min = max(missed_ratios) * 1.02 if missed_ratios else None
    required_max = min(false_pos_ratios) * 0.98 if false_pos_ratios else None

    if required_min is not None and required_max is not None:
        if required_min < required_max:
            suggested = (required_min + required_max) / 2.0
            explanation = (
                f"{len(missed_ratios)} manually-marked frame(s) were missed by detection and "
                f"{len(false_pos_ratios)} auto-flagged frame(s) were manually rejected. "
                f"A ratio around {suggested:.3f} should satisfy both."
            )
        else:
            suggested = current_ratio
            explanation = (
                "Manual marks/unmarks conflict (no single ratio satisfies both) -- "
                "consider reviewing individual frames rather than changing the global ratio."
            )
    elif required_min is not None:
        suggested = min(required_min, 0.95)
        explanation = (
            f"{len(missed_ratios)} manually-marked frame(s) were missed by detection. "
            f"Raising JITTER_SHRINK_RATIO to about {suggested:.3f} would catch them."
        )
    else:
        suggested = max(required_max, 0.05)
        explanation = (
            f"{len(false_pos_ratios)} auto-flagged frame(s) were manually rejected as false "
            f"positives. Lowering JITTER_SHRINK_RATIO to about {suggested:.3f} would avoid them."
        )

    return suggested, explanation


# =====================================================================
# Rendering
# =====================================================================

def draw_grid_overlay(canvas, iwx_value):
    h, w = canvas.shape[:2]
    color = (70, 70, 70)
    for i in (1, 2):
        x = w * i // 3
        cv2.line(canvas, (x, 0), (x, h), color, 1)
        y = h * i // 3
        cv2.line(canvas, (0, y), (w, y), color, 1)

    text = f"IWX: {iwx_value}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, text, (w - tw - 6, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)


def draw_debug_overlay(canvas, frame_idx, payload, iwx_value):
    lines = [f"Frame: {frame_idx}"]
    if payload is not None and payload["raw"] is not None:
        raw = payload["raw"]
        lines += [
            f"raw CenterX/Y: {raw['CenterX']:.1f}, {raw['CenterY']:.1f}",
            f"raw Major/Minor: {raw['MajorDiameter']:.1f}, {raw['MinorDiameter']:.1f}",
            f"raw Area: {raw['Area']:.1f}  Angle: {raw['Angle']:.1f}",
            f"IWX: {iwx_value}",
            f"AutoCorrected: {payload['auto_flag']}",
            f"Marked: {payload['marked']}",
        ]
    else:
        lines.append("No data")

    overlay = canvas.copy()
    box_h = 14 * len(lines) + 8
    cv2.rectangle(overlay, (0, 18), (170, 18 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = 32
    for line in lines:
        color = CORRECTED_ELLIPSE_COLOR if line.startswith("Marked: True") else (210, 210, 210)
        cv2.putText(canvas, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        y += 14


def draw_mark_button(canvas, marked):
    color = MARK_BTN_ON_COLOR if marked else MARK_BTN_OFF_COLOR
    thickness = -1 if marked else 1
    cv2.rectangle(canvas, (MARK_BTN_X0, MARK_BTN_Y0), (MARK_BTN_X1, MARK_BTN_Y1), color, thickness)
    label = "MARKED" if marked else "MARK"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    text_color = (0, 0, 0) if marked else (200, 200, 200)
    tx = MARK_BTN_X0 + (MARK_BTN_W - tw) // 2
    ty = MARK_BTN_Y0 + (MARK_BTN_H + th) // 2
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)


def render_cell(stream, frame_idx, show_underlay, show_debug, error_handling_on):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)

    if show_underlay:
        vid_frame = stream.get_video_frame(frame_idx)
        if vid_frame is not None:
            canvas = cv2.resize(vid_frame, (CELL_WIDTH, CELL_HEIGHT))

    payload = stream.get_render_payload(frame_idx, error_handling_on)
    scale_x = CELL_WIDTH / stream.video_w
    scale_y = CELL_HEIGHT / stream.video_h

    def draw_ellipse(values, color):
        cx = int(round(values["CenterX"] * scale_x))
        cy = int(round(values["CenterY"] * scale_y))
        maj = max(1, int(round(values["MajorDiameter"] * scale_x / 2.0)))
        minr = max(1, int(round(values["MinorDiameter"] * scale_y / 2.0)))
        cv2.ellipse(canvas, (cx, cy), (maj, minr), values["Angle"], 0, 360, color, 2)

    if payload is not None:
        if payload["raw"] is not None:
            draw_ellipse(payload["raw"], RAW_ELLIPSE_COLOR)
        if payload["show_corrected"]:
            draw_ellipse(payload["corrected"], CORRECTED_ELLIPSE_COLOR)

    draw_grid_overlay(canvas, stream.iwx_value)
    draw_mark_button(canvas, payload["marked"] if payload else False)

    cv2.putText(canvas, stream.label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1, cv2.LINE_AA)

    if show_debug:
        draw_debug_overlay(canvas, frame_idx, payload, stream.iwx_value)

    return canvas


def render_empty_cell(label="No Data"):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, label, (10, CELL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


# =====================================================================
# Timing
# =====================================================================

def compute_tick_wait(all_streams, frame_idx, playback_speed):
    """Real-time pacing: wait the largest inter-frame CaptureTimestampMs
    delta among streams that have data for both frame_idx and frame_idx+1,
    scaled by the current playback speed."""
    deltas = []
    for stream in all_streams:
        row_now = stream.get_row(frame_idx)
        row_next = stream.get_row(frame_idx + 1)
        if row_now is None or row_next is None:
            continue
        t_now = row_now.get("CaptureTimestampMs")
        t_next = row_next.get("CaptureTimestampMs")
        if t_now is None or t_next is None or pd.isna(t_now) or pd.isna(t_next):
            continue
        deltas.append(max(1, t_next - t_now))

    delta = max(deltas) if deltas else DEFAULT_FRAME_DELAY_MS
    return max(1, round(delta / playback_speed))


# =====================================================================
# Main
# =====================================================================

LEFT_ARROW_CODES = {81, 2424832, 63234, 65361}
RIGHT_ARROW_CODES = {83, 2555904, 63235, 65363}


def main():
    print(__doc__)

    parent_dir = select_directory("Select Patient Directory (parent folder containing the 4 trial sub-folders)")
    header_name = os.path.basename(os.path.normpath(parent_dir))
    prefer_corrected = ask_use_corrected_csv()

    subfolders = sorted(
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    )

    if len(subfolders) < 4:
        print(f"[warning] expected 4 sub-folders, found {len(subfolders)}")
    elif len(subfolders) > 4:
        print(f"[warning] found {len(subfolders)} sub-folders, using the first 4")

    subfolders = subfolders[:4]
    while len(subfolders) < 4:
        subfolders.append(None)

    print(f"Patient directory: {parent_dir}")
    print("Trials:")
    for f in subfolders:
        print(f"  - {f}")

    # grid[row][col] -> EyeStream or None ; row 0 = left eye, row 1 = right eye
    grid = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    stream_id = 0
    for col, folder in enumerate(subfolders):
        if folder is None:
            continue
        grid[0][col] = EyeStream(folder, "left", prefer_corrected, stream_id)
        stream_id += 1
        grid[1][col] = EyeStream(folder, "right", prefer_corrected, stream_id)
        stream_id += 1

    all_streams = [s for row in grid for s in row if s is not None]
    if not all_streams:
        print("No valid data found. Exiting.")
        sys.exit(1)

    max_frame = max(s.max_frame for s in all_streams)

    # ---- mutable session state (closed over by the mouse callback) ----
    state = {
        "frame_idx": 0,
        "playing": True,
        "show_underlay": False,
        "show_debug": False,
        "error_handling": DEFAULT_ENABLE_ERROR_HANDLING,
        "speed_idx": 0,
        "suppress_trackbar_cb": False,
    }
    interaction_log = []  # (stream_id, frame_idx, new_marked_state)

    window_name = header_name or "Pupil Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_trackbar(val):
        if state["suppress_trackbar_cb"]:
            return
        state["frame_idx"] = val
        state["playing"] = False

    cv2.createTrackbar("Frame", window_name, 0, max(1, max_frame), on_trackbar)

    def set_frame_idx(new_idx):
        new_idx = max(0, min(new_idx, max_frame))
        state["frame_idx"] = new_idx
        state["suppress_trackbar_cb"] = True
        cv2.setTrackbarPos("Frame", window_name, new_idx)
        state["suppress_trackbar_cb"] = False

    def on_mouse(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = mx // CELL_WIDTH
        row = my // CELL_HEIGHT
        if row not in (0, 1) or col not in range(GRID_COLS):
            return
        stream = grid[row][col]
        if stream is None:
            return
        local_x = mx - col * CELL_WIDTH
        local_y = my - row * CELL_HEIGHT
        if MARK_BTN_X0 <= local_x <= MARK_BTN_X1 and MARK_BTN_Y0 <= local_y <= MARK_BTN_Y1:
            stream.toggle_mark(state["frame_idx"], interaction_log)

    cv2.setMouseCallback(window_name, on_mouse)

    print("\nControls: [space]=play/pause  [left/right]=step  [slider]=scrub  "
          "[u]=underlay  [d]=debug  [e]=error-handling  [/]=speed  "
          "[click MARK]=flag frame  [s]=save+quit  [q/ESC]=quit\n")

    should_save = False

    while True:
        frame_idx = state["frame_idx"]

        canvas_full = np.zeros((CELL_HEIGHT * GRID_ROWS, CELL_WIDTH * GRID_COLS, 3), dtype=np.uint8)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                stream = grid[row][col]
                if stream is not None:
                    cell = render_cell(stream, frame_idx, state["show_underlay"],
                                        state["show_debug"], state["error_handling"])
                else:
                    cell = render_empty_cell()
                y0, y1 = row * CELL_HEIGHT, (row + 1) * CELL_HEIGHT
                x0, x1 = col * CELL_WIDTH, (col + 1) * CELL_WIDTH
                canvas_full[y0:y1, x0:x1] = cell

        speed = PLAYBACK_SPEEDS[state["speed_idx"]]
        status = (
            f"Frame {frame_idx}/{max_frame}  |  {'PLAYING' if state['playing'] else 'PAUSED'}  |  "
            f"Speed:{speed:.0f}x  |  Underlay:{'ON' if state['show_underlay'] else 'off'}  "
            f"Debug:{'ON' if state['show_debug'] else 'off'}  "
            f"ErrorHandling:{'ON' if state['error_handling'] else 'off'}"
        )
        cv2.putText(canvas_full, status, (5, canvas_full.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas_full)

        if state["playing"]:
            wait = compute_tick_wait(all_streams, frame_idx, speed)
        else:
            wait = 30  # idle poll so the UI (mouse/slider/toggles) stays responsive

        raw_key = cv2.waitKeyEx(wait)
        key_ascii = (raw_key & 0xFF) if raw_key != -1 else -1

        if raw_key in LEFT_ARROW_CODES:
            set_frame_idx(frame_idx - 1)
            state["playing"] = False
        elif raw_key in RIGHT_ARROW_CODES:
            set_frame_idx(frame_idx + 1)
            state["playing"] = False
        elif key_ascii == ord('q') or key_ascii == 27:  # ESC
            break
        elif key_ascii == ord('s'):
            should_save = True
            break
        elif key_ascii == ord(' '):
            state["playing"] = not state["playing"]
        elif key_ascii == ord('u'):
            state["show_underlay"] = not state["show_underlay"]
        elif key_ascii == ord('d'):
            state["show_debug"] = not state["show_debug"]
        elif key_ascii == ord('e'):
            state["error_handling"] = not state["error_handling"]
        elif key_ascii == ord('/'):
            state["speed_idx"] = (state["speed_idx"] + 1) % len(PLAYBACK_SPEEDS)
        elif state["playing"]:
            if frame_idx >= max_frame:
                set_frame_idx(max_frame)
                state["playing"] = False
            else:
                set_frame_idx(frame_idx + 1)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    if should_save:
        print("\nSaving corrected CSVs...")
        for s in all_streams:
            out_path = s.save_corrected(state["error_handling"])
            if out_path:
                print(f"  saved: {out_path}")

    suggested, explanation = suggest_jitter_ratio(all_streams, interaction_log, JITTER_SHRINK_RATIO)
    print("\n" + "=" * 60)
    print("JITTER_SHRINK_RATIO suggestion")
    print("=" * 60)
    print(f"Current value: {JITTER_SHRINK_RATIO}")
    if suggested is not None:
        print(f"Suggested value: {suggested:.3f}")
    print(explanation)
    print("=" * 60)

    for s in all_streams:
        if s.cap is not None:
            s.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()