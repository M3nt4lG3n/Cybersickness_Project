"""
Pupil Visualizer
================

Plays back left_pupil.csv / right_pupil.csv recordings from 4 patient
sub-folders simultaneously, synced by frame number, in a 2-row x 4-column
grid (row 0 = LEFT eye, row 1 = RIGHT eye, one column per sub-folder/trial).

EXPECTED DIRECTORY LAYOUT
--------------------------
<Patient Parent Directory>/
    <subfolder_1>/
        left_pupil.csv
        right_pupil.csv
        left_eye_cropped.mp4
        right_eye_cropped.mp4
        Patient_<Number>_<Value.Value>.iwxdata
    <subfolder_2>/
        ... (same 5 files)
    <subfolder_3>/
        ...
    <subfolder_4>/
        ...

CONTROLS
--------
    space       : play / pause
    left arrow  : step one frame backward (pauses playback)
    right arrow : step one frame forward  (pauses playback)
    u           : toggle underlay of original *_eye_cropped.mp4 video
    d           : toggle debug overlay (frame #, raw csv values, iwxdata value)
    q / ESC     : quit

REQUIREMENTS
------------
    pip install opencv-python pandas numpy
    (tkinter ships with most Python installs; on some Linux distros you may
     need `sudo apt-get install python3-tk`)
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

# --- Jitter / error-handling toggle -------------------------------------
# When True, sudden implausible drops in pupil diameter (acquisition
# jitter) are detected and the previous good reading is substituted.
# Frames where this substitution happens are rendered in GREEN instead
# of WHITE so you can visually confirm the correction is firing.
ENABLE_ERROR_HANDLING = True

# If the new frame's average diameter falls below this fraction of the
# previous accepted diameter, it is treated as a jitter/dropout frame.
JITTER_SHRINK_RATIO = 0.55

# --- Layout ---------------------------------------------------------------
GRID_ROWS = 2      # LEFT eye row, RIGHT eye row
GRID_COLS = 4       # one column per sub-folder/trial (always 4)
CELL_WIDTH = 320
CELL_HEIGHT = 240

# --- Playback ---------------------------------------------------------------
DEFAULT_FRAME_DELAY_MS = 100  # fallback playback speed (~10 fps)

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


def find_iwxdata_value(folder):
    """Locate the *.iwxdata file in `folder` and extract the Value.Value
    segment from a filename formatted as Patient_Number_Value.Value"""
    matches = glob.glob(os.path.join(folder, "*.iwxdata"))
    if not matches:
        return "N/A"

    base = os.path.splitext(os.path.basename(matches[0]))[0]
    # Split into at most 3 parts: "Patient", "Number", "Value.Value"
    # (maxsplit=2 keeps the value segment intact even though it contains a dot)
    parts = base.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    return base


def load_csv(path):
    if not os.path.exists(path):
        print(f"  [warning] missing CSV: {path}")
        return None
    df = pd.read_csv(path)
    df = df.set_index("Frame", drop=False)
    return df


# =====================================================================
# EyeStream: everything needed to render one eye of one sub-folder
# =====================================================================

class EyeStream:
    def __init__(self, folder, eye):
        self.folder = folder
        self.eye = eye  # "left" or "right"
        self.label = f"{os.path.basename(os.path.normpath(folder))} / {eye.upper()}"

        csv_path = os.path.join(folder, f"{eye}_pupil.csv")
        video_path = os.path.join(folder, f"{eye}_eye_cropped.mp4")

        self.df = load_csv(csv_path)
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

        # jitter-correction state
        self.last_good = None
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
        row = self.df.loc[frame_idx]
        # .loc on a duplicate-free unique index returns a Series
        return row

    # ---- jitter-aware value resolution ----
    def get_render_data(self, frame_idx):
        row = self.get_row(frame_idx)
        if row is None:
            return None, False

        values = {
            "Time_ms": float(row["Time_ms"]),
            "CenterX": float(row["CenterX"]),
            "CenterY": float(row["CenterY"]),
            "MajorDiameter": float(row["MajorDiameter"]),
            "MinorDiameter": float(row["MinorDiameter"]),
            "Area": float(row["Area"]),
            "Angle": float(row["Angle"]),
        }

        # Reject rows containing NaNs
        if any(np.isnan(v) for k, v in values.items() if k != "Time_ms"):
            if self.last_good is not None:
                return dict(self.last_good), True
            return None, False

        corrected = False
        if ENABLE_ERROR_HANDLING and self.last_good is not None:
            prev_diam = (self.last_good["MajorDiameter"] + self.last_good["MinorDiameter"]) / 2.0
            new_diam = (values["MajorDiameter"] + values["MinorDiameter"]) / 2.0
            if prev_diam > 0 and new_diam < prev_diam * JITTER_SHRINK_RATIO:
                # Sudden implausible shrink -> treat as acquisition jitter,
                # re-use the previous good reading instead.
                values = dict(self.last_good)
                corrected = True

        if not corrected:
            self.last_good = values

        return values, corrected


# =====================================================================
# Rendering
# =====================================================================

def draw_grid_overlay(canvas, iwx_value):
    """Reference grid (rule-of-thirds) plus the Value.Value label pulled
    from the sub-folder's .iwxdata filename."""
    h, w = canvas.shape[:2]
    color = (70, 70, 70)
    for i in (1, 2):
        x = w * i // 3
        cv2.line(canvas, (x, 0), (x, h), color, 1)
        y = h * i // 3
        cv2.line(canvas, (0, y), (w, y), color, 1)

    text = f"IWX: {iwx_value}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, text, (w - tw - 6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)


def draw_debug_overlay(canvas, frame_idx, values, iwx_value, corrected):
    lines = [f"Frame: {frame_idx}"]
    if values is not None:
        lines += [
            f"Time_ms: {values['Time_ms']:.1f}",
            f"CenterX: {values['CenterX']:.2f}",
            f"CenterY: {values['CenterY']:.2f}",
            f"MajorD: {values['MajorDiameter']:.2f}",
            f"MinorD: {values['MinorDiameter']:.2f}",
            f"Area: {values['Area']:.2f}",
            f"Angle: {values['Angle']:.2f}",
            f"IWX: {iwx_value}",
            f"Corrected: {corrected}",
        ]
    else:
        lines.append("No data")

    # semi-transparent background for readability
    overlay = canvas.copy()
    box_h = 14 * len(lines) + 8
    cv2.rectangle(overlay, (0, 18), (150, 18 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = 32
    for line in lines:
        color = (0, 255, 0) if (corrected and line.startswith("Corrected")) else (210, 210, 210)
        cv2.putText(canvas, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        y += 14


def render_cell(stream, frame_idx, show_underlay, show_debug):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)

    if show_underlay:
        vid_frame = stream.get_video_frame(frame_idx)
        if vid_frame is not None:
            canvas = cv2.resize(vid_frame, (CELL_WIDTH, CELL_HEIGHT))

    values, corrected = stream.get_render_data(frame_idx)

    if values is not None:
        scale_x = CELL_WIDTH / stream.video_w
        scale_y = CELL_HEIGHT / stream.video_h
        cx = int(round(values["CenterX"] * scale_x))
        cy = int(round(values["CenterY"] * scale_y))
        maj = max(1, int(round(values["MajorDiameter"] * scale_x / 2.0)))
        minr = max(1, int(round(values["MinorDiameter"] * scale_y / 2.0)))

        color = (0, 255, 0) if corrected else (255, 255, 255)
        thickness = 2 if show_underlay else -1  # outline over video, filled on black
        cv2.ellipse(canvas, (cx, cy), (maj, minr), values["Angle"], 0, 360, color, thickness)

    draw_grid_overlay(canvas, stream.iwx_value)

    cv2.putText(canvas, stream.label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1, cv2.LINE_AA)

    if show_debug:
        draw_debug_overlay(canvas, frame_idx, values, stream.iwx_value, corrected)

    return canvas


def render_empty_cell(label="No Data"):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, label, (10, CELL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


# =====================================================================
# Main
# =====================================================================

LEFT_ARROW_CODES = {81, 2424832, 63234, 65361}
RIGHT_ARROW_CODES = {83, 2555904, 63235, 65363}


def main():
    parent_dir = select_directory("Select Patient Directory (parent folder containing the 4 trial sub-folders)")
    header_name = os.path.basename(os.path.normpath(parent_dir))

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
    for col, folder in enumerate(subfolders):
        if folder is None:
            continue
        grid[0][col] = EyeStream(folder, "left")
        grid[1][col] = EyeStream(folder, "right")

    all_streams = [s for row in grid for s in row if s is not None]
    if not all_streams:
        print("No valid data found. Exiting.")
        sys.exit(1)

    max_frame = max(s.max_frame for s in all_streams)

    frame_idx = 0
    playing = True
    show_underlay = False
    show_debug = False

    window_name = header_name or "Pupil Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nControls: [space]=play/pause  [left/right]=step  [u]=underlay  [d]=debug  [q/ESC]=quit\n")

    while True:
        canvas_full = np.zeros((CELL_HEIGHT * GRID_ROWS, CELL_WIDTH * GRID_COLS, 3), dtype=np.uint8)

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                stream = grid[row][col]
                if stream is not None:
                    cell = render_cell(stream, frame_idx, show_underlay, show_debug)
                else:
                    cell = render_empty_cell()
                y0, y1 = row * CELL_HEIGHT, (row + 1) * CELL_HEIGHT
                x0, x1 = col * CELL_WIDTH, (col + 1) * CELL_WIDTH
                canvas_full[y0:y1, x0:x1] = cell

        # small status bar
        status = f"Frame {frame_idx}/{max_frame}  |  {'PLAYING' if playing else 'PAUSED'}  |  " \
                  f"Underlay:{'ON' if show_underlay else 'off'}  Debug:{'ON' if show_debug else 'off'}  " \
                  f"ErrorHandling:{'ON' if ENABLE_ERROR_HANDLING else 'off'}"
        cv2.putText(canvas_full, status, (5, canvas_full.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas_full)

        delay = DEFAULT_FRAME_DELAY_MS if playing else 0
        raw_key = cv2.waitKeyEx(delay)

        if raw_key == -1:
            key_ascii = -1
        else:
            key_ascii = raw_key & 0xFF

        if raw_key in LEFT_ARROW_CODES:
            frame_idx = max(0, frame_idx - 1)
            playing = False
        elif raw_key in RIGHT_ARROW_CODES:
            frame_idx = min(max_frame, frame_idx + 1)
            playing = False
        elif key_ascii == ord('q') or key_ascii == 27:  # ESC
            break
        elif key_ascii == ord(' '):
            playing = not playing
        elif key_ascii == ord('u'):
            show_underlay = not show_underlay
        elif key_ascii == ord('d'):
            show_debug = not show_debug
        elif playing:
            frame_idx += 1
            if frame_idx > max_frame:
                frame_idx = max_frame
                playing = False

        # window closed via the OS 'x' button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    for s in all_streams:
        if s.cap is not None:
            s.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()