"""
Pupil Visualizer
================

Plays back <eye>_pupil_readings.csv recordings from 4 patient sub-folders
simultaneously, in a 2-row x 4-column grid (row 0 = LEFT eye, row 1 =
RIGHT eye, one column per sub-folder/trial), drawing each frame's fitted
pupil ellipse on top of it.

Playback is synced by real CaptureTimestampMs, not by frame number.

For each trial sub-folder, the LEFT and RIGHT streams are anchored to a
shared zero point: whichever of the two has the smaller starting
CaptureTimestampMs becomes t=0 for that pair. Every frame in both
streams is then given an offset (CaptureTimestampMs - zero point), and
that offset -- not the CSV "Frame" column -- is what determines when a
frame is displayed. If, say, LEFT frame 0 is captured at t=0 and RIGHT
frame 0 is captured 18ms later, RIGHT's cell stays dark for those 18ms
before its first frame appears; the two frames are never assumed to be
simultaneous just because they share a frame number.

All 8 streams (4 trials x 2 eyes) share one session clock, but each
stream looks up its own current frame independently by comparing that
clock to its own offsets. The play/pause, slider, and step controls all
move through the merged timeline of every distinct timestamp across all
streams, so the shared clock always lands exactly on real capture
events. Because the capture hardware could lag, playback is not
expected to be perfectly smooth -- it reproduces the recorded cadence
exactly, gaps and all, rather than smoothing it out.

ELLIPSE COORDINATE SPACE
-------------------------
pupils.py / pupils_batch.py always run crop_to_aspect_ratio(frame, 640,
480) on every video frame -- center-cropping it to a 4:3 box and
resizing that box to exactly 640x480 -- before fitting the pupil
ellipse. So every CenterX/CenterY/MajorDiameter/MinorDiameter in the
readings CSVs lives in that 640x480 "detection frame" space, not in the
raw pixel space of *_eye_cropped.mp4. This tool inverts that same crop
+ resize (using each stream's own video resolution) to place the
ellipse back onto the actual video frame before scaling it into the
on-screen cell -- see compute_detection_to_video_transform().

EXPECTED DIRECTORY LAYOUT
--------------------------
<Patient Parent Directory>/
    <subfolder_1>/
        left_pupil_readings.csv
        right_pupil_readings.csv
        left_eye_cropped.mp4
        right_eye_cropped.mp4
        Patient_<Number>_<Value.Value>.iwxdata
    <subfolder_2>/ ... (same files)
    <subfolder_3>/ ...
    <subfolder_4>/ ...

EXPECTED CSV COLUMNS
---------------------
    Frame, Time_ms, CenterX, CenterY, MajorDiameter, MinorDiameter,
    Area, Angle, FrameNumber, CaptureTimestampMs, ReceiveTimestampMs

CONTROLS
--------
    space        : play / pause
    left arrow   : step to the previous synced timestamp (pauses playback)
    right arrow  : step to the next synced timestamp     (pauses playback)
    slider       : jump to any synced timestamp (works while playing or paused)
    u            : toggle underlay of original *_eye_cropped.mp4 video
    d            : toggle debug overlay (frame #, ellipse values)
    e            : toggle the pupil ellipse on/off
    /            : cycle playback speed through 1x -> 2x -> 3x -> 4x -> 1x
    q / ESC      : quit

All toggles (u/d/e//) work identically whether playback is running or
paused.

COLOR KEY
---------
    green ellipse : the detector's fitted pupil ellipse for this frame,
                     read straight from the readings CSV. Hollow outline,
                     matching pupils.py's own drawing style.

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

# --- Layout -----------------------------------------------------------
GRID_ROWS = 2       # LEFT eye row, RIGHT eye row
GRID_COLS = 4        # one column per sub-folder/trial (always 4)
CELL_WIDTH = 320
CELL_HEIGHT = 240

# --- Playback -----------------------------------------------------------
DEFAULT_FRAME_DELAY_MS = 100  # fallback pacing when no timestamps are available
PLAYBACK_SPEEDS = [1.0, 2.0, 3.0, 4.0]

# --- Colors (BGR) -------------------------------------------------------
RAW_ELLIPSE_COLOR = (55, 255, 0)  # green, matches pupils.py

BASE_CSV_COLUMNS = [
    "Frame", "Time_ms", "CenterX", "CenterY", "MajorDiameter",
    "MinorDiameter", "Area", "Angle", "FrameNumber",
    "CaptureTimestampMs", "ReceiveTimestampMs",
]
ELLIPSE_VALUE_COLUMNS = [
    "CenterX", "CenterY", "MajorDiameter", "MinorDiameter", "Area", "Angle",
]

# pupils.py / pupils_batch.py's crop_to_aspect_ratio() default target --
# every CSV reading's Center/Diameter values are expressed in this space.
DETECTION_FRAME_W = 640
DETECTION_FRAME_H = 480


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
    parts = base.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    return base


def raw_csv_path(folder, eye):
    return os.path.join(folder, f"{eye}_pupil_readings.csv")


def load_csv_for_stream(folder, eye):
    """Load this stream's readings CSV, if present."""
    path = raw_csv_path(folder, eye)
    if not os.path.exists(path):
        print(f"  [warning] no CSV found for {eye} in {folder}")
        return None, None

    missing = [c for c in BASE_CSV_COLUMNS if c not in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path)
    if missing:
        print(f"  [warning] {os.path.basename(path)} missing columns: {missing}")

    df = df.sort_values("Frame").reset_index(drop=True)
    df = df.set_index("Frame", drop=False)
    return df, path


def compute_zero_point(left_df, right_df):
    """The shared t=0 for one trial's LEFT/RIGHT pair: whichever stream's
    earliest CaptureTimestampMs is smaller. If only one side has data,
    that side's own earliest timestamp is used. This is computed once
    per trial folder and applied to both streams in that folder, so a
    frame's on-screen timing reflects real capture time, not frame
    number."""
    earliest = []
    for df in (left_df, right_df):
        if df is None or df.empty:
            continue
        cap_ms = pd.to_numeric(df["CaptureTimestampMs"], errors="coerce").dropna()
        if not cap_ms.empty:
            earliest.append(cap_ms.min())
    return min(earliest) if earliest else 0.0


def compute_detection_to_video_transform(video_w, video_h):
    """Inverts pupils.py/pupils_batch.py's crop_to_aspect_ratio(frame,
    640, 480) step, which every CSV reading's Center/Diameter values are
    expressed in. That function center-crops the raw video frame down to
    a 640:480 (4:3) box, then resizes that box to exactly 640x480 -- so
    CSV coordinates are never in the raw video's own pixel space unless
    the video itself already happens to be 640x480.

    Returns (offset_x, offset_y, scale) such that, for a value expressed
    in the 640x480 detection frame:
        video_px = offset + detection_value * scale
    `scale` is a single number (not separate x/y factors): the crop box
    is forced to exactly 4:3 before the resize, so that resize can never
    stretch x differently than y.
    """
    if video_w <= 0 or video_h <= 0:
        return 0.0, 0.0, 1.0

    desired_ratio = DETECTION_FRAME_W / DETECTION_FRAME_H
    current_ratio = video_w / video_h

    if current_ratio > desired_ratio:
        # Video is wider than 4:3 -- the crop trims left/right.
        crop_w = desired_ratio * video_h
        crop_h = float(video_h)
        offset_x = (video_w - crop_w) / 2.0
        offset_y = 0.0
    else:
        # Video is taller than (or exactly) 4:3 -- the crop trims top/bottom.
        crop_w = float(video_w)
        crop_h = video_w / desired_ratio
        offset_x = 0.0
        offset_y = (video_h - crop_h) / 2.0

    scale = crop_w / DETECTION_FRAME_W  # == crop_h / DETECTION_FRAME_H
    return offset_x, offset_y, scale


# =====================================================================
# EyeStream: everything needed to render one eye of one sub-folder
# =====================================================================

class EyeStream:
    def __init__(self, folder, eye, df, source_path, zero_point, stream_id):
        self.folder = folder
        self.eye = eye  # "left" or "right"
        self.stream_id = stream_id
        self.label = f"{os.path.basename(os.path.normpath(folder))} / {eye.upper()}"

        video_path = os.path.join(folder, f"{eye}_eye_cropped.mp4")

        self.df = df
        self.source_path = source_path
        self.zero_point = zero_point

        if self.df is not None:
            # Per-frame offset (ms) from this trial pair's shared zero point.
            # This -- not the Frame column -- is what drives playback timing.
            cap_ms = pd.to_numeric(self.df["CaptureTimestampMs"], errors="coerce")
            self.df["OffsetMs"] = cap_ms - zero_point
            # Positional arrays (rows already sorted by Frame ascending) for
            # fast "what frame is on screen at time t" lookups.
            self._offsets_arr = self.df["OffsetMs"].to_numpy(dtype=float)
            self._frames_arr = self.df["Frame"].to_numpy()
        else:
            self._offsets_arr = np.array([], dtype=float)
            self._frames_arr = np.array([])

        self.cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
        if self.cap is not None and not self.cap.isOpened():
            print(f"  [warning] could not open video: {video_path}")
            self.cap = None

        self.iwx_value = find_iwxdata_value(folder)

        self.max_frame = int(self.df["Frame"].max()) if self.df is not None else 0

        # Determine this stream's true frame dimensions the SAME way
        # pupils_batch.py's crop_to_aspect_ratio() does -- from an actual
        # decoded frame's .shape -- rather than trusting
        # CAP_PROP_FRAME_WIDTH/HEIGHT. Those metadata properties can
        # disagree with the real decoded pixel dimensions whenever a video
        # carries a rotation flag (common for sideways-mounted eye
        # cameras): cap.get() may report the raw, pre-rotation container
        # size while cap.read() hands back frames already rotated (or the
        # reverse), depending on the OpenCV/FFmpeg build. Since the CSV
        # readings were generated against whatever crop_to_aspect_ratio()
        # saw via .shape, we have to measure the same way or the inverse
        # transform below silently uses the wrong dimensions.
        self.video_w = CELL_WIDTH
        self.video_h = CELL_HEIGHT
        if self.cap is not None:
            reported_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            reported_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, probe_frame = self.cap.read()
            if ret and probe_frame is not None:
                probe_h, probe_w = probe_frame.shape[:2]
                if probe_w > 0 and probe_h > 0:
                    self.video_w, self.video_h = probe_w, probe_h
            # Rewind so playback still starts from frame 0.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            flag = "  <-- MISMATCH vs CAP_PROP" if (reported_w, reported_h) != (self.video_w, self.video_h) else ""
            print(f"  [{self.label}] decoded frame: {self.video_w}x{self.video_h}  "
                  f"(CAP_PROP reported: {reported_w}x{reported_h}){flag}")

        # Maps a CSV reading (640x480 detection-frame space) onto this
        # stream's actual video pixel space -- see
        # compute_detection_to_video_transform() for why this is needed.
        self.det_offset_x, self.det_offset_y, self.det_scale = \
            compute_detection_to_video_transform(self.video_w, self.video_h)

        self.last_video_idx = -1

    # ---- time-based sync: what frame is on screen at session time t ----
    def frame_at_time(self, playback_ms):
        """Return the Frame value that should be displayed at `playback_ms`
        of the shared session clock (the latest frame whose own
        CaptureTimestampMs-based offset has already been reached), or
        None if this stream's own first offset hasn't been reached yet --
        callers should render the cell dark in that case rather than
        showing a frame that wasn't actually captured yet."""
        offsets = self._offsets_arr
        if offsets.size == 0:
            return None
        idx = int(np.searchsorted(offsets, playback_ms, side="right") - 1)
        # Walk back over any frames with a missing/NaN timestamp.
        while idx >= 0 and np.isnan(offsets[idx]):
            idx -= 1
        if idx < 0:
            return None
        return int(self._frames_arr[idx])

    def valid_offsets(self):
        """This stream's timestamp offsets, with missing values dropped."""
        return self._offsets_arr[~np.isnan(self._offsets_arr)]

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

    # ---- resolved ellipse values for rendering / debug text ----
    def get_ellipse_values(self, frame_idx):
        row = self.get_row(frame_idx)
        if row is None:
            return None
        values = {c: row[c] for c in ELLIPSE_VALUE_COLUMNS}
        if any(pd.isna(v) for v in values.values()):
            return None
        return values


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


def draw_debug_overlay(canvas, frame_idx, values, iwx_value):
    lines = [f"Frame: {frame_idx}"]
    if values is not None:
        lines += [
            f"CenterX/Y: {values['CenterX']:.1f}, {values['CenterY']:.1f}",
            f"Major/Minor: {values['MajorDiameter']:.1f}, {values['MinorDiameter']:.1f}",
            f"Area: {values['Area']:.1f}  Angle: {values['Angle']:.1f}",
            f"IWX: {iwx_value}",
        ]
    else:
        lines.append("No data")
        lines.append(f"IWX: {iwx_value}")

    overlay = canvas.copy()
    box_h = 14 * len(lines) + 8
    cv2.rectangle(overlay, (0, 18), (170, 18 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = 32
    for line in lines:
        cv2.putText(canvas, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (210, 210, 210), 1, cv2.LINE_AA)
        y += 14


def render_cell(stream, frame_idx, show_underlay, show_debug, show_ellipse):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)

    # ---- background / underlay ----
    if show_underlay:
        vid_frame = stream.get_video_frame(frame_idx)
        if vid_frame is not None:
            canvas = cv2.resize(vid_frame, (CELL_WIDTH, CELL_HEIGHT))

    values = stream.get_ellipse_values(frame_idx)

    # ---- ellipse: above the underlay/background, below the debug overlay ----
    if show_ellipse and values is not None:
        # CSV reading (640x480 detection-frame space) -> this stream's
        # actual video pixel space.
        video_x = stream.det_offset_x + values["CenterX"] * stream.det_scale
        video_y = stream.det_offset_y + values["CenterY"] * stream.det_scale
        video_major = values["MajorDiameter"] * stream.det_scale
        video_minor = values["MinorDiameter"] * stream.det_scale

        # Video pixel space -> this cell's on-screen canvas space. (These
        # two factors are only unequal if the video's own aspect ratio
        # isn't 4:3, in which case a rotated ellipse's true skewed shape
        # is approximated by scaling its axes independently.)
        canvas_scale_x = CELL_WIDTH / stream.video_w
        canvas_scale_y = CELL_HEIGHT / stream.video_h

        cx = int(round(video_x * canvas_scale_x))
        cy = int(round(video_y * canvas_scale_y))
        maj = max(1, int(round(video_major * canvas_scale_x / 2.0)))
        minr = max(1, int(round(video_minor * canvas_scale_y / 2.0)))
        cv2.ellipse(canvas, (cx, cy), (maj, minr), values["Angle"], 0, 360,
                    RAW_ELLIPSE_COLOR, 2)

    draw_grid_overlay(canvas, stream.iwx_value)

    cv2.putText(canvas, stream.label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1, cv2.LINE_AA)

    # ---- debug overlay: always on top ----
    if show_debug:
        draw_debug_overlay(canvas, frame_idx, values, stream.iwx_value)

    return canvas


def render_empty_cell(label="No Data"):
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, label, (10, CELL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def render_dark_cell(stream):
    """This stream has real data, but the session clock hasn't reached
    this stream's first CaptureTimestampMs-based offset yet -- stay
    fully dark rather than showing a frame that wasn't captured yet."""
    canvas = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, stream.label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (70, 70, 70), 1, cv2.LINE_AA)
    return canvas


# =====================================================================
# Timing
# =====================================================================

def compute_tick_wait(master_events, event_idx, playback_speed):
    """Real-time pacing: wait exactly the CaptureTimestampMs gap between
    the current merged-timeline event and the next one, scaled by the
    current playback speed. Every entry in `master_events` is a real,
    distinct capture timestamp (relative to its trial's zero point) from
    some stream, so this reproduces the actual recorded cadence -- lag
    and all -- instead of a fixed or averaged frame delay."""
    if event_idx >= len(master_events) - 1:
        return DEFAULT_FRAME_DELAY_MS
    delta = master_events[event_idx + 1] - master_events[event_idx]
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
        left_df, left_path = load_csv_for_stream(folder, "left")
        right_df, right_path = load_csv_for_stream(folder, "right")
        # Zero point is shared by both eyes of this trial: whichever side
        # started capturing first becomes t=0 for the pair.
        zero_point = compute_zero_point(left_df, right_df)

        grid[0][col] = EyeStream(folder, "left", left_df, left_path, zero_point, stream_id)
        stream_id += 1
        grid[1][col] = EyeStream(folder, "right", right_df, right_path, zero_point, stream_id)
        stream_id += 1

    all_streams = [s for row in grid for s in row if s is not None]
    if not all_streams:
        print("No valid data found. Exiting.")
        sys.exit(1)

    # Merged real-time schedule: every distinct CaptureTimestampMs-based
    # offset across every stream, sorted. Playback steps through this
    # timeline (not through frame numbers), so the shared session clock
    # always lands exactly on a real capture event from some stream.
    event_set = set()
    for s in all_streams:
        event_set.update(float(x) for x in s.valid_offsets())
    if not event_set:
        print("No CaptureTimestampMs data found across any stream. Exiting.")
        sys.exit(1)
    master_events = np.array(sorted(event_set))
    last_event_idx = len(master_events) - 1

    # ---- mutable session state ----
    state = {
        "event_idx": 0,
        "playing": True,
        "show_underlay": False,
        "show_debug": False,
        "show_ellipse": True,
        "speed_idx": 0,
        "suppress_trackbar_cb": False,
    }

    window_name = header_name or "Pupil Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_trackbar(val):
        if state["suppress_trackbar_cb"]:
            return
        state["event_idx"] = val
        state["playing"] = False

    cv2.createTrackbar("Event", window_name, 0, max(1, last_event_idx), on_trackbar)

    def set_event_idx(new_idx):
        new_idx = max(0, min(new_idx, last_event_idx))
        state["event_idx"] = new_idx
        state["suppress_trackbar_cb"] = True
        cv2.setTrackbarPos("Event", window_name, new_idx)
        state["suppress_trackbar_cb"] = False

    print("\nControls: [space]=play/pause  [left/right]=step  [slider]=scrub  "
          "[u]=underlay  [d]=debug  [e]=ellipse  [/]=speed  [q/ESC]=quit\n")

    while True:
        event_idx = state["event_idx"]
        playback_ms = master_events[event_idx]

        canvas_full = np.zeros((CELL_HEIGHT * GRID_ROWS, CELL_WIDTH * GRID_COLS, 3), dtype=np.uint8)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                stream = grid[row][col]
                if stream is None or stream.df is None:
                    cell = render_empty_cell()
                else:
                    frame_idx = stream.frame_at_time(playback_ms)
                    if frame_idx is None:
                        # This stream's own first CaptureTimestampMs offset
                        # hasn't been reached yet -- stay dark rather than
                        # showing a frame out of sync with real capture time.
                        cell = render_dark_cell(stream)
                    else:
                        cell = render_cell(stream, frame_idx, state["show_underlay"],
                                            state["show_debug"], state["show_ellipse"])
                y0, y1 = row * CELL_HEIGHT, (row + 1) * CELL_HEIGHT
                x0, x1 = col * CELL_WIDTH, (col + 1) * CELL_WIDTH
                canvas_full[y0:y1, x0:x1] = cell

        speed = PLAYBACK_SPEEDS[state["speed_idx"]]
        status = (
            f"t={playback_ms:.0f}ms  Event {event_idx}/{last_event_idx}  |  "
            f"{'PLAYING' if state['playing'] else 'PAUSED'}  |  "
            f"Speed:{speed:.0f}x  |  Underlay:{'ON' if state['show_underlay'] else 'off'}  "
            f"Debug:{'ON' if state['show_debug'] else 'off'}  "
            f"Ellipse:{'ON' if state['show_ellipse'] else 'off'}"
        )
        cv2.putText(canvas_full, status, (5, canvas_full.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas_full)

        if state["playing"]:
            wait = compute_tick_wait(master_events, event_idx, speed)
        else:
            wait = 30  # idle poll so the UI (slider/toggles) stays responsive

        raw_key = cv2.waitKeyEx(wait)
        key_ascii = (raw_key & 0xFF) if raw_key != -1 else -1

        if raw_key in LEFT_ARROW_CODES:
            set_event_idx(event_idx - 1)
            state["playing"] = False
        elif raw_key in RIGHT_ARROW_CODES:
            set_event_idx(event_idx + 1)
            state["playing"] = False
        elif key_ascii == ord('q') or key_ascii == 27:  # ESC
            break
        elif key_ascii == ord(' '):
            state["playing"] = not state["playing"]
        elif key_ascii == ord('u'):
            state["show_underlay"] = not state["show_underlay"]
        elif key_ascii == ord('d'):
            state["show_debug"] = not state["show_debug"]
        elif key_ascii == ord('e'):
            state["show_ellipse"] = not state["show_ellipse"]
        elif key_ascii == ord('/'):
            state["speed_idx"] = (state["speed_idx"] + 1) % len(PLAYBACK_SPEEDS)
        elif state["playing"]:
            if event_idx >= last_event_idx:
                set_event_idx(last_event_idx)
                state["playing"] = False
            else:
                set_event_idx(event_idx + 1)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    for s in all_streams:
        if s.cap is not None:
            s.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()