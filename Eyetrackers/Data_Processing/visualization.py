"""
visualization.py

Video rendering utilities for the LabScribe
cybersickness analysis pipeline.

Responsibilities:

    • Render balance board visualization
    • Display force sensor states
    • Display center of pressure
    • Render COP trail
    • Display ECG scrolling monitor
    • Export synchronized analysis video


This module does not:

    • Calculate ECG features
    • Calculate COP metrics
    • Modify raw analysis data


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass

from collections import deque

import numpy as np
import pandas as pd

import cv2



# ============================================================
# Configuration
# ============================================================

@dataclass(slots=True)
class VisualizationConfig:
    """
    Rendering configuration.
    """

    width: int = 1280

    height: int = 720


    fps: int = 30


    board_width: int = 450

    board_height: int = 450


    ecg_height: int = 200


    cop_trail_length: int = 300


    sensor_radius: int = 35


    output_codec: str = "mp4v"



# ============================================================
# Board Geometry
# ============================================================

@dataclass(slots=True)
class BoardGeometry:
    """
    Cached balance board geometry.

    Calculated once before rendering.
    """

    top_left: tuple[int, int]

    top_right: tuple[int, int]

    bottom_left: tuple[int, int]

    bottom_right: tuple[int, int]

    center: tuple[int, int]



# ============================================================
# ECG Window State
# ============================================================

@dataclass(slots=True)
class ECGWindow:
    """
    State for scrolling ECG display.
    """

    samples: deque

    timestamps: deque

    window_seconds: float = 5.0

    newest_position: float = 0.85



# ============================================================
# Default Constructors
# ============================================================

def create_ecg_window(
    max_samples: int = 5000,
) -> ECGWindow:
    """
    Create scrolling ECG buffer.
    """

    return ECGWindow(

        samples=deque(
            maxlen=max_samples
        ),

        timestamps=deque(
            maxlen=max_samples
        ),
    )

# ============================================================
# Board Geometry Creation
# ============================================================

def create_board_geometry(
    config: VisualizationConfig,
) -> BoardGeometry:
    """
    Create balance board drawing coordinates.

    This is calculated once before rendering.
    """

    center_x = (
        config.width // 4
    )


    center_y = (
        config.height // 2
    )


    half_width = (
        config.board_width // 2
    )


    half_height = (
        config.board_height // 2
    )


    offset_x = (
        half_width // 2
    )


    offset_y = (
        half_height // 2
    )


    return BoardGeometry(

        top_left=(
            center_x - offset_x,
            center_y - offset_y,
        ),

        top_right=(
            center_x + offset_x,
            center_y - offset_y,
        ),

        bottom_left=(
            center_x - offset_x,
            center_y + offset_y,
        ),

        bottom_right=(
            center_x + offset_x,
            center_y + offset_y,
        ),

        center=(
            center_x,
            center_y,
        ),
    )



# ============================================================
# Force Color Mapping
# ============================================================

def interpolate_color(
    value: float,
) -> tuple[int, int, int]:
    """
    Convert normalized force to color.

    Scale:

        0.00  Blue

        0.25  Green

        0.50  Yellow

        0.75  Orange

        1.00  Red


    Returns OpenCV BGR color.
    """

    value = np.clip(
        value,
        0,
        1,
    )


    color_stops = [

        (
            0.0,
            (255, 0, 0),
        ),

        (
            0.25,
            (0, 255, 0),
        ),

        (
            0.50,
            (0, 255, 255),
        ),

        (
            0.75,
            (0, 165, 255),
        ),

        (
            1.0,
            (0, 0, 255),
        ),

    ]


    for index in range(
        len(color_stops) - 1
    ):

        lower_value, lower_color = (
            color_stops[index]
        )

        upper_value, upper_color = (
            color_stops[index + 1]
        )


        if value <= upper_value:

            ratio = (
                value - lower_value
            ) / (
                upper_value -
                lower_value
            )


            b = int(
                lower_color[0]
                +
                ratio *
                (
                    upper_color[0]
                    -
                    lower_color[0]
                )
            )


            g = int(
                lower_color[1]
                +
                ratio *
                (
                    upper_color[1]
                    -
                    lower_color[1]
                )
            )


            r = int(
                lower_color[2]
                +
                ratio *
                (
                    upper_color[2]
                    -
                    lower_color[2]
                )
            )


            return (
                b,
                g,
                r,
            )


    return color_stops[-1][1]



# ============================================================
# Force Normalization
# ============================================================

def normalize_force(
    force: float,
    maximum_force: float,
) -> float:
    """
    Normalize sensor force for display.
    """

    if maximum_force <= 0:

        return 0.0


    return float(
        np.clip(
            force / maximum_force,
            0,
            1,
        )
    )



# ============================================================
# Sensor Drawing
# ============================================================

def draw_force_sensor(
    frame: np.ndarray,
    position: tuple[int, int],
    force: float,
    maximum_force: float,
    radius: int,
) -> None:
    """
    Draw one force sensor.

    Color represents normalized force.
    """

    normalized = normalize_force(
        force,
        maximum_force,
    )


    color = interpolate_color(
        normalized
    )


    cv2.circle(

        frame,

        position,

        radius,

        color,

        thickness=-1,
    )


    cv2.circle(

        frame,

        position,

        radius,

        (255, 255, 255),

        thickness=2,
    )


    cv2.putText(

        frame,

        f"{force:.1f}",

        (
            position[0] - 25,
            position[1] + 5,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1,

        cv2.LINE_AA,
    )

# ============================================================
# COP Trail State
# ============================================================

@dataclass(slots=True)
class COPTrail:
    """
    Stores recent COP positions for rendering.
    """

    points: deque



def create_cop_trail(
    length: int = 300,
) -> COPTrail:
    """
    Create COP trail buffer.
    """

    return COPTrail(

        points=deque(
            maxlen=length
        )
    )



# ============================================================
# COP Coordinate Conversion
# ============================================================

def cop_to_pixel(
    cop_x: float,
    cop_y: float,
    geometry: BoardGeometry,
    config: VisualizationConfig,
) -> tuple[int, int]:
    """
    Convert normalized COP coordinates into
    board pixel coordinates.

    COP coordinate assumptions:

        X:
            -1 = left
             0 = center
            +1 = right


        Y:
            -1 = back
             0 = center
            +1 = front
    """

    board_width = (
        config.board_width
        /
        2
    )


    board_height = (
        config.board_height
        /
        2
    )


    pixel_x = (
        geometry.center[0]
        +
        cop_x * board_width
    )


    pixel_y = (
        geometry.center[1]
        -
        cop_y * board_height
    )


    return (

        int(pixel_x),

        int(pixel_y),
    )



# ============================================================
# COP Trail Update
# ============================================================

def update_cop_trail(
    trail: COPTrail,
    cop_x: float,
    cop_y: float,
    geometry: BoardGeometry,
    config: VisualizationConfig,
) -> None:
    """
    Add a new COP point.

    Uses smoothed COP values.
    """

    point = cop_to_pixel(
        cop_x,
        cop_y,
        geometry,
        config,
    )


    trail.points.append(
        point
    )



# ============================================================
# COP Drawing
# ============================================================

def draw_cop_trail(
    frame: np.ndarray,
    trail: COPTrail,
) -> None:
    """
    Draw COP movement trail.
    """

    points = list(
        trail.points
    )


    if len(points) < 2:

        return



    for index in range(
        1,
        len(points),
    ):

        cv2.line(

            frame,

            points[index - 1],

            points[index],

            (255, 255, 255),

            2,

            cv2.LINE_AA,
        )



def draw_cop_position(
    frame: np.ndarray,
    point: tuple[int, int],
) -> None:
    """
    Draw current COP location.
    """

    cv2.circle(

        frame,

        point,

        10,

        (255, 255, 255),

        thickness=-1,
    )


    cv2.circle(

        frame,

        point,

        10,

        (0, 0, 0),

        thickness=2,
    )



# ============================================================
# Balance Board Rendering
# ============================================================

def draw_balance_board(
    frame: np.ndarray,
    row: pd.Series,
    geometry: BoardGeometry,
    trail: COPTrail,
    config: VisualizationConfig,
    maximum_force: float,
) -> None:
    """
    Render complete balance board.

    Uses:

        TL
        TR
        BL
        BR

        COP_X_Smooth
        COP_Y_Smooth
    """

    # --------------------------------------------------------
    # Board outline
    # --------------------------------------------------------

    cv2.rectangle(

        frame,

        (
            geometry.top_left[0] - 40,
            geometry.top_left[1] - 40,
        ),

        (
            geometry.bottom_right[0] + 40,
            geometry.bottom_right[1] + 40,
        ),

        (255, 255, 255),

        2,
    )


    # --------------------------------------------------------
    # Force sensors
    # --------------------------------------------------------

    draw_force_sensor(

        frame,

        geometry.top_left,

        row["TL"],

        maximum_force,

        config.sensor_radius,
    )


    draw_force_sensor(

        frame,

        geometry.top_right,

        row["TR"],

        maximum_force,

        config.sensor_radius,
    )


    draw_force_sensor(

        frame,

        geometry.bottom_left,

        row["BL"],

        maximum_force,

        config.sensor_radius,
    )


    draw_force_sensor(

        frame,

        geometry.bottom_right,

        row["BR"],

        maximum_force,

        config.sensor_radius,
    )


    # --------------------------------------------------------
    # COP
    # --------------------------------------------------------

    if (
        "COP_X_Smooth" in row
        and
        "COP_Y_Smooth" in row
    ):

        cop_x = row["COP_X_Smooth"]

        cop_y = row["COP_Y_Smooth"]


        update_cop_trail(
            trail,
            cop_x,
            cop_y,
            geometry,
            config,
        )


        point = cop_to_pixel(
            cop_x,
            cop_y,
            geometry,
            config,
        )


        draw_cop_trail(
            frame,
            trail,
        )


        draw_cop_position(
            frame,
            point,
        )

# ============================================================
# ECG Display Scaling
# ============================================================

@dataclass(slots=True)
class ECGScale:
    """
    Cached ECG visualization bounds.
    """

    baseline: float

    ymin: float

    ymax: float



def calculate_ecg_scale(
    signal: pd.Series | np.ndarray,
) -> ECGScale:
    """
    Calculate robust ECG display scaling.

    Prevents large artifacts from compressing
    the waveform.
    """

    values = np.asarray(
        signal,
        dtype=float,
    )


    baseline = np.median(
        values
    )


    amplitude = np.max(
        np.abs(
            values - baseline
        )
    )


    if amplitude == 0:

        amplitude = 1.0



    ymin = (
        baseline
        -
        amplitude * 1.2
    )


    ymax = (
        baseline
        +
        amplitude * 1.2
    )


    return ECGScale(

        baseline=baseline,

        ymin=ymin,

        ymax=ymax,
    )



# ============================================================
# ECG Grid
# ============================================================

def draw_ecg_grid(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """
    Draw ECG monitor grid.
    """

    vertical_spacing = 40

    horizontal_spacing = 25


    for offset in range(
        0,
        width,
        vertical_spacing,
    ):

        cv2.line(

            frame,

            (
                x + offset,
                y,
            ),

            (
                x + offset,
                y + height,
            ),

            (80, 80, 80),

            1,
        )


    for offset in range(
        0,
        height,
        horizontal_spacing,
    ):

        cv2.line(

            frame,

            (
                x,
                y + offset,
            ),

            (
                x + width,
                y + offset,
            ),

            (80, 80, 80),

            1,
        )



# ============================================================
# ECG Coordinate Conversion
# ============================================================

def ecg_to_pixel(
    value: float,
    index: int,
    scale: ECGScale,
    width: int,
    height: int,
    newest_position: float,
    total_samples: int,
) -> tuple[int, int]:
    """
    Convert ECG sample to screen coordinates.

    Newest sample appears at approximately
    85% of the monitor width.
    """

    newest_x = (
        width *
        newest_position
    )


    pixels_per_sample = (
        newest_x /
        max(
            total_samples,
            1,
        )
    )


    x = (
        newest_x
        -
        (
            total_samples -
            index
        )
        *
        pixels_per_sample
    )


    normalized = (
        value - scale.ymin
    ) / (
        scale.ymax -
        scale.ymin
    )


    y = (
        height
        -
        normalized *
        height
    )


    return (

        int(x),

        int(y),
    )



# ============================================================
# ECG Waveform Rendering
# ============================================================

def draw_ecg_waveform(
    frame: np.ndarray,
    ecg_values: list[float],
    scale: ECGScale,
    x: int,
    y: int,
    width: int,
    height: int,
    newest_position: float = 0.85,
) -> None:
    """
    Draw scrolling ECG waveform.
    """

    if len(ecg_values) < 2:

        return


    points = []


    for index, value in enumerate(
        ecg_values
    ):

        px, py = ecg_to_pixel(

            value,

            index,

            scale,

            width,

            height,

            newest_position,

            len(ecg_values),
        )


        points.append(
            (
                x + px,
                y + py,
            )
        )


    for index in range(
        1,
        len(points),
    ):

        cv2.line(

            frame,

            points[index - 1],

            points[index],

            (0, 255, 0),

            2,

            cv2.LINE_AA,
        )



# ============================================================
# Fiducial Marker Rendering
# ============================================================

def draw_ecg_markers(
    frame: np.ndarray,
    markers: dict[str, list[int]],
    x: int,
    y: int,
    width: int,
    height: int,
    scale: ECGScale,
    samples: list[float],
) -> None:
    """
    Draw P/Q/R/S/T locations.

    Markers are optional and missing
    fiducials are ignored.
    """

    marker_colors = {

        "P": (255, 0, 255),

        "Q": (255, 255, 0),

        "R": (0, 0, 255),

        "S": (255, 165, 0),

        "T": (0, 255, 255),

    }


    total_samples = len(samples)


    for name, indices in markers.items():

        if name not in marker_colors:

            continue


        for index in indices:

            if index >= total_samples:

                continue


            px, py = ecg_to_pixel(

                samples[index],

                index,

                scale,

                width,

                height,

                0.85,

                total_samples,
            )


            cv2.circle(

                frame,

                (
                    x + px,
                    y + py,
                ),

                5,

                marker_colors[name],

                -1,
            )



# ============================================================
# ECG Window Rendering
# ============================================================

def draw_ecg_window(
    frame: np.ndarray,
    ecg_values: list[float],
    scale: ECGScale,
    heart_rate: float | None,
    markers=None,
    config: VisualizationConfig | None = None,
) -> None:
    """
    Render complete ECG monitor.
    """

    if config is None:

        config = VisualizationConfig()



    x = (
        config.width
        -
        550
    )


    y = 30


    width = 500

    height = (
        config.ecg_height
    )


    draw_ecg_grid(
        frame,
        x,
        y,
        width,
        height,
    )


    draw_ecg_waveform(
        frame,
        ecg_values,
        scale,
        x,
        y,
        width,
        height,
    )


    if markers:

        draw_ecg_markers(
            frame,
            markers,
            x,
            y,
            width,
            height,
            scale,
            ecg_values,
        )


    if heart_rate is not None:

        cv2.putText(

            frame,

            f"HR: {heart_rate:.1f} BPM",

            (
                x,
                y - 10,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2,
        )

# ============================================================
# Timestamp Utilities
# ============================================================

def find_closest_index(
    timestamps: np.ndarray,
    target: int | float,
) -> int:
    """
    Find closest timestamp index.

    Used to synchronize:

        ECG

        balance

        events (future)
    """

    index = np.searchsorted(
        timestamps,
        target,
    )


    if index <= 0:

        return 0


    if index >= len(timestamps):

        return len(timestamps) - 1


    before = timestamps[index - 1]

    after = timestamps[index]


    if abs(target - before) < abs(after - target):

        return index - 1


    return index



# ============================================================
# Frame Rendering
# ============================================================

def create_blank_frame(
    config: VisualizationConfig,
) -> np.ndarray:
    """
    Create empty video frame.
    """

    return np.zeros(
        (
            config.height,
            config.width,
            3,
        ),
        dtype=np.uint8,
    )



def draw_timestamp(
    frame: np.ndarray,
    timestamp_ms: int,
) -> None:
    """
    Draw current timestamp.
    """

    seconds = (
        timestamp_ms /
        1000.0
    )


    cv2.putText(

        frame,

        f"Time: {seconds:.2f}s",

        (
            30,
            40,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2,
    )



def render_frame(
    balance_row: pd.Series,
    ecg_values: list[float],
    heart_rate: float | None,
    geometry: BoardGeometry,
    trail: COPTrail,
    ecg_scale: ECGScale,
    maximum_force: float,
    config: VisualizationConfig,
    markers=None,
) -> np.ndarray:
    """
    Render one complete analysis frame.
    """

    frame = create_blank_frame(
        config
    )


    draw_balance_board(

        frame,

        balance_row,

        geometry,

        trail,

        config,

        maximum_force,
    )


    draw_ecg_window(

        frame,

        ecg_values,

        ecg_scale,

        heart_rate,

        markers,

        config,
    )


    if "UnixTime_ms" in balance_row:

        draw_timestamp(

            frame,

            int(
                balance_row["UnixTime_ms"]
            ),
        )


    return frame



# ============================================================
# Video Writer
# ============================================================

def create_video_writer(
    output_file: str,
    config: VisualizationConfig,
) -> cv2.VideoWriter:
    """
    Create MP4 writer.
    """

    fourcc = cv2.VideoWriter_fourcc(
        *config.output_codec
    )


    return cv2.VideoWriter(

        output_file,

        fourcc,

        config.fps,

        (
            config.width,
            config.height,
        ),
    )



# ============================================================
# Balance Video Rendering
# ============================================================

def render_balance_video(
    balance_df: pd.DataFrame,
    ecg_df: pd.DataFrame,
    output_file: str,
    config: VisualizationConfig | None = None,
    markers=None,
) -> None:
    """
    Render synchronized ECG + balance video.

    Plays back in real time. LabScribe data is typically
    sampled far more densely (e.g. 100 Hz) than the output
    video's frame rate (config.fps), so a video frame is NOT
    generated for every data row -- that would make playback
    run roughly (sample_rate / fps) times slower than the
    actual recording. Instead, one frame is generated every
    1000 / config.fps milliseconds of *recorded* time,
    spanning the full UnixTime_ms range of the recording, and
    each frame is built from the balance/ECG samples nearest
    to that frame's real-world timestamp.


    Required balance columns:

        UnixTime_ms

        TL
        TR
        BL
        BR

        COP_X_Smooth

        COP_Y_Smooth



    Required ECG columns:

        UnixTime_ms

        ECG signal
    """

    if config is None:

        config = VisualizationConfig()


    validate_video_inputs(
        balance_df,
        ecg_df,
    )



    # --------------------------------------------------------
    # PRECOMPUTE STATIC VALUES
    # --------------------------------------------------------

    geometry = create_board_geometry(
        config
    )


    trail = create_cop_trail(
        config.cop_trail_length
    )


    maximum_force = np.median(

        balance_df["TotalWeight"]

    ) / 4.0 * 1.8



    ecg_signal = ecg_df["ECG"].values


    ecg_scale = calculate_ecg_scale(
        ecg_signal
    )


    heart_rate_values = (
        ecg_df["HeartRate"].values
        if "HeartRate" in ecg_df.columns
        else None
    )


    writer = create_video_writer(
        output_file,
        config,
    )



    balance_times = (
        balance_df["UnixTime_ms"]
        .to_numpy(dtype=float)
    )


    ecg_times = (
        ecg_df["UnixTime_ms"]
        .to_numpy(dtype=float)
    )



    # --------------------------------------------------------
    # REAL-TIME FRAME SCHEDULE
    #
    # One frame every 1000/fps ms of recorded time, spanning
    # the full recording -- this is what makes playback real
    # time regardless of how densely the source data was
    # sampled.
    # --------------------------------------------------------

    start_time = float(
        balance_times[0]
    )

    end_time = float(
        balance_times[-1]
    )

    frame_interval_ms = (
        1000.0
        /
        config.fps
    )

    if end_time <= start_time:

        total_frames = 1

    else:

        total_frames = int(
            np.floor(
                (end_time - start_time)
                / frame_interval_ms
            )
        ) + 1


    ecg_window = create_ecg_window()

    ecg_window_ms = (
        ecg_window.window_seconds
        * 1000.0
    )



    # --------------------------------------------------------
    # RENDER LOOP
    # --------------------------------------------------------

    for frame_index in range(
        total_frames
    ):

        target_time = (
            start_time
            +
            frame_index * frame_interval_ms
        )


        balance_index = find_closest_index(

            balance_times,

            target_time,
        )

        row = balance_df.iloc[
            balance_index
        ]


        ecg_index = find_closest_index(

            ecg_times,

            target_time,
        )


        # Trailing window of ECG samples covering the last
        # ecg_window_ms of recorded time, for the scrolling
        # ECG waveform display.

        window_start_index = np.searchsorted(

            ecg_times,

            target_time - ecg_window_ms,

            side="left",
        )

        window_end_index = ecg_index + 1

        if window_end_index <= window_start_index:

            window_start_index = max(
                ecg_index - 1,
                0,
            )


        ecg_values = ecg_signal[
            window_start_index:window_end_index
        ].tolist()


        hr = None

        if heart_rate_values is not None:

            hr = heart_rate_values[
                ecg_index
            ]


        frame = render_frame(

            row,

            ecg_values,

            hr,

            geometry,

            trail,

            ecg_scale,

            maximum_force,

            config,

            markers,
        )


        writer.write(
            frame
        )



    writer.release()

# ============================================================
# Visualization Metadata
# ============================================================

@dataclass(slots=True)
class RenderProgress:
    """
    Tracks rendering progress.
    """

    current_frame: int = 0

    total_frames: int = 0


    def percentage(self) -> float:
        """
        Return rendering completion percentage.
        """

        if self.total_frames <= 0:

            return 0.0


        return (
            self.current_frame
            /
            self.total_frames
            *
            100.0
        )



# ============================================================
# Event Marker Support (Future)
# ============================================================

@dataclass(slots=True)
class VisualizationEvent:
    """
    Future event marker representation.

    Not currently used by rendering.

    Intended for:

        Baseline

        VR_Start

        Stimulus_1

        Recovery
    """

    name: str

    timestamp_ms: int



def find_active_events(
    events: list[VisualizationEvent],
    timestamp_ms: int,
) -> list[VisualizationEvent]:
    """
    Find events occurring at or before
    the current timestamp.

    Future helper for synchronized
    experiment markers.
    """

    active = []


    for event in events:

        if event.timestamp_ms <= timestamp_ms:

            active.append(
                event
            )


    return active



# ============================================================
# Video Validation
# ============================================================

def validate_video_inputs(
    balance_df: pd.DataFrame,
    ecg_df: pd.DataFrame,
) -> None:
    """
    Validate inputs before rendering.
    """

    balance_required = [

        "UnixTime_ms",

        "TL",

        "TR",

        "BL",

        "BR",

        "COP_X_Smooth",

        "COP_Y_Smooth",

        "TotalWeight",

    ]


    missing_balance = [

        column

        for column in balance_required

        if column not in balance_df.columns

    ]


    if missing_balance:

        raise ValueError(
            f"Missing balance columns: {missing_balance}"
        )



    ecg_required = [

        "UnixTime_ms",

        "ECG",

    ]


    missing_ecg = [

        column

        for column in ecg_required

        if column not in ecg_df.columns

    ]


    if missing_ecg:

        raise ValueError(
            f"Missing ECG columns: {missing_ecg}"
        )



# ============================================================
# Public API
# ============================================================

__all__ = [

    # Configuration

    "VisualizationConfig",

    "BoardGeometry",

    "ECGWindow",

    "ECGScale",


    # Geometry

    "create_board_geometry",


    # Balance rendering

    "draw_balance_board",

    "draw_force_sensor",

    "interpolate_color",

    "normalize_force",


    # COP

    "COPTrail",

    "create_cop_trail",

    "draw_cop_trail",

    "draw_cop_position",

    "cop_to_pixel",


    # ECG

    "calculate_ecg_scale",

    "draw_ecg_window",

    "draw_ecg_waveform",

    "draw_ecg_markers",


    # Video

    "render_balance_video",

    "render_frame",

    "create_video_writer",


    # Validation

    "validate_video_inputs",


    # Future events

    "VisualizationEvent",

    "find_active_events",


    # Progress

    "RenderProgress",
]