"""
unity_visualization.py

Renders Unity headset position/rotation data (PosX/Y/Z,
QuatX/Y/Z/W) as an animated 3D scene: a sphere translating
through space with a set of orientation axes attached to it
that rotate according to the recorded quaternion, plus a
fading trail of recent positions.

Mirrors the shape of visualization.py's balance-board video
renderer (dataclass config in, mp4 path out) so it can be
dropped into the same pipeline.
"""

from __future__ import annotations


from dataclasses import dataclass

from pathlib import Path

from typing import Optional

import numpy as np

import pandas as pd

import cv2

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ============================================================
# Required Columns
# ============================================================

POSITION_COLUMNS = ("PosX", "PosY", "PosZ")

QUATERNION_COLUMNS = ("QuatX", "QuatY", "QuatZ", "QuatW")

TIME_COLUMN = "UnixTime_ms"


# ============================================================
# Configuration
# ============================================================

@dataclass(slots=True)
class UnityVisualizationConfig:
    """
    User configurable Unity position/rotation video settings.
    """

    fps: int = 30

    # Playback speed multiplier. 1.0 = real-time,
    # 2.0 = twice as fast (video is half as long), etc.
    playback_speed: float = 1.0

    # Optional cap on rendered frames, mainly useful for
    # quick previews / debugging on long recordings.
    max_frames: Optional[int] = None

    sphere_radius: float = 0.05

    sphere_resolution: int = 10

    # Number of past position samples to draw as a fading trail.
    trail_length: int = 150

    axis_length: float = 0.15

    figsize: tuple[float, float] = (8.0, 8.0)

    dpi: int = 100

    elev: float = 20.0

    azim: float = -60.0

    # Extra padding (in data units) added around the bounding
    # box of recorded positions so the sphere never clips the
    # edge of the frame.
    padding: float = 0.1


# ============================================================
# Loading
# ============================================================

def load_unity_csv(path: Path) -> pd.DataFrame:
    """
    Load a Unity biometrics CSV and validate that the columns
    needed for the position/rotation video are present.
    """

    df = pd.read_csv(path)

    required = (TIME_COLUMN,) + POSITION_COLUMNS + QUATERNION_COLUMNS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Unity CSV is missing required column(s): "
            f"{', '.join(missing)}"
        )

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    return df


# ============================================================
# Quaternion Math
# ============================================================

def quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    """
    Convert a (x, y, z, w) quaternion into a 3x3 rotation
    matrix. Input is normalized defensively in case of
    floating point drift in the recorded data.
    """

    norm = np.sqrt(x * x + y * y + z * z + w * w)

    if norm == 0:

        return np.eye(3)

    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


# ============================================================
# Frame Sampling
# ============================================================

def _select_frame_indices(
    df: pd.DataFrame,
    config: UnityVisualizationConfig,
) -> np.ndarray:
    """
    Pick which rows of df to render as video frames, resampled
    to a constant fps so the output video's playback duration
    matches the recording (scaled by playback_speed).
    """

    time_ms = df[TIME_COLUMN].to_numpy()

    duration_s = (time_ms[-1] - time_ms[0]) / 1000.0

    if duration_s <= 0:

        return np.arange(len(df))

    video_duration_s = duration_s / config.playback_speed

    frame_count = max(
        int(round(video_duration_s * config.fps)),
        1,
    )

    if config.max_frames is not None:

        frame_count = min(
            frame_count,
            config.max_frames,
        )

    sample_times = np.linspace(
        time_ms[0],
        time_ms[-1],
        frame_count,
    )

    indices = np.searchsorted(
        time_ms,
        sample_times,
        side="left",
    )

    indices = np.clip(
        indices,
        0,
        len(df) - 1,
    )

    return indices


# ============================================================
# Geometry Helpers
# ============================================================

def _unit_sphere_mesh(
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute a unit sphere surface mesh so each frame only
    needs to scale + translate it.
    """

    u = np.linspace(0, 2 * np.pi, resolution)

    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))

    y = np.outer(np.sin(u), np.sin(v))

    z = np.outer(np.ones_like(u), np.cos(v))

    return x, y, z


def _axis_bounds(
    df: pd.DataFrame,
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Compute fixed x/y/z plot limits from the full recording so
    the camera framing doesn't jump around between frames.
    """

    bounds = []

    for column in POSITION_COLUMNS:

        values = df[column].to_numpy()

        low, high = values.min(), values.max()

        span = high - low

        # Guard against a near-static axis (e.g. a seated user
        # barely moving on one axis) collapsing to zero range.
        if span < padding:

            center = (high + low) / 2.0

            low, high = center - padding, center + padding

        bounds.append(
            (low - padding, high + padding)
        )

    return tuple(bounds)  # type: ignore[return-value]


# ============================================================
# Rendering
# ============================================================

def render_unity_video(
    unity_df: pd.DataFrame,
    output_file: str,
    config: UnityVisualizationConfig = UnityVisualizationConfig(),
) -> None:
    """
    Render the position/rotation video: a sphere that
    translates and rotates through 3D space according to the
    recorded PosX/Y/Z and QuatX/Y/Z/W columns.
    """

    frame_indices = _select_frame_indices(
        unity_df,
        config,
    )

    positions = unity_df[list(POSITION_COLUMNS)].to_numpy()

    quaternions = unity_df[list(QUATERNION_COLUMNS)].to_numpy()

    time_ms = unity_df[TIME_COLUMN].to_numpy()

    t0 = time_ms[0]

    xlim, ylim, zlim = _axis_bounds(
        unity_df,
        config.padding,
    )

    sphere_x, sphere_y, sphere_z = _unit_sphere_mesh(
        config.sphere_resolution
    )

    fig = plt.figure(
        figsize=config.figsize,
        dpi=config.dpi,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.view_init(
        elev=config.elev,
        azim=config.azim,
    )

    axis_colors = ("red", "green", "blue")

    video_writer: cv2.VideoWriter | None = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:

        for frame_number, row_index in enumerate(frame_indices):

            ax.cla()

            ax.set_xlim(*xlim)

            ax.set_ylim(*ylim)

            ax.set_zlim(*zlim)

            ax.set_xlabel("X")

            ax.set_ylabel("Y")

            ax.set_zlabel("Z")

            ax.view_init(
                elev=config.elev,
                azim=config.azim,
            )

            px, py, pz = positions[row_index]

            qx, qy, qz, qw = quaternions[row_index]

            # --------------------------------------------
            # Trail of recent positions
            # --------------------------------------------

            trail_start = max(
                0,
                row_index - config.trail_length,
            )

            trail = positions[trail_start:row_index + 1]

            if len(trail) > 1:

                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    trail[:, 2],
                    color="gray",
                    alpha=0.5,
                    linewidth=1.0,
                )

            # --------------------------------------------
            # Sphere at current position
            # --------------------------------------------

            ax.plot_surface(
                sphere_x * config.sphere_radius + px,
                sphere_y * config.sphere_radius + py,
                sphere_z * config.sphere_radius + pz,
                color="steelblue",
                alpha=0.85,
                linewidth=0,
                antialiased=True,
            )

            # --------------------------------------------
            # Orientation axes (rotated unit vectors)
            # --------------------------------------------

            rotation = quaternion_to_rotation_matrix(
                qx, qy, qz, qw
            )

            for axis_index in range(3):

                direction = rotation[:, axis_index] * config.axis_length

                ax.plot(
                    [px, px + direction[0]],
                    [py, py + direction[1]],
                    [pz, pz + direction[2]],
                    color=axis_colors[axis_index],
                    linewidth=2.0,
                )

            elapsed_s = (
                time_ms[row_index] - t0
            ) / 1000.0

            ax.set_title(
                f"t = {elapsed_s:6.2f}s   "
                f"frame {frame_number + 1}/{len(frame_indices)}"
            )

            # --------------------------------------------
            # Rasterize the figure and hand the frame to
            # OpenCV. This avoids depending on a system
            # ffmpeg binary being on PATH.
            # --------------------------------------------

            fig.canvas.draw()

            width, height = fig.canvas.get_width_height()

            frame_rgba = np.asarray(
                fig.canvas.buffer_rgba()
            ).reshape(height, width, 4)

            frame_bgr = cv2.cvtColor(
                frame_rgba,
                cv2.COLOR_RGBA2BGR,
            )

            if video_writer is None:

                video_writer = cv2.VideoWriter(
                    str(output_file),
                    fourcc,
                    config.fps,
                    (width, height),
                )

                if not video_writer.isOpened():

                    raise RuntimeError(
                        "Failed to open video writer for "
                        f"'{output_file}'. The mp4v codec may "
                        "not be available in this OpenCV build."
                    )

            video_writer.write(frame_bgr)

    finally:

        if video_writer is not None:

            video_writer.release()

    plt.close(fig)