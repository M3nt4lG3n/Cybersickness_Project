"""
config.py

Central configuration for the ESP32 Eye Tracker system.

This file contains no application logic—only configuration values and
factory functions used by the rest of the application.
"""

from __future__ import annotations

from pathlib import Path

from Eyetrackers.Core.tracker_types import CameraConfig


# ==========================================================
# Program Information
# ==========================================================

APP_NAME = "ESP32 Eye Tracker Recorder"

VERSION = "2.0"


# ==========================================================
# Directories
# ==========================================================

PROJECT_ROOT = Path(".")

RAW_VIDEO_DIR = PROJECT_ROOT / "Raw_Eye_Recordings"

CSV_DIR = PROJECT_ROOT / "Test_CSVs"

LOG_DIR = PROJECT_ROOT / "Logs"


for directory in (
    RAW_VIDEO_DIR,
    CSV_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# Output Files
# ==========================================================

LEFT_VIDEO = RAW_VIDEO_DIR / "left_eye.mp4"

RIGHT_VIDEO = RAW_VIDEO_DIR / "right_eye.mp4"

CSV_FILE = CSV_DIR / "test.csv"


# ==========================================================
# ESP32 Stream URLs
#
# Change ONLY these if the camera IPs change.
# ==========================================================

LEFT_STREAM = "http://140.228.2.93:81/stream"

RIGHT_STREAM = "http://140.228.2.118:81/stream"

# ==========================================================
# Enable Debug Messages
# ==========================================================

DEBUG_TIMESTAMPS = False

# ==========================================================
# Recording
# ==========================================================

TARGET_FPS = 10

FRAME_PERIOD_MS = 1000 // TARGET_FPS

SYNC_WINDOW_MS = 150

SYNC_TOLERANCE_MS = 25

BUFFER_SIZE = 120

OUTPUT_FPS = 15

VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240


# ==========================================================
# Camera Buffers
# ==========================================================

BUFFER_SIZE = 30


# ==========================================================
# Image Processing
# ==========================================================

BRIGHTNESS = 0

CONTRAST = 1.5

SATURATION = 0.0

GAMMA = 1.8

DRAW_DEBUG_OVERLAY = True


# ==========================================================
# Network
# ==========================================================

HTTP_TIMEOUT = 10

READ_SIZE = 4096

RECONNECT_DELAY = 2.0


# ==========================================================
# OpenCV
# ==========================================================

WINDOW_LEFT = "Left Eye"

WINDOW_RIGHT = "Right Eye"

WINDOW_DELAY_MS = 1


# ==========================================================
# Video Writer
# ==========================================================

VIDEO_CODEC = "mp4v"


# ==========================================================
# FPS Reporting
# ==========================================================

FPS_REPORT_INTERVAL = 1.0


# ==========================================================
# CSV Header
# ==========================================================

CSV_HEADER = [

    "SyncTickMs",

    "LeftFrame",
    "LeftCaptureMs",
    "LeftReceiveMs",

    "RightFrame",
    "RightCaptureMs",
    "RightReceiveMs",

    "CaptureDeltaMs",
    "ReceiveDeltaMs",

    "LeftLatencyMs",
    "RightLatencyMs",

    "Status"

]


# ==========================================================
# Camera Factory
# ==========================================================

WINDOW_STEREO = "Stereo"

LEFT_CAMERA = CameraConfig(

    name="LEFT",

    stream_url=LEFT_STREAM,

    output_video=str(LEFT_VIDEO),

    buffer_size=BUFFER_SIZE,

    sync_window_ms=SYNC_WINDOW_MS,

    brightness=BRIGHTNESS,

    contrast=CONTRAST,

    saturation=SATURATION,

    gamma=GAMMA

)


RIGHT_CAMERA = CameraConfig(

    name="RIGHT",

    stream_url=RIGHT_STREAM,

    output_video=str(RIGHT_VIDEO),

    buffer_size=BUFFER_SIZE,

    sync_window_ms=SYNC_WINDOW_MS,

    brightness=BRIGHTNESS,

    contrast=CONTRAST,

    saturation=SATURATION,

    gamma=GAMMA

)


CAMERAS = (
    LEFT_CAMERA,
    RIGHT_CAMERA,
)
