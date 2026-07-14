"""
tracker_types.py

Shared data structures used throughout the ESP32 eye tracker application.

This module intentionally contains no networking or application logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time

import numpy as np


# ==========================================================
# Camera States
# ==========================================================

class CameraState(Enum):
    """Current operating state of a camera."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    STREAMING = auto()
    RECONNECTING = auto()
    STOPPING = auto()
    STOPPED = auto()


# ==========================================================
# Synchronizer States
# ==========================================================

class SyncState(Enum):
    IDLE = auto()
    WAITING_FOR_CAMERAS = auto()
    RUNNING = auto()
    STOPPED = auto()


# ==========================================================
# Recorder States
# ==========================================================

class RecorderState(Enum):
    IDLE = auto()
    RECORDING = auto()
    STOPPED = auto()


# ==========================================================
# ESP32 Metadata
# ==========================================================

@dataclass(slots=True)
class ESP32Metadata:
    """
    Metadata embedded by the ESP32 in each MJPEG frame.
    """

    frame_number: int

    capture_timestamp_ms: int

    receive_timestamp_ms: int

    clock_offset_ms: int

    @property
    def latency_ms(self) -> int:
        return self.receive_timestamp_ms - self.capture_timestamp_ms


# ==========================================================
# Frame Packet
# ==========================================================

@dataclass(slots=True)
class FramePacket:
    """
    Represents one decoded frame and its metadata.
    """

    metadata: ESP32Metadata

    image: np.ndarray

    decode_time: float = field(default_factory=time.perf_counter)

    @property
    def capture_ms(self) -> int:
        return self.metadata.capture_timestamp_ms

    @property
    def receive_ms(self) -> int:
        return self.metadata.receive_timestamp_ms

    @property
    def frame_number(self) -> int:
        return self.metadata.frame_number

    @property
    def latency_ms(self) -> int:
        return self.metadata.latency_ms


# ==========================================================
# Camera Statistics
# ==========================================================

@dataclass(slots=True)
class CameraStatistics:
    """
    Runtime statistics for a camera.
    """

    frames_received: int = 0

    frames_decoded: int = 0

    bad_headers: int = 0

    decode_failures: int = 0

    reconnects: int = 0

    bytes_received: int = 0

    fps: float = 0.0

    last_report_time: float = field(default_factory=time.perf_counter)

    frames_since_report: int = 0


# ==========================================================
# Camera Configuration
# ==========================================================

@dataclass(slots=True)
class CameraConfig:
    """
    Static configuration for one ESP32 camera.
    """

    name: str

    stream_url: str

    output_video: str | None = None

    buffer_size: int = 45

    sync_window_ms: int = 150

    brightness: int = 0

    contrast: float = 1.5

    saturation: float = 0.0

    gamma: float = 1.8


# ==========================================================
# Synchronization Result
# ==========================================================

@dataclass(slots=True)
class SyncPair:
    """
    Result returned by the synchronizer for one recording tick.
    """

    left: FramePacket
    right: FramePacket

    sync_timestamp_ms: int
    delta_ms: int

    status: str = "SYNCED"

    @property
    def capture_delta_ms(self) -> Optional[int]:
        if self.dropped:
            return None

        return abs(
            self.left.capture_ms -
            self.right.capture_ms
        )

    @property
    def receive_delta_ms(self) -> Optional[int]:
        if self.dropped:
            return None

        return abs(
            self.left.receive_ms -
            self.right.receive_ms
        )
    
    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    @property
    def dropped(self) -> bool:
        """
        Return True if this synchronization pair represents
        a dropped or incomplete frame pair.

        A pair is considered dropped if either camera frame
        is missing or the synchronization status indicates
        anything other than a successful match.
        """

        if self.left is None:
            return True

        if self.right is None:
            return True

        status = getattr(self, "status", "SYNCED")

        return status != "SYNCED"


# ==========================================================
# CSV Record
# ==========================================================

@dataclass(slots=True)
class CSVRecord:
    """
    One row written to the synchronization CSV.
    """

    tick_ms: int

    left_frame: Optional[int]

    left_capture_ms: Optional[int]

    left_receive_ms: Optional[int]

    right_frame: Optional[int]

    right_capture_ms: Optional[int]

    right_receive_ms: Optional[int]

    capture_delta_ms: Optional[int]

    receive_delta_ms: Optional[int]

    left_latency_ms: Optional[int]

    right_latency_ms: Optional[int]

    status: str
