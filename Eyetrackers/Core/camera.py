"""
camera.py

High-level camera abstraction.

The Camera class owns:

    - MJPEG stream
    - acquisition thread
    - reconnect logic
    - frame buffer
    - clock synchronization
    - runtime statistics

Synchronization, recording, and display are handled elsewhere.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from Eyetrackers.Core.mjpeg import MJPEGStream, StreamDisconnected
from Eyetrackers.Core.tracker_types import (
    CameraConfig,
    CameraState,
    CameraStatistics,
    FramePacket,
)

from Eyetrackers.Core import config
from Eyetrackers.Core.framebuffer import FrameBuffer

class Camera:
    """
    Represents one ESP32 camera.

    A Camera continuously acquires frames from an MJPEG stream,
    timestamps them, estimates clock offset, and stores them
    in a ring buffer for later synchronization.
    """

    def __init__(self, camera_config: CameraConfig):

        self.config = camera_config

        self.state = CameraState.DISCONNECTED

        self.stream = MJPEGStream(
            camera_config.stream_url
        )

        #
        # Buffer containing FramePacket instances ordered by
        # ESP capture timestamp.
        #
        self.buffer = FrameBuffer()

        self.stop_event = threading.Event()

        self.thread: Optional[threading.Thread] = None

        #
        # Runtime statistics.
        #
        self.stats = CameraStatistics()

        #
        # Clock synchronization.
        #
        # This replaces the original CameraStream.clock_offset.
        #
        self.clock_offset_ms: Optional[int] = None

        #
        # Timestamp of the most recently received frame.
        #
        self.last_capture_timestamp: Optional[int] = None

        self.last_receive_timestamp: Optional[int] = None

        print("Camera buffer initialized")

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):

        if self.thread is not None:

            if self.thread.is_alive():
                return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=self.config.name
        )

        self.thread.start()

    # --------------------------------------------------

    def stop(self):

        self.stop_event.set()

        if self.thread is not None:

            self.thread.join(timeout=2.0)

        try:
            self.stream.disconnect()
        except Exception:
            pass

        self.state = CameraState.STOPPED

    # --------------------------------------------------
    # Buffer
    # --------------------------------------------------

    def clear_buffer(self):

        self.buffer.clear()

    # --------------------------------------------------

    def has_frames(self) -> bool:
        return self.buffer.size() > 0

    # --------------------------------------------------

    def buffer_empty(self) -> bool:
        return self.buffer_size() == 0
    
    # --------------------------------------------------

    def buffer_size(self) -> int:
        return self.buffer.size()

    # --------------------------------------------------

    def latest_frame(self) -> Optional[FramePacket]:
        return self.buffer.newest()

    # --------------------------------------------------

    def closest_frame(
        self,
        capture_timestamp_ms: int,
        tolerance_ms: int
    ) -> Optional[FramePacket]:
        """
        Return the frame whose capture timestamp is
        closest to the requested timestamp.
        """
        return self.buffer.closest(
            capture_timestamp_ms,
            tolerance_ms,
        )
    
    def consume_closest(
        self,
        capture_timestamp_ms: int,
        tolerance_ms: int,
    ) -> Optional[FramePacket]:
        """
        Return and remove the closest frame within the
        specified tolerance.
        """
        return self.buffer.consume_closest(
            capture_timestamp_ms,
            tolerance_ms,
        )

    # --------------------------------------------------

    def append_frame(self, packet: FramePacket):
        self.buffer.add(packet)

    # --------------------------------------------------

    def wait_for_first_frame(
        self,
        timeout: float = 10.0
    ) -> bool:

        deadline = time.time() + timeout

        while time.time() < deadline:

            if self.has_frames():
                return True

            time.sleep(0.01)

        return False

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    def is_running(self) -> bool:

        return (
            self.thread is not None
            and
            self.thread.is_alive()
        )
    
    @property
    def connected(self) -> bool:
        return self.state is CameraState.STREAMING

    # --------------------------------------------------
    # Acquisition thread
    # --------------------------------------------------

    def _run(self):

        while not self.stop_event.is_set():

            try:

                self.state = CameraState.CONNECTING

                self.stream.connect()

                self.state = CameraState.STREAMING

                self._capture_loop()

            except StreamDisconnected:

                self.stats.reconnects += 1

                self.state = CameraState.RECONNECTING

                try:
                    self.stream.disconnect()
                except Exception:
                    pass

                if not self.stop_event.wait(
                    config.RECONNECT_DELAY
                ):
                    continue

            except Exception as exc:

                print(
                    f"[{self.config.name}] "
                    f"Unexpected error: {exc}"
                )

                self.stats.reconnects += 1

                self.state = CameraState.RECONNECTING

                try:
                    self.stream.disconnect()
                except Exception:
                    pass

                if not self.stop_event.wait(
                    config.RECONNECT_DELAY
                ):
                    continue

    # --------------------------------------------------

    def _capture_loop(self):

        while not self.stop_event.is_set():

            try:

                packet = self.stream.next_frame(
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                )

            except StreamDisconnected:
                raise

            except Exception as exc:

                print(
                    f"[{self.config.name}] "
                    f"Frame decode failed: {exc}"
                )

                continue

            self.append_frame(packet)

            self.stats.frames_received += 1
            self.stats.frames_decoded += 1

            self.last_capture_timestamp = packet.capture_ms
            self.last_receive_timestamp = packet.receive_ms

            self._update_clock_offset(packet)

            self._update_fps()

    # --------------------------------------------------
    # Clock Synchronization
    # --------------------------------------------------

    def _update_clock_offset(
        self,
        packet: FramePacket
    ):

        newest_offset = packet.metadata.clock_offset_ms

        if config.DEBUG_TIMESTAMPS:
            print(
                f"[{self.config.name}] "
                f"Frame {packet.frame_number} "
                f"Capture={packet.capture_ms} "
                f"Latency={packet.latency_ms} ms"
            )

        #
        # Initialize from the first frame.
        #
        if self.clock_offset_ms is None:

            self.clock_offset_ms = newest_offset

            return

        #
        # Exponential moving average.
        #
        # Keeps the estimate stable while allowing
        # slow drift correction.
        #
        alpha = 0.02

        self.clock_offset_ms = int(

            (1.0 - alpha)
            *
            self.clock_offset_ms

            +

            alpha
            *
            newest_offset

        )

    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    def _update_fps(self):

        self.stats.frames_since_report += 1

        now = time.perf_counter()

        elapsed = (
            now -
            self.stats.last_report_time
        )

        if elapsed < config.FPS_REPORT_INTERVAL:
            return

        self.stats.fps = (

            self.stats.frames_since_report
            /
            elapsed

        )

        print(

            f"{self.config.name}: "

            f"{self.stats.fps:.1f} fps"

        )

        self.stats.frames_since_report = 0

        self.stats.last_report_time = now

    # --------------------------------------------------

    def newest_frame(self) -> Optional[FramePacket]:

        return self.latest_frame()
    
    @property
    def frame_count(self) -> int:
        return self.buffer.size()

    # --------------------------------------------------

    def remove_before(
        self,
        capture_timestamp_ms: int
    ) -> int:
        return self.buffer.remove_before(
            capture_timestamp_ms
        )

    # --------------------------------------------------

    def wait_until_ready(
        self,
        timeout: float = 10.0
    ) -> bool:
        """
        Wait until at least one valid frame has been
        received and clock synchronization has been
        established.
        """

        deadline = (
            time.time() +
            timeout
        )

        while time.time() < deadline:

            if (
                self.clock_offset_ms
                is not None
                and
                self.has_frames()
            ):
                return True

            if self.stop_event.is_set():
                return False

            time.sleep(0.01)

        return False
    
    # --------------------------------------------------

    def statistics(self) -> CameraStatistics:
        """
        Return the runtime statistics object.
        """
        return self.stats

    # --------------------------------------------------

    def __len__(self):

        return self.buffer_size()

    # --------------------------------------------------

    def snapshot(self):
        """
        Return a snapshot of the current buffer contents.
        """
        return self.buffer.snapshot()

    def __iter__(self):

        return iter(self.snapshot())

    # --------------------------------------------------

    def __repr__(self):
        return (
            f"Camera("
            f"name={self.config.name!r}, "
            f"state={self.state.name}, "
            f"frames={self.buffer.size()}, "
            f"fps={self.stats.fps:.1f}"
            f")"
        )

    def oldest_frame(self) -> Optional[FramePacket]:
        return self.buffer.oldest()

    def pop_oldest(self) -> Optional[FramePacket]:
        return self.buffer.remove_oldest()
    
    # --------------------------------------------------

    def consume_oldest(self) -> Optional[FramePacket]:
        """
        Remove and return the oldest buffered frame.

        This is the preferred API for consumers that process
        frames sequentially.
        """
        return self.buffer.remove_oldest()
