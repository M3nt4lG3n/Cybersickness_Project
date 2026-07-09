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
from collections import deque
from typing import Optional

from mjpeg import MJPEGStream, StreamDisconnected
from tracker_types import (
    CameraConfig,
    CameraState,
    CameraStatistics,
    FramePacket,
)

import config

class CameraBuffer:

    def __init__(self, max_size=120):

        self.frames = deque(maxlen=max_size)

        self.lock = threading.Lock()


    def add(self, frame):

        with self.lock:
            self.frames.append(frame)


    def oldest(self):

        with self.lock:

            if not self.frames:
                return None

            return self.frames[0]


    def remove_oldest(self):

        with self.lock:

            if not self.frames:
                return None

            return self.frames.popleft()


    def closest(self, timestamp_ms):

        with self.lock:

            if not self.frames:
                return None

            return min(
                self.frames,
                key=lambda f:
                    abs(
                        f.capture_ms -
                        timestamp_ms
                    )
            )


    def clear(self):

        with self.lock:
            self.frames.clear()


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
        # Ring buffer of FramePacket objects.
        #
        self.buffer = CameraBuffer()

        #
        # Synchronization primitives.
        #
        self.buffer_lock = threading.Lock()

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

        with self.buffer_lock:

            self.buffer.clear()

    # --------------------------------------------------

    def buffer_size(self) -> int:

        with self.buffer_lock:

            return len(self.buffer)

    # --------------------------------------------------

    def latest_frame(self) -> Optional[FramePacket]:

        with self.buffer_lock:

            if not self.buffer:
                return None

            return self.buffer[-1]

    # --------------------------------------------------

    def append_frame(
        self,
        packet: FramePacket
    ):

        with self.buffer_lock:

            self.buffer.append(packet)

    # --------------------------------------------------

    def wait_for_first_frame(
        self,
        timeout: float = 10.0
    ) -> bool:

        deadline = time.time() + timeout

        while time.time() < deadline:

            if self.buffer_size() > 0:
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

            packet = FramePacket(
                image=image,
                frame_number=frame_number,
                capture_ms=esp_timestamp,
                receive_ms=current_time
            )
            
            self.buffer.add(packet)

            self.stats.frames_received += 1
            self.stats.frames_decoded += 1

            self.last_capture_timestamp = (
                packet.capture_ms
            )

            self.last_receive_timestamp = (
                packet.receive_ms
            )

            self._update_clock_offset(packet)

            self.append_frame(packet)

            self._update_fps()

    # --------------------------------------------------
    # Clock Synchronization
    # --------------------------------------------------

    def _update_clock_offset(
        self,
        packet: FramePacket
    ):

        newest_offset = (
            packet.receive_ms -
            packet.capture_ms
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
    # Synchronization API
    # --------------------------------------------------

    def closest_to(
        self,
        target_receive_ms: int
    ) -> Optional[FramePacket]:
        """
        Return the buffered frame whose capture timestamp is
        closest to the requested PC receive time.

        The PC timestamp is converted into the ESP32 clock
        domain using the current clock offset estimate.
        """

        if self.clock_offset_ms is None:
            return None

        target_capture_ms = (
            target_receive_ms -
            self.clock_offset_ms
        )

        with self.buffer_lock:

            if not self.buffer:
                return None

            best = min(
                self.buffer,
                key=lambda packet: abs(
                    packet.capture_ms -
                    target_capture_ms
                )
            )

        if (
            abs(
                best.capture_ms -
                target_capture_ms
            )
            >
            self.config.sync_window_ms
        ):
            return None

        return best

    # --------------------------------------------------

    def oldest_frame(self) -> Optional[FramePacket]:

        with self.buffer_lock:

            if not self.buffer:
                return None

            return self.buffer[0]

    # --------------------------------------------------

    def newest_frame(self) -> Optional[FramePacket]:

        return self.latest_frame()

    # --------------------------------------------------

    def pop_oldest(self) -> Optional[FramePacket]:

        with self.buffer_lock:

            if not self.buffer:
                return None

            return self.buffer.popleft()

    # --------------------------------------------------

    def remove_before(
        self,
        capture_timestamp_ms: int
    ) -> int:
        """
        Remove frames older than the specified capture time.

        Returns
        -------
        int
            Number of discarded frames.
        """

        removed = 0

        with self.buffer_lock:

            while self.buffer:

                if (
                    self.buffer[0].capture_ms
                    >=
                    capture_timestamp_ms
                ):
                    break

                self.buffer.popleft()

                removed += 1

        return removed

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
                self.buffer_size() > 0
            ):
                return True

            if self.stop_event.is_set():
                return False

            time.sleep(0.01)

        return False

    # --------------------------------------------------

    def __len__(self):

        return self.buffer_size()

    # --------------------------------------------------

    def __iter__(self):

        with self.buffer_lock:

            #
            # Iterate over a snapshot so the acquisition
            # thread can continue writing without blocking.
            #
            return iter(
                list(self.buffer)
            )

    # --------------------------------------------------

    def __repr__(self):

        return (

            f"Camera("

            f"name={self.config.name!r}, "

            f"state={self.state.name}, "

            f"frames={len(self.buffer)}, "

            f"fps={self.stats.fps:.1f}"

            f")"

        )

    def oldest_frame(self):

        return self.buffer.oldest()



    def pop_oldest(self):
    
        return self.buffer.remove_oldest()
    
    
    
    def closest_to(self, timestamp_ms):
    
        return self.buffer.closest(
            timestamp_ms
        )
