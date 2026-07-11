"""
mjpeg.py

Low-level MJPEG stream reader for ESP32 CameraWebServer.

This module intentionally performs NO synchronization,
recording, buffering, or threading.

Its only responsibility is to convert an HTTP multipart
MJPEG stream into decoded OpenCV frames with ESP32 metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests

from tracker_types import ESP32Metadata, FramePacket
import config


# ==========================================================
# MJPEG Stream Errors
# ==========================================================

class MJPEGError(Exception):
    """Base class for MJPEG stream errors."""


class StreamDisconnected(MJPEGError):
    """Raised when the HTTP stream unexpectedly ends."""


class InvalidFrame(MJPEGError):
    """Raised when a JPEG cannot be decoded."""


# ==========================================================
# Raw multipart frame
# ==========================================================

@dataclass(slots=True)
class MultipartFrame:

    header: bytes

    jpeg: bytes


# ==========================================================
# MJPEG Stream Reader
# ==========================================================

class MJPEGStream:
    """
    Reads one ESP32 MJPEG stream.

    This class preserves the custom parser used by the
    original project so the ESP32 metadata is never lost.
    """

    def __init__(
        self,
        url: str,
        timeout: int = config.HTTP_TIMEOUT
    ):

        self.url = url
        self.timeout = timeout

        self.response: Optional[requests.Response] = None

        self.stream = None

        self.buffer = b""

    # ------------------------------------------------------
    # Connection
    # ------------------------------------------------------

    def connect(self) -> None:

        self.response = requests.get(
            self.url,
            stream=True,
            timeout=self.timeout
        )

        self.response.raise_for_status()

        self.stream = self.response.raw

        self.buffer = b""

    # ------------------------------------------------------

    def disconnect(self) -> None:

        if self.response is not None:
            self.response.close()

        self.response = None
        self.stream = None
        self.buffer = b""

    # ------------------------------------------------------

    def reconnect(self):

        self.disconnect()

        time.sleep(config.RECONNECT_DELAY)

        self.connect()

    # ------------------------------------------------------
    # Reading
    # ------------------------------------------------------

    def read_chunk(self):

        if self.stream is None:
            raise StreamDisconnected(
                "MJPEG stream is not connected."
            )

        chunk = self.stream.read(config.READ_SIZE)

        if not chunk:
            raise StreamDisconnected(
                "No data received from stream."
            )

        self.buffer += chunk

    # ------------------------------------------------------
    # Multipart extraction
    # ------------------------------------------------------

    def _extract_multipart(self) -> Optional[MultipartFrame]:

        start = self.buffer.find(b"\xff\xd8")
        end = self.buffer.find(b"\xff\xd9")

        if start == -1 or end == -1:
            return None

        if end < start:
            self.buffer = self.buffer[start:]
            return None

        header = self.buffer[:start]

        jpeg = self.buffer[start:end + 2]

        self.buffer = self.buffer[end + 2:]

        return MultipartFrame(
            header=header,
            jpeg=jpeg
        )

    # ------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------

    @staticmethod
    def _parse_metadata(header: bytes) -> Optional[ESP32Metadata]:
        """
        Parse ESP32-specific metadata from the multipart header.

        Expected headers include:

            X-Frame: 1234
            X-UnixMs: 1712345678901
        """

        frame_number: Optional[int] = None
        capture_ms: Optional[int] = None

        try:
            header_text = header.decode(
                "utf-8",
                errors="ignore"
            )
        except Exception:
            return None

        for line in header_text.split("\r\n"):

            line = line.strip()

            if not line:
                continue

            if line.startswith("X-Frame:"):

                try:
                    frame_number = int(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass

            elif line.startswith("X-UnixMs:"):

                try:
                    capture_ms = int(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass

        if frame_number is None:
            return None

        if capture_ms is None:
            return None

        receive_ms = time.time_ns() // 1_000_000

        return ESP32Metadata(
            frame_number=frame_number,
            capture_timestamp_ms=capture_ms,
            receive_timestamp_ms=receive_ms,
            clock_offset_ms=receive_ms - capture_ms
        )

    # ------------------------------------------------------
    # JPEG decoding
    # ------------------------------------------------------

    @staticmethod
    def _decode_frame(jpeg: bytes) -> np.ndarray:
        """
        Decode one JPEG into an OpenCV BGR image.
        """

        image = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise InvalidFrame(
                "Unable to decode JPEG."
            )

        return image

    # ------------------------------------------------------
    # Image adjustments
    # ------------------------------------------------------

    @staticmethod
    def _adjust_image(
        image: np.ndarray,
        brightness: int,
        contrast: float
    ) -> np.ndarray:
        """
        Apply brightness and contrast corrections.

        Saturation and gamma corrections are intentionally
        handled later by the camera processing pipeline.
        """

        return cv2.convertScaleAbs(
            image,
            alpha=contrast,
            beta=brightness
        )

    # ------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------

    def next_frame(
        self,
        brightness: int = 0,
        contrast: float = 1.0
    ) -> FramePacket:
        """
        Return the next complete frame from the MJPEG stream.

        Blocks until a valid multipart JPEG has been received.
        """

        while True:

            multipart = self._extract_multipart()

            if multipart is None:
                self.read_chunk()
                continue

            metadata = self._parse_metadata(
                multipart.header
            )

            if metadata is None:
                #
                # Skip malformed multipart headers while
                # continuing to consume the stream.
                #
                continue

            image = self._decode_frame(
                multipart.jpeg
            )

            image = self._adjust_image(
                image,
                brightness,
                contrast
            )

            return FramePacket(
                metadata=metadata,
                image=image
            )

        # ------------------------------------------------------
        # Iterator support
        # ------------------------------------------------------

    def __iter__(self):
        """
        Allow:

            for frame in stream:

                ...
        """

        return self

    def __next__(self) -> FramePacket:

        return self.next_frame()

    # ------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------

    def __enter__(self):

        self.connect()

        return self

    def __exit__(
            self,
            exc_type,
            exc_value,
            traceback
    ):

        self.disconnect()

        return False

    # ==========================================================
    # Convenience function
    # ==========================================================

    def open_stream(url: str) -> MJPEGStream:
        """
        Create and connect an MJPEG stream.

        Example
        -------

        with open_stream(url) as stream:

            frame = stream.next_frame()
        """

        stream = MJPEGStream(url)

        stream.connect()

        return stream

    # ==========================================================
    # Module self-test
    # ==========================================================

    if __name__ == "__main__":

        import config

        print("Opening MJPEG stream...")

        try:

            with open_stream(
                    config.LEFT_STREAM
            ) as stream:

                while True:

                    packet = stream.next_frame()

                    image = packet.image

                    metadata = packet.metadata

                    cv2.putText(
                        image,
                        f"Frame: {metadata.frame_number}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Capture: {metadata.capture_timestamp_ms}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Latency: {metadata.latency_ms} ms",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow(
                        "MJPEG Test",
                        image
                    )

                    key = cv2.waitKey(1)

                    if key == ord("q"):
                        break

        finally:

            cv2.destroyAllWindows()
