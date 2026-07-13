"""
Regression test for framebuffer.py

Verifies buffering, lookup, and consumption behavior.
"""

import numpy as np

from Eyetrackers.Core.framebuffer import FrameBuffer
from Eyetrackers.Core.tracker_types import (
    ESP32Metadata,
    FramePacket,
)


def make_packet(
    frame_number,
    capture_ms,
    receive_ms=None,
):

    if receive_ms is None:
        receive_ms = capture_ms + 5

    image = np.zeros((10, 10, 3), dtype=np.uint8)

    metadata = ESP32Metadata(
        frame_number=frame_number,
        capture_timestamp_ms=capture_ms,
        receive_timestamp_ms=receive_ms,
        clock_offset_ms=8,
    )

    return FramePacket(
        image=image,
        metadata=metadata,
    )


def run():

    print("Running FrameBuffer tests...")

    buffer = FrameBuffer()

    p1 = make_packet(1, 1000)
    p2 = make_packet(2, 1033)
    p3 = make_packet(3, 1066)

    #
    # Add frames
    #

    buffer.add(p1)
    buffer.add(p2)
    buffer.add(p3)

    assert buffer.size() == 3

    #
    # Oldest should not remove
    #

    oldest = buffer.oldest()

    assert oldest.frame_number == 1
    assert buffer.size() == 3

    #
    # Remove oldest
    #

    removed = buffer.remove_oldest()

    assert removed.frame_number == 1
    assert buffer.size() == 2

    #
    # Closest lookup
    #

    closest = buffer.closest(
        1040,
        tolerance_ms=20,
    )

    assert closest is not None
    assert closest.frame_number == 2

    #
    # Lookup outside tolerance
    #

    closest = buffer.closest(
        5000,
        tolerance_ms=20,
    )

    assert closest is None

    #
    # Consume closest
    #

    consumed = buffer.consume_closest(
        1040,
        tolerance_ms=20,
    )

    assert consumed.frame_number == 2

    assert buffer.size() == 1

    #
    # Ensure consumed frame cannot be reused
    #

    closest = buffer.closest(
        1040,
        tolerance_ms=20,
    )

    assert closest is None

    #
    # Remaining frame should be #3
    #

    remaining = buffer.oldest()

    assert remaining.frame_number == 3

    print("PASS")


if __name__ == "__main__":
    run()