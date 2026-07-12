"""
Regression test for tracker_types.py

Verifies that the core dataclasses behave exactly as expected.
"""

import numpy as np

from Eyetrackers.Core.tracker_types import (
    FrameMetadata,
    FramePacket,
    SyncPair,
)


def run():

    print("Running tracker_types tests...")

    #
    # Create a dummy image
    #
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    #
    # Metadata
    #
    metadata = FrameMetadata(
        frame_number=42,
        unix_ms=1_000_000,
        receive_ms=1_000_008,
        clock_offset_ms=8,
    )

    #
    # FramePacket
    #
    packet = FramePacket(
        image=image,
        metadata=metadata,
    )

    #
    # Basic fields
    #
    assert packet.frame_number == 42
    assert packet.capture_ms == 1_000_000
    assert packet.receive_ms == 1_000_008

    #
    # Latency should be receive-capture
    #
    assert packet.latency_ms == 8

    #
    # Image should be unchanged
    #
    assert packet.image.shape == (480, 640, 3)

    #
    # Create second packet
    #
    metadata2 = FrameMetadata(
        frame_number=43,
        unix_ms=1_000_003,
        receive_ms=1_000_010,
        clock_offset_ms=7,
    )

    packet2 = FramePacket(
        image=image.copy(),
        metadata=metadata2,
    )

    #
    # SyncPair
    #
    pair = SyncPair(
        left=packet,
        right=packet2,
        sync_timestamp_ms=min(
            packet.capture_ms,
            packet2.capture_ms,
        ),
        delta_ms=3,
    )

    #
    # Verify pair
    #
    assert pair.left.frame_number == 42
    assert pair.right.frame_number == 43

    assert pair.sync_timestamp_ms == 1_000_000
    assert pair.delta_ms == 3

    print("PASS")


if __name__ == "__main__":
    run()