"""
Regression test for tracker_types.py

Verifies that the core dataclasses behave exactly as expected.
"""

import numpy as np

from Eyetrackers.Core.tracker_types import (
    ESP32Metadata,
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
    # Create metadata
    #
    metadata = ESP32Metadata(
        frame_number=42,
        capture_timestamp_ms=1_000_000,
        receive_timestamp_ms=1_000_008,
        clock_offset_ms=7,
    )

    #
    # Create FramePacket
    #
    packet = FramePacket(
        metadata=metadata,
        image=image,
    )

    #
    # Verify FramePacket properties
    #
    assert packet.frame_number == 42
    assert packet.capture_ms == 1_000_000
    assert packet.receive_ms == 1_000_008
    assert packet.latency_ms == 8

    #
    # Verify metadata is preserved
    #
    assert packet.metadata.frame_number == 42
    assert packet.metadata.capture_timestamp_ms == 1_000_000
    assert packet.metadata.receive_timestamp_ms == 1_000_008
    assert packet.metadata.clock_offset_ms == 7

    #
    # Image should be unchanged
    #
    assert packet.image.shape == (480, 640, 3)
    assert np.array_equal(packet.image, image)

    #
    # Create second metadata object
    #
    metadata2 = ESP32Metadata(
        frame_number=43,
        capture_timestamp_ms=1_000_003,
        receive_timestamp_ms=1_000_010,
        clock_offset_ms=7,
    )

    #
    # Create second packet
    #
    packet2 = FramePacket(
        metadata=metadata2,
        image=image.copy(),
    )

    #
    # Verify second packet
    #
    assert packet2.frame_number == 43
    assert packet2.capture_ms == 1_000_003
    assert packet2.receive_ms == 1_000_010
    assert packet2.latency_ms == 7

    #
    # Create SyncPair
    #
    pair = SyncPair(
        left=packet,
        right=packet2,
        sync_timestamp_ms=min(packet.capture_ms, packet2.capture_ms),
        delta_ms=3,
    )

    #
    # Verify SyncPair contents
    #
    assert pair.left is packet
    assert pair.right is packet2

    assert pair.left.frame_number == 42
    assert pair.right.frame_number == 43

    assert pair.sync_timestamp_ms == 1_000_000
    assert pair.delta_ms == 3

    #
    # Verify images were not modified
    #
    assert np.array_equal(pair.left.image, image)
    assert np.array_equal(pair.right.image, image)

    print("PASS")


if __name__ == "__main__":
    run()