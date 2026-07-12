"""
Regression test for sync.py

Verifies timestamp-based stereo synchronization.
"""

import numpy as np

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Core.sync import StereoSynchronizer
from Eyetrackers.Core.tracker_types import (
    CameraConfig,
    ESP32Metadata,
    FramePacket,
)


def make_packet(frame_number: int, capture_ms: int):

    image = np.zeros((10, 10, 3), dtype=np.uint8)

    metadata = ESP32Metadata(
        frame_number=frame_number,
        unix_ms=capture_ms,
        receive_ms=capture_ms + 5,
        clock_offset_ms=5,
    )

    return FramePacket(
        image=image,
        metadata=metadata,
    )


def make_camera(name: str):

    cfg = CameraConfig(
        name=name,
        stream_url="http://dummy",
        brightness=0,
        contrast=0,
    )

    return Camera(cfg)


def run():

    print("Running StereoSynchronizer tests...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    #
    # Perfectly synchronized streams
    #

    left.append_frame(make_packet(1, 1000))
    left.append_frame(make_packet(2, 1033))
    left.append_frame(make_packet(3, 1066))

    right.append_frame(make_packet(11, 1002))
    right.append_frame(make_packet(12, 1034))
    right.append_frame(make_packet(13, 1068))

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    #
    # First pair
    #

    pair = sync.get_pair()

    assert pair is not None
    assert pair.left.frame_number == 1
    assert pair.right.frame_number == 11

    #
    # Second pair
    #

    pair = sync.get_pair()

    assert pair is not None
    assert pair.left.frame_number == 2
    assert pair.right.frame_number == 12

    #
    # Third pair
    #

    pair = sync.get_pair()

    assert pair is not None
    assert pair.left.frame_number == 3
    assert pair.right.frame_number == 13

    #
    # Nothing left
    #

    assert sync.get_pair() is None

    #
    # Statistics
    #

    assert sync.stats.total_pairs == 3
    assert sync.stats.failed_matches == 0

    assert sync.success_rate == 1.0

    print("PASS")

def failure_test():

    print("Running failure test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    left.append_frame(
        make_packet(1, 1000)
    )

    right.append_frame(
        make_packet(2, 5000)
    )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    pair = sync.get_pair()

    assert pair is None

    assert sync.stats.total_pairs == 0
    assert sync.stats.failed_matches == 1

    print("PASS")

def no_reuse_test():

    print("Running no-reuse test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    left.append_frame(
        make_packet(1, 1000)
    )

    left.append_frame(
        make_packet(2, 1004)
    )

    right.append_frame(
        make_packet(10, 1002)
    )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=5,
    )

    pair = sync.get_pair()

    assert pair is not None
    assert pair.right.frame_number == 10

    #
    # The right frame should have been consumed.
    # Therefore there is no second match.
    #

    pair = sync.get_pair()

    assert pair is None

    assert sync.stats.total_pairs == 1
    assert sync.stats.failed_matches == 1

    print("PASS")


if __name__ == "__main__":
    run()

    failure_test()

    no_reuse_test()