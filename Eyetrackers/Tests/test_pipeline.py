"""
Regression test for the complete synchronization pipeline.

No ESP32 cameras are required.

The test generates synthetic FramePackets and verifies that
the synchronizer produces the expected SyncPairs.
"""

import numpy as np

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Core.sync import StereoSynchronizer
from Eyetrackers.Core.tracker_types import (
    CameraConfig,
    ESP32Metadata,
    FramePacket,
)

import random


NUM_FRAMES = 100


def make_packet(frame_number, capture_ms):

    image = np.zeros((20, 20, 3), dtype=np.uint8)

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


def make_camera(name):

    cfg = CameraConfig(
        name=name,
        stream_url="dummy",
        brightness=0,
        contrast=0,
    )

    return Camera(cfg)


def run():

    print("Running pipeline regression test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    #
    # Simulate a 30 FPS stereo capture.
    #
    for i in range(NUM_FRAMES):

        timestamp = i * 33

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

        #
        # Right camera delayed by 2 ms
        #
        right.append_frame(
            make_packet(
                i,
                timestamp + 2,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    pairs = []

    while True:

        pair = sync.get_pair()

        if pair is None:
            break

        pairs.append(pair)

    #
    # Every frame should synchronize.
    #
    assert len(pairs) == NUM_FRAMES

    #
    # Verify ordering.
    #
    for i, pair in enumerate(pairs):

        assert pair.left.frame_number == i
        assert pair.right.frame_number == i

    #
    # Verify synchronization accuracy.
    #
    average_delta = sum(
        p.delta_ms
        for p in pairs
    ) / len(pairs)

    assert average_delta == 2

    #
    # Verify statistics.
    #
    assert sync.stats.total_pairs == NUM_FRAMES

    assert sync.stats.failed_matches == 0

    print()

    print("Frames synchronized :", len(pairs))

    print("Average delta       :", average_delta)

    print("Success rate        :", sync.success_rate)

    print()

    print("PASS")


def jitter_test():

    print("Running jitter test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    random.seed(42)

    expected_pairs = 100

    for i in range(expected_pairs):

        timestamp = i * 33

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

        jitter = random.randint(-5, 5)

        right.append_frame(
            make_packet(
                i,
                timestamp + jitter,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    pairs = []

    while True:

        pair = sync.get_pair()

        if pair is None:
            break

        pairs.append(pair)

    assert len(pairs) == expected_pairs

    for pair in pairs:

        assert pair.delta_ms <= 5

    assert sync.stats.failed_matches == 0

    print("PASS")

def dropped_frame_test():

    print("Running dropped frame test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    timestamps = [
        0,
        33,
        66,
        99,
        132,
        165,
        198,
        231,
    ]

    #
    # Left receives every frame.
    #

    for i, timestamp in enumerate(timestamps):

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

    #
    # Right drops frame 66 and 165.
    #

    right_frames = [
        (0, 0),
        (1, 33),
        (3, 99),
        (4, 132),
        (6, 198),
        (7, 231),
    ]

    for frame_number, timestamp in right_frames:

        right.append_frame(
            make_packet(
                frame_number,
                timestamp,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    pairs = []

    while True:

        pair = sync.get_pair()

        if pair is None:

            #
            # Stop when no left frames remain.
            #

            if left.buffer_size() == 0:
                break

            continue

        pairs.append(pair)

    assert len(pairs) == 6

    assert sync.stats.total_pairs == 6
    assert sync.stats.failed_matches == 2

    print("PASS")

def statistics_test():

    print("Running statistics test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    deltas = [1, 2, 3, 4, 5]

    for i, delta in enumerate(deltas):

        timestamp = i * 100

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

        right.append_frame(
            make_packet(
                i,
                timestamp + delta,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    while sync.get_pair() is not None:
        pass

    expected_average = sum(deltas) / len(deltas)

    assert sync.stats.total_pairs == 5
    assert sync.stats.failed_matches == 0

    assert abs(
        sync.stats.average_delta_ms -
        expected_average
    ) < 1e-6

    print("PASS")

def long_session_test():

    print("Running long session test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    frames = 5000

    for i in range(frames):

        timestamp = i * 33

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

        right.append_frame(
            make_packet(
                i,
                timestamp + 2,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    count = 0

    while True:

        pair = sync.get_pair()

        if pair is None:
            break

        count += 1

    assert count == frames

    assert sync.success_rate == 1.0

    print("PASS")

def buffer_cleanup_test():

    print("Running buffer cleanup test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    for i in range(50):

        timestamp = i * 33

        left.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

        right.append_frame(
            make_packet(
                i,
                timestamp,
            )
        )

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=5,
    )

    while sync.get_pair() is not None:
        pass

    #
    # Left should always be empty.
    #

    assert left.buffer_size() == 0

    #
    # Right should also be empty if consume_closest()
    # is working correctly.
    #

    assert right.buffer_size() == 0

    print("PASS")

def out_of_order_test():

    print("Running out-of-order arrival test...")

    left = make_camera("LEFT")
    right = make_camera("RIGHT")

    #
    # Left camera arrives normally.
    #

    for i in range(10):

        capture = i * 33

        packet = make_packet(i, capture)

        #
        # Simulate receive time
        #
        packet.metadata.receive_ms = capture + 5

        left.append_frame(packet)

    #
    # Right camera receives packets wildly out of order.
    # Capture timestamps remain correct.
    #

    receive_offsets = [
        30,
        4,
        22,
        1,
        18,
        6,
        27,
        2,
        14,
        5,
    ]

    for i in range(10):

        capture = i * 33 + 2

        packet = make_packet(i, capture)

        packet.metadata.receive_ms = (
            capture +
            receive_offsets[i]
        )

        right.append_frame(packet)

    sync = StereoSynchronizer(
        left,
        right,
        tolerance_ms=10,
    )

    pairs = []

    while True:

        pair = sync.get_pair()

        if pair is None:
            break

        pairs.append(pair)

    assert len(pairs) == 10

    for i, pair in enumerate(pairs):

        #
        # Synchronization should completely ignore
        # receive order.
        #

        assert pair.left.frame_number == i
        assert pair.right.frame_number == i

        assert pair.delta_ms == 2

    print("PASS")

if __name__ == "__main__":
    run()

    jitter_test()

    dropped_frame_test()

    statistics_test()

    long_session_test()

    buffer_cleanup_test()

    out_of_order_test()