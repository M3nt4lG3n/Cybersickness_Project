"""
Regression test for display.py

Ensures that the Display module accepts a SyncPair
without throwing exceptions.
"""

import numpy as np

from Eyetrackers.Outputs.display import Display
from Eyetrackers.Core.tracker_types import (
    ESP32Metadata,
    FramePacket,
    SyncPair,
)


def make_packet(frame_number):

    image = np.zeros((480, 640, 3), dtype=np.uint8)

    metadata = ESP32Metadata(
        frame_number=frame_number,
        unix_ms=frame_number * 33,
        receive_ms=frame_number * 33 + 5,
        clock_offset_ms=5,
    )

    return FramePacket(
        image=image,
        metadata=metadata,
    )


def run():

    print("Running display test...")

    display = Display()

    pair = SyncPair(
        left=make_packet(1),
        right=make_packet(2),
        sync_timestamp_ms=33,
        delta_ms=2,
    )

    #
    # The test passes if no exception is raised.
    #

    display.show(pair)

    #
    # Allow the window to render briefly.
    #

    import cv2

    cv2.waitKey(100)

    if hasattr(display, "close"):
        display.close()

    print("PASS")


if __name__ == "__main__":
    run()