"""
Regression test for recorder.py

Verifies that stereo video files are written correctly.
"""

import os
import tempfile

import cv2
import numpy as np

from Eyetrackers.Outputs.recorder import Recorder
from Eyetrackers.Core.tracker_types import (
    ESP32Metadata,
    FramePacket,
    SyncPair,
)


FRAME_SIZE = (320, 240)
FPS = 30
NUM_FRAMES = 30


def make_packet(frame_number):

    image = np.zeros(
        (FRAME_SIZE[1], FRAME_SIZE[0], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        image,
        str(frame_number),
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

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

    print("Running recorder test...")

    with tempfile.TemporaryDirectory() as temp:

        left_file = os.path.join(
            temp,
            "left.mp4",
        )

        right_file = os.path.join(
            temp,
            "right.mp4",
        )

        recorder = Recorder(
            left_file,
            right_file,
            FPS,
            FRAME_SIZE,
        )

        for i in range(NUM_FRAMES):

            pair = SyncPair(
                left=make_packet(i),
                right=make_packet(i),
                sync_timestamp_ms=i * 33,
                delta_ms=0,
            )

            recorder.write(pair)

        recorder.close()

        assert os.path.exists(left_file)
        assert os.path.exists(right_file)

        left = cv2.VideoCapture(left_file)

        frames = 0

        while True:

            ok, frame = left.read()

            if not ok:
                break

            frames += 1

        left.release()

        assert frames == NUM_FRAMES

        print("PASS")


if __name__ == "__main__":
    run()