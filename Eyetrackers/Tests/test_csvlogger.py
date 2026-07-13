"""
Regression test for csvlogger.py

Verifies that the logger writes the expected CSV data.
"""

import csv
import os
import tempfile

import numpy as np

from Eyetrackers.Outputs.csvlogger import CSVLogger
from Eyetrackers.Core.tracker_types import (
    ESP32Metadata,
    FramePacket,
    SyncPair,
)


NUM_ROWS = 25


def make_packet(frame_number, capture_ms):

    image = np.zeros((20, 20, 3), dtype=np.uint8)

    metadata = ESP32Metadata(
        frame_number=frame_number,
        capture_timestamp_ms=capture_ms,
        receive_timestamp_ms=capture_ms + 5,
        clock_offset_ms=5,
    )

    return FramePacket(
        image=image,
        metadata=metadata,
    )


def run():

    print("Running CSV logger test...")

    with tempfile.TemporaryDirectory() as temp:

        csv_file = os.path.join(
            temp,
            "session.csv",
        )

        logger = CSVLogger(csv_file)

        for i in range(NUM_ROWS):

            pair = SyncPair(
                left=make_packet(i, i * 33),
                right=make_packet(i, i * 33 + 2),
                sync_timestamp_ms=i * 33,
                delta_ms=2,
            )

            logger.write(pair)

        logger.close()

        assert os.path.exists(csv_file)

        with open(
            csv_file,
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

        #
        # Correct number of rows.
        #

        assert len(rows) == NUM_ROWS

        #
        # Verify first row.
        #

        first = rows[0]

        assert int(first["left_frame"]) == 0
        assert int(first["right_frame"]) == 0

        assert int(first["left_capture_ms"]) == 0
        assert int(first["right_capture_ms"]) == 2

        assert int(first["capture_delta_ms"]) == 2

        #
        # Verify last row.
        #

        last = rows[-1]

        assert int(last["left_frame"]) == NUM_ROWS - 1
        assert int(last["right_frame"]) == NUM_ROWS - 1

        print("PASS")


if __name__ == "__main__":
    run()