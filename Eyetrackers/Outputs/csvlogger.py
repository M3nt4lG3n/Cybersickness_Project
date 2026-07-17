"""
CSV output for synchronized stereo frames.
"""


import csv
from Eyetrackers.Core import config

from Eyetrackers.Core.tracker_types import FramePacket


class CSVLogger:


    def __init__(self, filename):

        self.file = open(
            filename,
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )


        self.writer.writerow(config.CSV_HEADER)



    def write(self, frame: FramePacket):
        self.writer.writerow(
            [
                frame.frame_number,
                frame.capture_ms,
                frame.receive_ms,
                frame.latency_ms,
            ]
        )

        self.file.flush()



    def close(self):

        self.file.close()
