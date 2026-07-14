"""
CSV output for synchronized stereo frames.
"""


import csv
from Eyetrackers.Core import config



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



    def write(self, pair):

        left = pair.left
        right = pair.right


        self.writer.writerow(
            [
                pair.sync_timestamp_ms,

                left.frame_number,
                left.capture_ms,
                left.receive_ms,

                right.frame_number,
                right.capture_ms,
                right.receive_ms,

                pair.delta_ms,
                pair.receive_delta_ms,

                left.latency_ms,
                right.latency_ms,

                pair.status,
            ]
        )


        self.file.flush()



    def close(self):

        self.file.close()
