"""
CSV logger for synchronized eye camera frames.
"""

import csv
import time


class CSVLogger:


    HEADER = [
        "sync_time_ms",

        "left_frame",
        "left_capture_ms",
        "left_receive_ms",

        "right_frame",
        "right_capture_ms",
        "right_receive_ms",

        "capture_delta_ms",
        "receive_delta_ms",

        "left_latency_ms",
        "right_latency_ms"
    ]


    def __init__(self, filename):

        self.file = open(
            filename,
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow(
            self.HEADER
        )


    def log_pair(
        self,
        pair
    ):

        left = pair.left
        right = pair.right


        self.writer.writerow([

            pair.timestamp_ms,


            left.metadata.frame_number,
            left.metadata.unix_ms,
            left.receive_time_ms,


            right.metadata.frame_number,
            right.metadata.unix_ms,
            right.receive_time_ms,


            abs(
                left.metadata.unix_ms -
                right.metadata.unix_ms
            ),


            abs(
                left.receive_time_ms -
                right.receive_time_ms
            ),


            left.receive_time_ms -
            left.metadata.unix_ms,


            right.receive_time_ms -
            right.metadata.unix_ms
        ])


    def close(self):

        self.file.close()
