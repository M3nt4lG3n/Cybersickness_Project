"""
CSV output for synchronized stereo frames.
"""


import csv



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


        self.writer.writerow(
            [
                "sync_time_ms",

                "left_frame",
                "left_capture_ms",
                "left_receive_ms",

                "right_frame",
                "right_capture_ms",
                "right_receive_ms",

                "capture_delta_ms",

                "left_latency_ms",
                "right_latency_ms"
            ]
        )



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


                left.latency_ms,


                right.latency_ms

            ]
        )


        self.file.flush()



    def close(self):

        self.file.close()
