"""
Stereo frame synchronization.

Matches independent ESP32 cameras using
capture timestamps.
"""

from dataclasses import dataclass
from tracker_types import SyncPair

@dataclass
class SyncStatistics:

    total_pairs: int = 0
    failed_matches: int = 0
    average_delta_ms: float = 0.0



class StereoSynchronizer:


    def __init__(
        self,
        left_camera,
        right_camera,
        tolerance_ms: int = 25
    ):

        self.left_camera = left_camera
        self.right_camera = right_camera

        self.tolerance_ms = tolerance_ms

        self.stats = SyncStatistics()

    @property
    def success_rate(self):
        return (self.stats.total_pairs / (
        self.stats.total_pairs +
        self.stats.failed_matches
        ))



    def get_pair(self):

        left = self.left_camera.oldest_frame()


        if left is None:
            return None



        left_time = left.capture_ms


        right = self.right_camera.buffer.closest(
            left_time
        )


        if right is None:

            self.stats.failed_matches += 1

            self.left_camera.pop_oldest()

            return None



        delta = abs(
            left.capture_ms -
            right.capture_ms
        )


        if delta > self.tolerance_ms:

            self.stats.failed_matches += 1

            self.left_camera.pop_oldest()

            return None



        self.left_camera.pop_oldest()


        self.stats.total_pairs += 1


        self.stats.average_delta_ms = (

            (
                self.stats.average_delta_ms *
                (self.stats.total_pairs - 1)
            )
            +
            delta

        ) / self.stats.total_pairs



        return SyncPair(
            left=left,
            right=right,
            sync_timestamp_ms = min(
                left.capture_ms,
                right.capture_ms
            ),
            delta_ms=delta,
        )
