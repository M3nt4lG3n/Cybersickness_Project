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
        tolerance_ms=25
    ):

        self.left_camera = left_camera
        self.right_camera = right_camera

        self.tolerance_ms = tolerance_ms

        self.stats = SyncStatistics()



    def get_pair(self):

        left = self.left_camera.oldest_frame()


        if left is None:
            return None



        left_time = (
            left.metadata.unix_ms
        )


        right = (
            self.right_camera.closest_to(
                left_time
            )
        )


        if right is None:

            self.stats.failed_matches += 1

            self.left_camera.pop_oldest()

            return None



        delta = abs(
            left.metadata.unix_ms -
            right.metadata.unix_ms
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
            sync_timestamp_ms=left_time,
            delta_ms=delta,
        )
