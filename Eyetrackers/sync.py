"""
Stereo synchronization using ESP32 capture timestamps.

The camera acquisition threads run independently.
This module selects matching frames after acquisition.
"""

from dataclasses import dataclass
from typing import Optional

from tracker_types import FramePacket, SyncPair


@dataclass
class SyncStatistics:
    pairs_created: int = 0
    failed_matches: int = 0
    average_delta_ms: float = 0.0


class StereoSynchronizer:

    def __init__(
        self,
        left_camera,
        right_camera,
        tolerance_ms=25
    ):

        self.left = left_camera
        self.right = right_camera

        self.tolerance_ms = tolerance_ms

        self.stats = SyncStatistics()


    def get_pair(self) -> Optional[SyncPair]:

        left_frame = self.left.oldest_frame()

        if left_frame is None:
            return None


        # Match using ESP32 clock domain
        right_frame = self.right.closest_to(
            left_frame.receive_ms
        )


        if right_frame is None:

            self.stats.failed_matches += 1

            # discard frames that are too old
            self.left.pop_oldest()

            return None



        delta = abs(
            left_frame.capture_ms -
            right_frame.capture_ms
        )


        if delta > self.tolerance_ms:

            self.stats.failed_matches += 1

            self.left.pop_oldest()

            return None



        self.left.pop_oldest()


        self.stats.pairs_created += 1


        self.stats.average_delta_ms = (
            (
                self.stats.average_delta_ms *
                (self.stats.pairs_created - 1)
            )
            +
            delta
        ) / self.stats.pairs_created



        return SyncPair(
            left=left_frame,
            right=right_frame,
            timestamp_ms=min(
                left_frame.capture_ms,
                right_frame.capture_ms
            )
        )
