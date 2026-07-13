"""
Stereo frame synchronization.

Matches independent ESP32 cameras using
capture timestamps.
"""

from typing import Optional
from dataclasses import dataclass

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Core.tracker_types import SyncPair

@dataclass
class SyncStatistics:

    total_pairs: int = 0
    failed_matches: int = 0
    average_delta_ms: float = 0.0



class StereoSynchronizer:


    def __init__(
        self,
        left_camera: Camera,
        right_camera: Camera,
        tolerance_ms: int = 25,
    ):

        self.left_camera = left_camera
        self.right_camera = right_camera

        self.tolerance_ms = tolerance_ms

        self.stats = SyncStatistics()

    def __repr__(self):

        return (

            f"StereoSynchronizer("

            f"pairs={self.stats.total_pairs}, "

            f"failed={self.stats.failed_matches}, "

            f"success={self.success_rate:.1%}"

            f")"

        )

    @property
    def success_rate(self) -> float:

        total = (
            self.stats.total_pairs +
            self.stats.failed_matches
        )

        if total == 0:
            return 0.0

        return self.stats.total_pairs / total



    def get_pair(self) -> Optional[SyncPair]:

        left = self.left_camera.oldest_frame()

        if left is None:
            return None

        left_time = left.capture_ms


        right = self.right_camera.consume_closest(
            left_time,
            self.tolerance_ms
        )


        if right is None:

            self._record_failure()

            self.left_camera.pop_oldest()

            return None



        delta = abs(
            left.capture_ms -
            right.capture_ms
        )



        self.left_camera.pop_oldest()

        self._record_success(delta)



        return SyncPair(
            left=left,
            right=right,
            sync_timestamp_ms = min(
                left.capture_ms,
                right.capture_ms
            ),
            delta_ms=delta,
        )
    
    def _record_success(
        self,
        delta_ms: int,
    ) -> None:

        self.stats.total_pairs += 1

        self.stats.average_delta_ms = (

            (
                self.stats.average_delta_ms
                *
                (self.stats.total_pairs - 1)
            )

            +

            delta_ms

        ) / self.stats.total_pairs


    def _record_failure(self) -> None:
        self.stats.failed_matches += 1


    def reset_statistics(self) -> None:
        self.stats = SyncStatistics()
