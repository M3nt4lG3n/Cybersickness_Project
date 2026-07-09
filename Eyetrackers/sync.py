
"""
Timestamp based stereo frame synchronization.

Uses ESP32 capture timestamps rather than PC arrival time.
"""

from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque

from tracker_types import FramePacket, SyncPair


@dataclass
class SyncConfig:
    max_time_difference_ms: int = 25
    max_buffer_size: int = 120


class FrameSynchronizer:
    def __init__(self, config: SyncConfig = SyncConfig()):
        self.config = config

        self.left_buffer = deque(
            maxlen=config.max_buffer_size
        )

        self.right_buffer = deque(
            maxlen=config.max_buffer_size
        )


    def add_frame(self, frame: FramePacket):

        if frame.camera_id == "left":
            self.left_buffer.append(frame)

        elif frame.camera_id == "right":
            self.right_buffer.append(frame)


    def find_match(
        self,
        source: FramePacket,
        candidates: Deque[FramePacket]
    ) -> Optional[FramePacket]:

        if not candidates:
            return None


        best = min(
            candidates,
            key=lambda f:
                abs(
                    f.metadata.unix_ms -
                    source.metadata.unix_ms
                )
        )


        delta = abs(
            best.metadata.unix_ms -
            source.metadata.unix_ms
        )


        if delta <= self.config.max_time_difference_ms:
            return best


        return None


    def get_pair(self) -> Optional[SyncPair]:

        if not self.left_buffer:
            return None

        if not self.right_buffer:
            return None


        left = self.left_buffer[0]


        right = self.find_match(
            left,
            self.right_buffer
        )


        if right is None:
            return None


        self.left_buffer.remove(left)
        self.right_buffer.remove(right)


        return SyncPair(
            left=left,
            right=right,
            timestamp_ms=
                min(
                    left.metadata.unix_ms,
                    right.metadata.unix_ms
                )
        )
