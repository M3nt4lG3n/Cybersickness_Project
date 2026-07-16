"""
Timestamp based frame buffer.

Stores FramePacket objects from ESP32 cameras.
Frames are matched using ESP32 capture timestamps,
not PC arrival time.
"""

from collections import deque
import threading
from typing import Optional

from Eyetrackers.Core.tracker_types import FramePacket


class FrameBuffer:

    def __init__(self, max_size=120):

        self.frames: deque[FramePacket] = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.max_match_index = 0


    def add(self, frame: FramePacket) -> None:

        with self.lock:
            self.frames.append(frame)


    def oldest(self) -> Optional[FramePacket]:

        with self.lock:

            if len(self.frames) == 0:
                return None

            return self.frames[0]


    def remove_oldest(self) -> Optional[FramePacket]:

        with self.lock:

            if len(self.frames) == 0:
                return None

            return self.frames.popleft()


    def closest(
        self,
        timestamp_ms: int,
        tolerance_ms: int,
    ) -> Optional[FramePacket]:

        with self.lock:

            if not self.frames:
                return None

            best = min(
                self.frames,
                key=lambda frame:
                    abs(
                        frame.capture_ms -
                        timestamp_ms
                    )
            )

            if abs(best.capture_ms - timestamp_ms) > tolerance_ms:
                return None

            return best
        
    def consume_closest(
        self,
        timestamp_ms: int,
        tolerance_ms: int,
    ) -> Optional[FramePacket]:

        with self.lock:

            if not self.frames:
                return None

            best_index = min(
                range(len(self.frames)),
                key=lambda i:
                    abs(
                        self.frames[i].capture_ms -
                        timestamp_ms
                    )
            )

            best = self.frames[best_index]

            if abs(best.capture_ms - timestamp_ms) > tolerance_ms:
                return None

            print(
                f"Matched index {best_index}"
            )

            self.max_match_index = max(
                self.max_match_index,
                best_index,
            )

            if best_index > 0:

                stale = best.capture_ms - self.frames[0].capture_ms

                print(
                    f"Match index={best_index} "
                    f"stale span={stale} ms"
                )

            best = self.frames[best_index]

            #
            # Remove every frame older than the matched frame.
            #
            for _ in range(best_index + 1):
                self.frames.popleft()

            return best

            


    def clear(self) -> None:

        with self.lock:
            self.frames.clear()


    def size(self) -> int:

        with self.lock:
            return len(self.frames)
        
    def snapshot(self) -> list[FramePacket]:
        with self.lock:
            return list(self.frames)
        
    @property
    def capacity(self) -> int:
        return self.frames.maxlen
    
    def age_span_ms(self):

        with self.lock:

            if len(self.frames) < 2:
                return 0

            return (
                self.frames[-1].capture_ms
                -
                self.frames[0].capture_ms
            )
        
    def remove_before(
        self,
        timestamp_ms: int,
    ) -> int:

        removed = 0

        with self.lock:

            while (
                self.frames
                and
                self.frames[0].capture_ms < timestamp_ms
            ):
                self.frames.popleft()
                removed += 1

        return removed
