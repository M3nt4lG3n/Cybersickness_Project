"""
Timestamp based frame buffer.

Stores FramePacket objects from ESP32 cameras.
Frames are matched using ESP32 capture timestamps,
not PC arrival time.
"""

from collections import deque
import threading


class FrameBuffer:

    def __init__(self, max_size=120):

        self.frames = deque(maxlen=max_size)
        self.lock = threading.Lock()


    def add(self, frame):

        with self.lock:
            self.frames.append(frame)


    def oldest(self):

        with self.lock:

            if len(self.frames) == 0:
                return None

            return self.frames[0]


    def remove_oldest(self):

        with self.lock:

            if len(self.frames) == 0:
                return None

            return self.frames.popleft()


    def closest(self, timestamp_ms):

        with self.lock:

            if len(self.frames) == 0:
                return None


            return min(
                self.frames,
                key=lambda frame:
                    abs(
                        frame.metadata.unix_ms -
                        timestamp_ms
                    )
            )


    def clear(self):

        with self.lock:
            self.frames.clear()


    def size(self):

        with self.lock:
            return len(self.frames)
