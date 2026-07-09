"""
Timestamp indexed frame buffer.
"""

from collections import deque


class FrameBuffer:

    def __init__(
        self,
        max_size=120
    ):

        self.frames = deque(
            maxlen=max_size
        )


    def add(self, frame):

        self.frames.append(frame)


    def closest(
        self,
        timestamp_ms,
        tolerance_ms
    ):

        if not self.frames:
            return None


        best = min(
            self.frames,
            key=lambda f:
                abs(
                    f.metadata.unix_ms -
                    timestamp_ms
                )
        )


        delta = abs(
            best.metadata.unix_ms -
            timestamp_ms
        )


        if delta <= tolerance_ms:
            return best


        return None


    def remove(self, frame):

        try:
            self.frames.remove(frame)

        except ValueError:
            pass


    def clear(self):

        self.frames.clear()


    def __len__(self):

        return len(self.frames)
