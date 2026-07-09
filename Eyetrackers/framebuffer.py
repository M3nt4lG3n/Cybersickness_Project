"""
Generic timestamp frame buffer.
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


    def add(
        self,
        frame
    ):

        self.frames.append(frame)



    def closest(
        self,
        timestamp_ms
    ):

        if not self.frames:
            return None


        return min(
            self.frames,
            key=lambda frame:
            abs(
                frame.capture_ms -
                timestamp_ms
            )
        )



    def remove_oldest(self):

        if self.frames:
            return self.frames.popleft()

        return None



    def clear(self):

        self.frames.clear()



    def __len__(self):

        return len(self.frames)
