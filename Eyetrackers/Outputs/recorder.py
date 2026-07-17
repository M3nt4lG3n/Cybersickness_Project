import cv2
import os
import time

import numpy as np

from Eyetrackers.Core.tracker_types import FramePacket

class Recorder:

    def __init__(
        self,
        filename,
        fps,
        frame_size
    ):

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = self.VideoWriter = cv2.VideoWriter(
            filename,
            fourcc,
            fps,
            (frame_size)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Unable to open video writer: {filename}"
            )

        self.frames = 0



    def write(
        self,
        frame: FramePacket,
    ) -> None:

        self.writer.write(frame.image)

        self.frames += 1



    def close(self) -> None:
        self.writer.release()

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()


    @property
    def queue_size(self) -> int:
        if self.mode == BufferMode.FIFO:
            return self.queue.qsize()

        return 0
