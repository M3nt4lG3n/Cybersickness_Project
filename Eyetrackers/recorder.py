import cv2
import os
import time

import numpy as np

class StereoRecorder:

    def __init__(
        self,
        left_filename,
        right_filename,
        fps,
        frame_size
    ):


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        self.left_writer = cv2.VideoWriter(
            left_filename,
            fourcc,
            fps,
            (frame_size)
        )


        self.right_writer = cv2.VideoWriter(
            right_filename,
            fourcc,
            fps,
            (frame_size)
        )


        self.frames = 0



    def write(self, pair):

        left = pair.left
        right = pair.right

        self.left_writer.write(left.image)
        self.right_writer.write(right.image)

        self.frames += 1



    def close(self):
        self.left_writer.release()
        self.right_writer.release()
