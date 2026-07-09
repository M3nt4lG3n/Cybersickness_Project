import cv2
import os
import time


class StereoRecorder:


    def __init__(
        self,
        left_file,
        right_file,
        fps,
        width,
        height
    ):


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        self.left = cv2.VideoWriter(
            left_file,
            fourcc,
            fps,
            (width,height)
        )


        self.right = cv2.VideoWriter(
            right_file,
            fourcc,
            fps,
            (width,height)
        )


        self.frames = 0



    def write(
        self,
        pair
    ):


        self.left.write(
            pair.left.image
        )


        self.right.write(
            pair.right.image
        )


        self.frames += 1



    def close(self):

        self.left.release()

        self.right.release()
