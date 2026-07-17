import cv2
import numpy as np
import threading

from Eyetrackers.Core.tracker_types import FramePacket
class Display:

    def __init__(self):

        self.window_name = "Eyetrackers"

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        #
        # Window state
        #
        self._should_close = False
        self._frames_displayed = 0

        #
        # Latest rendered frame from the worker.
        #
        self._frame_lock = threading.Lock()
        self._left_frame = None
        self._right_frame = None

        self._left_metadata = None
        self._right_metadata = None

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def should_close(self) -> bool:
        """
        True once the user has requested the display close.
        """
        return self._should_close


    @property
    def frames_displayed(self) -> int:
        """
        Number of synchronized frame pairs displayed.
        """
        return self._frames_displayed


    def render_left(self, frame: FramePacket) -> None:

        with self._frame_lock:
            self._left_frame = frame.image.copy()
            self._left_metadata = frame

            self._left_frame = frame.image.copy()
            self._left_metadata = frame



    def render_right(self, frame: FramePacket) -> None:

        with self._frame_lock:
            self._right_frame = frame.image.copy()
            self._right_metadata = frame

            self._right_frame = frame.image.copy()
            self._right_metadata = frame

    def present(self) -> None:
        """
        Present the newest rendered frame.

        Must be called from the main thread.
        """

        with self._frame_lock:
            left = None if self._left_frame is None else self._left_frame.copy()
            right = None if self._right_frame is None else self._right_frame.copy()

            left_meta = self._left_metadata
            right_meta = self._right_metadata

        if left is None and right is None:
            return

        if left is None:
            left = np.zeros_like(right)

        if right is None:
            right = np.zeros_like(left)

        combined = cv2.hconcat(
            [
                left,
                right,
            ]
        )

        cv2.setWindowTitle(
            self.window_name,
            "Eyetrackers",
        )

        cv2.imshow(
            self.window_name,
            combined,
        )

        self._frames_displayed += 1

        key = cv2.waitKey(1)

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            self._should_close = True


    def close(self):

        self._should_close = True

        cv2.destroyWindow(
            self.window_name
        )

        cv2.waitKey(1)
