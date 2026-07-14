import cv2
import numpy as np

from Eyetrackers.Core import config
from Eyetrackers.Core.tracker_types import SyncPair
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


    def show(self, pair: SyncPair):
        left = pair.left.image
        right = pair.right.image

        left_text = (
            f"L Frame:{pair.left.frame_number} "
            f"Time:{pair.left.capture_ms}"
        )

        right_text = (
            f"R Frame:{pair.right.frame_number} "
            f"Time:{pair.right.capture_ms}"
        )

        delta_text = (
            f"Δ {pair.delta_ms:.1f} ms"
        )

        latency_text = (
            f"Lag L:{pair.left.latency_ms} "
            f"R:{pair.right.latency_ms}"
        )

        title = (
            f"Stereo View | Δ={pair.delta_ms:.1f} ms"
        )


        combined = cv2.hconcat(
            [
                left,
                right
            ]
        )

        combined = combined.copy()


        cv2.putText(
            combined,
            left_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            combined,
            right_text,
            (left.shape[1] + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            combined,
            delta_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            combined,
            latency_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )


        cv2.setWindowTitle(self.window_name, title)
        cv2.imshow(
            self.window_name,
            combined,
        )

        key = cv2.waitKey(1)

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            self._should_close = True

        self._frames_displayed += 1


    def close(self):

        self._should_close = True

        cv2.destroyWindow(
            self.window_name
        )

        cv2.waitKey(1)
