import cv2
import numpy as np

from Eyetrackers.Core.tracker_types import SyncPair
class Display:


    def __init__(self):

        cv2.namedWindow(
            "Stereo",
            cv2.WINDOW_NORMAL
        )


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


        cv2.setWindowTitle("Stereo", title)
        cv2.imshow("Stereo", combined)


        return cv2.waitKey(1)



    def close(self):

        cv2.destroyAllWindows()
