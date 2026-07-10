import cv2
import numpy as np

from tracker_types import SyncPair
class StereoDisplay:


    def __init__(self):

        cv2.namedWindow(
            "Stereo",
            cv2.WINDOW_NORMAL
        )


    def show(self, pair: SyncPair):
        left = pair.left.image
        right = pair.right.image

        left_meta = pair.left.metadata
        right_meta = pair.right.metadata

        left_text = (
            f"L #{left_meta.frame_number} "
            f"{left_meta.unix_ms}"
        )

        right_text = (
            f"R #{right_meta.frame_number} "
            f"{right_meta.unix_ms}"
        )

        delta_text = (
            f"Δ {pair.delta_ms:.1f} ms"
        )

        latency_text = (
            f"Lag L:{left_meta.latency_ms} "
            f"R:{right_meta.latency_ms}"
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


        cv2.putText(
            combined,
            left_text,
            right_text,
            delta_text,
            latency_text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        cv2.imshow(title, combined)


        return cv2.waitKey(1)



    def close(self):

        cv2.destroyAllWindows()
