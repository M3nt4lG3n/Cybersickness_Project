import cv2


class StereoDisplay:


    def __init__(self):

        cv2.namedWindow(
            "Stereo",
            cv2.WINDOW_NORMAL
        )


    def show(
        self,
        pair
    ):


        left = pair.left.image
        right = pair.right.image


        combined = cv2.hconcat(
            [
                left,
                right
            ]
        )


        cv2.putText(
            combined,
            f"Delta: {abs(pair.left.capture_ms - pair.right.capture_ms)} ms",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        cv2.imshow(
            "Stereo",
            combined
        )


        return cv2.waitKey(1)



    def close(self):

        cv2.destroyAllWindows()
