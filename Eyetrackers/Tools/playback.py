from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import csv
import cv2
import numpy as np


# ==========================================================
# Filenames
# ==========================================================

LEFT_VIDEO = "left_eye.mp4"
RIGHT_VIDEO = "right_eye.mp4"

LEFT_CSV = "left_eye.csv"
RIGHT_CSV = "right_eye.csv"


# ==========================================================
# Playback Metadata
# ==========================================================

@dataclass(slots=True)
class PlaybackFrame:
    """
    Metadata describing one recorded frame.
    """

    frame_number: int

    capture_timestamp_ms: int

    receive_timestamp_ms: int


# ==========================================================
# Playback Session
# ==========================================================

@dataclass(slots=True)
class PlaybackSession:

    patient_directory: Path

    left_capture: cv2.VideoCapture
    right_capture: cv2.VideoCapture

    left_rows: list[PlaybackFrame]
    right_rows: list[PlaybackFrame]

    left_index: int = 0
    right_index: int = 0

    left_image: np.ndarray | None = None
    right_image: np.ndarray | None = None

    left_metadata: PlaybackFrame | None = None
    right_metadata: PlaybackFrame | None = None

    playback_clock_ms: int = 0

    previous_capture_time: int | None = None

    paused: bool = False

    playback_speed: float = 1.0

    show_overlay: bool = True

    frames_presented: int = 0


# ==========================================================
# Drawing Helpers
# ==========================================================

def draw_text(
    image,
    text: str,
    x: int,
    y: int,
    color=(255, 255, 255),
) -> None:

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


# ==========================================================
# CSV Loading
# ==========================================================

def load_csv(
    csv_path: Path,
) -> list[PlaybackFrame]:
    """
    Load a per-camera recording CSV.
    """

    rows: list[PlaybackFrame] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        required = (
            "FrameNumber",
            "CaptureTimestampMs",
            "ReceiveTimestampMs",
        )

        missing = [
            c
            for c in required
            if c not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                f"{csv_path.name} missing columns: {missing}"
            )

        for row in reader:

            rows.append(

                PlaybackFrame(

                    frame_number=int(
                        row["FrameNumber"]
                    ),

                    capture_timestamp_ms=int(
                        row["CaptureTimestampMs"]
                    ),

                    receive_timestamp_ms=int(
                        row["ReceiveTimestampMs"]
                    ),
                )

            )

    return rows


# ==========================================================
# Session Loading
# ==========================================================

def load_session(
    patient_directory: Path,
) -> PlaybackSession:

    patient_directory = Path(
        patient_directory
    )

    if not patient_directory.exists():
        raise FileNotFoundError(
            patient_directory
        )

    left_video = patient_directory / LEFT_VIDEO
    right_video = patient_directory / RIGHT_VIDEO

    left_csv = patient_directory / LEFT_CSV
    right_csv = patient_directory / RIGHT_CSV

    required_files = (

        left_video,
        right_video,

        left_csv,
        right_csv,

    )

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    left_capture = cv2.VideoCapture(
        str(left_video)
    )

    right_capture = cv2.VideoCapture(
        str(right_video)
    )

    if not left_capture.isOpened():

        raise RuntimeError(
            f"Unable to open {left_video}"
        )

    if not right_capture.isOpened():

        raise RuntimeError(
            f"Unable to open {right_video}"
        )

    left_rows = load_csv(
        left_csv
    )

    right_rows = load_csv(
        right_csv
    )

    print("\nPlayback Session")

    print(
        f"Left frames : "
        f"{len(left_rows)}"
    )

    print(
        f"Right frames: "
        f"{len(right_rows)}"
    )

    print(
        f"Left video : "
        f"{int(left_capture.get(cv2.CAP_PROP_FRAME_COUNT))}"
    )

    print(
        f"Right video: "
        f"{int(right_capture.get(cv2.CAP_PROP_FRAME_COUNT))}"
    )

    return PlaybackSession(

        patient_directory=patient_directory,

        left_capture=left_capture,
        right_capture=right_capture,

        left_rows=left_rows,
        right_rows=right_rows,

    )

# ==========================================================
# Frame Advancement
# ==========================================================

def advance_left(
    session: PlaybackSession,
) -> None:
    """
    Advance the left video until its capture timestamp
    exceeds the playback clock.
    """

    while session.left_index < len(session.left_rows):

        row = session.left_rows[session.left_index]

        if (
            row.capture_timestamp_ms
            > session.playback_clock_ms
        ):
            break

        success, image = session.left_capture.read()

        if not success:
            break

        session.left_image = image
        session.left_metadata = row

        session.left_index += 1


def advance_right(
    session: PlaybackSession,
) -> None:
    """
    Advance the right video until its capture timestamp
    exceeds the playback clock.
    """

    while session.right_index < len(session.right_rows):

        row = session.right_rows[session.right_index]

        if (
            row.capture_timestamp_ms
            > session.playback_clock_ms
        ):
            break

        success, image = session.right_capture.read()

        if not success:
            break

        session.right_image = image
        session.right_metadata = row

        session.right_index += 1


# ==========================================================
# Overlay
# ==========================================================

def draw_overlay(
    session: PlaybackSession,
    image: np.ndarray,
) -> None:

    y = 20
    step = 18

    draw_text(
        image,
        f"Playback Clock : {session.playback_clock_ms} ms",
        10,
        y,
    )

    y += step

    draw_text(
        image,
        f"Speed : {session.playback_speed:.2f}x",
        10,
        y,
    )

    y += step

    draw_text(
        image,
        f"Presented : {session.frames_presented}",
        10,
        y,
    )

    #
    # Left camera
    #

    if session.left_metadata is not None:

        y += step * 2

        draw_text(
            image,
            "LEFT",
            10,
            y,
            (0, 255, 0),
        )

        y += step

        draw_text(
            image,
            f"Frame : {session.left_metadata.frame_number}",
            10,
            y,
        )

        y += step

        draw_text(
            image,
            (
                "Capture : "
                f"{session.left_metadata.capture_timestamp_ms}"
            ),
            10,
            y,
        )

        y += step

        draw_text(
            image,
            (
                "Receive : "
                f"{session.left_metadata.receive_timestamp_ms}"
            ),
            10,
            y,
        )

    #
    # Right camera
    #

    if session.right_metadata is not None:

        x = image.shape[1] // 2 + 20
        y = 56

        draw_text(
            image,
            "RIGHT",
            x,
            y,
            (0, 255, 0),
        )

        y += step

        draw_text(
            image,
            f"Frame : {session.right_metadata.frame_number}",
            x,
            y,
        )

        y += step

        draw_text(
            image,
            (
                "Capture : "
                f"{session.right_metadata.capture_timestamp_ms}"
            ),
            x,
            y,
        )

        y += step

        draw_text(
            image,
            (
                "Receive : "
                f"{session.right_metadata.receive_timestamp_ms}"
            ),
            x,
            y,
        )

    #
    # Bottom help
    #

    draw_text(
        image,
        "Space Pause | +/- Speed | H HUD | Q Quit",
        10,
        image.shape[0] - 10,
        (180, 180, 180),
    )

    if session.paused:

        cv2.putText(
            image,
            "PAUSED",
            (
                image.shape[1] // 2 - 70,
                image.shape[0] // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ==========================================================
# Frame Composition
# ==========================================================

def build_display_frame(
    session: PlaybackSession,
):

    if (
        session.left_image is None
        and
        session.right_image is None
    ):
        return None

    if session.left_image is None:

        left = np.zeros_like(
            session.right_image
        )

    else:

        left = session.left_image.copy()

    if session.right_image is None:

        right = np.zeros_like(
            session.left_image
        )

    else:

        right = session.right_image.copy()

    combined = cv2.hconcat(
        [
            left,
            right,
        ]
    )

    if session.show_overlay:

        draw_overlay(
            session,
            combined,
        )

    return combined


# ==========================================================
# Playback Timing
# ==========================================================

def update_clock(
    session: PlaybackSession,
):

    next_times = []

    if session.left_index < len(session.left_rows):

        next_times.append(
            session.left_rows[
                session.left_index
            ].capture_timestamp_ms
        )

    if session.right_index < len(session.right_rows):

        next_times.append(
            session.right_rows[
                session.right_index
            ].capture_timestamp_ms
        )

    if not next_times:

        return False

    next_time = min(next_times)

    if session.previous_capture_time is None:

        delta = 1

    else:

        delta = max(
            1,
            next_time
            - session.previous_capture_time
        )

    session.previous_capture_time = next_time

    wait = max(
        1,
        round(
            delta
            / session.playback_speed
        ),
    )

    key = cv2.waitKey(wait)

    session.playback_clock_ms = next_time

    return key

# ==========================================================
# Playback
# ==========================================================

def playback(
    session: PlaybackSession,
) -> None:
    """
    Play both recordings using their capture timestamps.

    Each camera advances independently according to its own
    CSV timestamps while sharing a common playback clock.
    """

    cv2.namedWindow(
        "Playback",
        cv2.WINDOW_NORMAL,
    )

    while True:

        #
        # End once both recordings have been exhausted.
        #

        if (
            session.left_index >= len(session.left_rows)
            and
            session.right_index >= len(session.right_rows)
        ):
            break

        #
        # Advance whichever cameras should update at the
        # current playback clock.
        #

        advance_left(session)
        advance_right(session)

        image = build_display_frame(session)

        if image is not None:

            cv2.imshow(
                "Playback",
                image,
            )

            session.frames_presented += 1

        key = update_clock(session)

        #
        # Quit
        #

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            break

        #
        # Pause
        #

        elif key == ord(" "):

            session.paused = not session.paused

            while session.paused:

                image = build_display_frame(session)

                if image is not None:

                    cv2.imshow(
                        "Playback",
                        image,
                    )

                pause_key = cv2.waitKey(30)

                if pause_key == ord(" "):
                    session.paused = False

                elif pause_key in (
                    27,
                    ord("q"),
                    ord("Q"),
                ):
                    return

                elif pause_key in (
                    ord("h"),
                    ord("H"),
                ):
                    session.show_overlay = (
                        not session.show_overlay
                    )

        #
        # Speed controls
        #

        elif key in (
            ord("+"),
            ord("="),
        ):

            session.playback_speed *= 2.0

        elif key in (
            ord("-"),
            ord("_"),
        ):

            session.playback_speed /= 2.0

        #
        # Toggle overlay
        #

        elif key in (
            ord("h"),
            ord("H"),
        ):

            session.show_overlay = (
                not session.show_overlay
            )

        #
        # Clamp playback speed
        #

        session.playback_speed = max(
            0.25,
            min(
                session.playback_speed,
                8.0,
            ),
        )


# ==========================================================
# Cleanup
# ==========================================================

def close_session(
    session: PlaybackSession,
) -> None:

    session.left_capture.release()
    session.right_capture.release()

    cv2.destroyAllWindows()


# ==========================================================
# Entry Point
# ==========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Playback an Eyetrackers recording."
        )
    )

    parser.add_argument(
        "patient_directory",
        type=Path,
        help="Patient recording directory",
    )

    args = parser.parse_args()

    session = load_session(
        args.patient_directory,
    )

    try:

        playback(session)

    finally:

        close_session(session)


if __name__ == "__main__":
    main()