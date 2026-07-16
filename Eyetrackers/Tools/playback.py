from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import argparse

import cv2

LEFT_VIDEO = "left_eye.mp4"
RIGHT_VIDEO = "right_eye.mp4"
SYNC_CSV = "eyetracker.csv"

@dataclass(slots=True)
class PlaybackRow:
    """
    One synchronized frame pair from the CSV log.
    """

    left_frame: int
    right_frame: int

    timestamp_ms: int

    delta_ms: float

    status: str

@dataclass(slots=True)
class PlaybackSession:
    """
    Resources required for synchronized playback.
    """

    patient_directory: Path

    left_capture: cv2.VideoCapture
    right_capture: cv2.VideoCapture

    rows: list[PlaybackRow]

    current_index: int = 0
    frames_presented: int = 0

    paused: bool = False

    playback_speed: float = 1.0

    previous_timestamp_ms: int | None = None

    show_overlay: bool = True

def draw_text(
    image,
    text: str,
    x: int,
    y: int,
    color=(255, 255, 255),
) -> None:
    """
    Draw consistently formatted overlay text.
    """

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

def draw_overlay(
    session: PlaybackSession,
    row: PlaybackRow,
    image,
) -> None:
    """
    Draw the playback diagnostic HUD.
    """

    y = 20
    step = 18

    draw_text(
        image,
        "Playback",
        10,
        y,
    )

    y += step

    draw_text(
        image,
        (
            f"Frame: "
            f"{session.current_index}"
            f"/"
            f"{len(session.rows)}"
        ),
        10,
        y,
    )

    y += step

    draw_text(
        image,
        (
            f"Timestamp: "
            f"{row.timestamp_ms} ms"
        ),
        10,
        y,
    )

    y += step

    draw_text(
        image,
        (
            f"Capture Δ: "
            f"{row.delta_ms:.1f} ms"
        ),
        10,
        y,
    )

    y += step

    status_color = (
        (0, 255, 0)
        if row.status == "SYNCED"
        else (0, 0, 255)
    )

    draw_text(
        image,
        (
            f"Status: "
            f"{row.status}"
        ),
        10,
        y,
        status_color,
    )

    #
    # Right side
    #

    x = image.shape[1] - 180

    draw_text(
        image,
        (
            f"Speed: "
            f"{session.playback_speed:.2f}x"
        ),
        x,
        20,
    )

    #
    # Bottom help
    #

    help_text = (
        "Space Pause | H HUD | +/- Speed | Q Quit"
    )

    draw_text(
        image,
        help_text,
        10,
        image.shape[0] - 10,
        (180, 180, 180),
    )

    #
    # Pause banner
    #

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

def load_rows(csv_path: Path) -> list[PlaybackRow]:
    """
    Load synchronization metadata from CSV.
    """

    rows: list[PlaybackRow] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = (
            "LeftFrame",
            "RightFrame",
            "SyncTickMs",
            "CaptureDeltaMs",
            "Status",
        )

        missing = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}"
            )

        for row in reader:

            rows.append(
                PlaybackRow(
                    left_frame=int(row["LeftFrame"]),
                    right_frame=int(row["RightFrame"]),
                    timestamp_ms=int(row["SyncTickMs"]),
                    delta_ms=float(row["CaptureDeltaMs"]),
                    status=row["Status"],
                )
            )

    return rows

def load_session(
    patient_directory: Path,
) -> PlaybackSession:
    """
    Load an existing patient recording.
    """

    patient_directory = Path(patient_directory)

    if not patient_directory.exists():
        raise FileNotFoundError(patient_directory)

    left_path = patient_directory / LEFT_VIDEO
    right_path = patient_directory / RIGHT_VIDEO
    csv_path = patient_directory / SYNC_CSV

    for path in (
        left_path,
        right_path,
        csv_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    left_capture = cv2.VideoCapture(str(left_path))
    right_capture = cv2.VideoCapture(str(right_path))

    if not left_capture.isOpened():
        raise RuntimeError(
            f"Unable to open {left_path}"
        )

    if not right_capture.isOpened():
        raise RuntimeError(
            f"Unable to open {right_path}"
        )

    rows = load_rows(csv_path)

    left_frames = int(
        left_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    right_frames = int(
        right_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nPlayback Session")

    print(
        f"CSV rows:      {len(rows)}"
    )

    print(
        f"Left frames:   {left_frames}"
    )

    print(
        f"Right frames:  {right_frames}"
    )

    if left_frames != len(rows):
        print(
            "WARNING: Left video frame count "
            "does not match CSV."
        )

    if right_frames != len(rows):
        print(
            "WARNING: Right video frame count "
            "does not match CSV."
        )

    return PlaybackSession(
        patient_directory=patient_directory,
        left_capture=left_capture,
        right_capture=right_capture,
        rows=rows,
    )

def read_next_pair(
    session: PlaybackSession,
) -> tuple[
    PlaybackRow,
    np.ndarray,
    np.ndarray,
] | None:
    """
    Read the next synchronized stereo frame pair.

    Returns
    -------
    PlaybackRow, left_frame, right_frame

    or

    None when playback has reached the end.
    """

    if session.current_index >= len(session.rows):
        return None

    success_left, left = session.left_capture.read()
    success_right, right = session.right_capture.read()

    if not success_left:
        raise RuntimeError(
            "Unexpected end of left video."
        )

    if not success_right:
        raise RuntimeError(
            "Unexpected end of right video."
        )

    row = session.rows[session.current_index]

    session.current_index += 1

    return row, left, right

def close_session(
    session: PlaybackSession,
) -> None:
    """
    Release all playback resources.
    """

    session.left_capture.release()
    session.right_capture.release()

    cv2.destroyAllWindows()

def build_display_frame(
    session: PlaybackSession,
    row: PlaybackRow,
    left,
    right,
):
    combined = cv2.hconcat(
        [
            left,
            right,
        ]
    )

    combined = combined.copy()

    if session.show_overlay:
        draw_overlay(
            session,
            row,
            combined,
        )

    return combined

def present_frame(
    image,
    delay_ms: int,
) -> int:
    cv2.imshow(
        "Playback",
        image,
    )

    return cv2.waitKey(delay_ms)

def playback(
    session: PlaybackSession,
):
    while True:

        result = read_next_pair(session)

        if result is None:
            break

        row, left, right = result

        image = build_display_frame(
            session,
            row,
            left,
            right,
        )

        delay_ms = playback_delay(
            session,
            row,
        )

        key = present_frame(
            image,
            delay_ms,
        )

        if key == ord(" "):
            session.paused = not session.paused
        elif key == ord("+"):

            session.playback_speed *= 2

        elif key == ord("-"):

            session.playback_speed /= 2

        
        elif key in (
            ord("h"),
            ord("H"),
        ):
            session.show_overlay = (
                not session.show_overlay
            )
        
        session.playback_speed = min(
            max(
                session.playback_speed,
                0.25,
            ),
            8.0,
        )


        while session.paused:
            key = cv2.waitKey(30)

            if key == ord(" "):
                session.paused = False

            elif key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                return

        session.frames_presented += 1

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            break

def playback_delay(
    session: PlaybackSession,
    row: PlaybackRow,
) -> int:
    if session.previous_timestamp_ms is None:

        session.previous_timestamp_ms = row.timestamp_ms

        return 1

    delay = (
        row.timestamp_ms
        - session.previous_timestamp_ms
    )

    session.previous_timestamp_ms = row.timestamp_ms

    delay = max(delay, 1)

    delay = max(
        1,
        round(
            delay / session.playback_speed
        ),
    )

    return delay

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "patient_directory",
        type=Path,
    )

    args = parser.parse_args()

    session = load_session(
        args.patient_directory
    )

    try:
        playback(session)
    finally:
        close_session(session)