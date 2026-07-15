from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

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

    return PlaybackSession(
        patient_directory=patient_directory,
        left_capture=left_capture,
        right_capture=right_capture,
        rows=rows,
    )

def read_next_pair(
    session: PlaybackSession,
) -> tuple[PlaybackRow, any, any] | None:
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

    if not success_left or not success_right:
        return None

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

    return combined

def present_frame(
    image,
) -> int:
    """
    Display one playback frame.

    Returns the pressed key.
    """

    cv2.imshow(
        "Playback",
        image,
    )

    return cv2.waitKey(1)

def playback(
    session: PlaybackSession,
):
    while True:

        result = read_next_pair(session)

        if result is None:
            break

        row, left, right = result

        image = build_display_frame(
            row,
            left,
            right,
        )

        key = present_frame(image)

        session.frames_presented += 1

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            break