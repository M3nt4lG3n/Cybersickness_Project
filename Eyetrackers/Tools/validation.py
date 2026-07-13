"""
validation.py

Reusable analysis library for completed Eyetracker recording sessions.

This module performs post-acquisition analysis only.

It never modifies recordings.

validate_session.py is responsible for:
    - Selecting a recording directory
    - Printing reports
    - Saving reports

This module is responsible for:
    - Discovering recording files
    - Computing statistics
    - Returning a structured ValidationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import csv
import statistics


# ============================================================================
# Report Dataclasses
# ============================================================================


@dataclass(slots=True)
class VideoStatistics:
    """
    Statistics for one recorded video.
    """

    filename: str = ""

    exists: bool = False

    width: int = 0
    height: int = 0

    fps: float = 0.0

    frame_count: int = 0

    duration_seconds: float = 0.0

    file_size_bytes: int = 0

    codec: str = ""


@dataclass(slots=True)
class CSVStatistics:
    """
    Statistics extracted from eyetracker.csv.
    """

    filename: str = ""

    exists: bool = False

    rows: int = 0

    duplicate_frames: int = 0

    missing_frames: int = 0

    timestamp_wraps: int = 0


@dataclass(slots=True)
class SynchronizationStatistics:
    """
    Statistics describing synchronization quality.
    """

    pair_count: int = 0

    average_delta_ms: float = 0.0

    median_delta_ms: float = 0.0

    minimum_delta_ms: float = 0.0

    maximum_delta_ms: float = 0.0

    stddev_delta_ms: float = 0.0

    percentile95_ms: float = 0.0

    percentile99_ms: float = 0.0

    average_left_latency_ms: float = 0.0

    average_right_latency_ms: float = 0.0


@dataclass(slots=True)
class ValidationReport:
    """
    Complete validation report returned by SessionAnalyzer.
    """

    session_directory: Path

    left_video: VideoStatistics = field(
        default_factory=VideoStatistics
    )

    right_video: VideoStatistics = field(
        default_factory=VideoStatistics
    )

    csv: CSVStatistics = field(
        default_factory=CSVStatistics
    )

    synchronization: SynchronizationStatistics = field(
        default_factory=SynchronizationStatistics
    )

    warnings: list[str] = field(default_factory=list)

    summary: list[str] = field(default_factory=list)


# ============================================================================
# Session Analyzer
# ============================================================================


class SessionAnalyzer:
    """
    Performs post-acquisition analysis of an Eyetracker recording session.

    Expected directory structure:

        Patient_xxx/
            Eyetrackers/
                left_eye.mp4
                right_eye.mp4
                eyetracker.csv
    """

    def __init__(self, eyetracker_directory: Path):

        self.directory = Path(eyetracker_directory)

        self.left_video_path: Optional[Path] = None

        self.right_video_path: Optional[Path] = None

        self.csv_path: Optional[Path] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def analyze(self) -> ValidationReport:
        """
        Analyze the recording session.
        """

        report = ValidationReport(
            session_directory=self.directory
        )

        self._discover_files(report)

        self._analyze_video(
            self.left_video_path,
            report.left_video,
        )

        self._analyze_video(
            self.right_video_path,
            report.right_video,
        )

        self._analyze_csv(report)

        return report

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    def _discover_files(
        self,
        report: ValidationReport,
    ) -> None:
        """
        Locate all expected recording files.
        """

        self.left_video_path = (
            self.directory / "left_eye.mp4"
        )

        self.right_video_path = (
            self.directory / "right_eye.mp4"
        )

        self.csv_path = (
            self.directory / "eyetracker.csv"
        )

        #
        # Left video
        #

        report.left_video.filename = self.left_video_path.name

        report.left_video.exists = (
            self.left_video_path.exists()
        )

        if report.left_video.exists:
            report.left_video.file_size_bytes = (
                self.left_video_path.stat().st_size
            )

        #
        # Right video
        #

        report.right_video.filename = self.right_video_path.name

        report.right_video.exists = (
            self.right_video_path.exists()
        )

        if report.right_video.exists:
            report.right_video.file_size_bytes = (
                self.right_video_path.stat().st_size
            )

        #
        # CSV
        #

        report.csv.filename = self.csv_path.name

        report.csv.exists = self.csv_path.exists()

        #
        # Missing file warnings
        #

        if not report.left_video.exists:
            report.warnings.append(
                "Missing left_eye.mp4"
            )

        if not report.right_video.exists:
            report.warnings.append(
                "Missing right_eye.mp4"
            )

        if not report.csv.exists:
            report.warnings.append(
                "Missing eyetracker.csv"
            )
    
    # ------------------------------------------------------------------
    # Video Analysis
    # ------------------------------------------------------------------

    def _analyze_video(
        self,
        video_path: Optional[Path],
        stats: VideoStatistics,
    ) -> None:
        """
        Analyze one recorded video.

        Populates the supplied VideoStatistics object.
        """

        if video_path is None:
            return

        if not video_path.exists():
            return

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            return

        try:

            stats.width = int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            stats.height = int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )

            frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            stats.resolution = (
                stats.width,
                stats.height,
            )

            stats.fps = fps

            stats.frame_count = frame_count

            if fps > 0:

                stats.duration_seconds = (
                    frame_count / fps
                )

            stats.codec = self._decode_fourcc(
                int(
                    capture.get(
                        cv2.CAP_PROP_FOURCC
                    )
                )
            )

        finally:

            capture.release()
    
    def _decode_fourcc(
        self,
        value: int,
    ) -> str:
        """
        Decode an OpenCV FOURCC integer into a codec string.
        """

        chars = []

        for shift in (0, 8, 16, 24):

            chars.append(
                chr(
                    (value >> shift) & 0xFF
                )
            )

        return "".join(chars).strip()
    
    # ------------------------------------------------------------------
    # CSV Analysis
    # ------------------------------------------------------------------

    def _analyze_csv(
        self,
        report: ValidationReport,
    ) -> None:
        """
        Analyze eyetracker.csv.
        """

        if self.csv_path is None:
            return

        if not self.csv_path.exists():
            return

        left_frames = []
        right_frames = []

        capture_deltas = []

        left_latencies = []
        right_latencies = []

        previous_left_capture = None
        previous_right_capture = None

        with open(
            self.csv_path,
            newline="",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:

                report.csv.rows += 1

                left_frame = int(row["left_frame"])
                right_frame = int(row["right_frame"])

                left_capture = int(row["left_capture_ms"])
                right_capture = int(row["right_capture_ms"])

                left_receive = int(row["left_receive_ms"])
                right_receive = int(row["right_receive_ms"])

                left_frames.append(left_frame)
                right_frames.append(right_frame)

                capture_deltas.append(
                    abs(
                        left_capture -
                        right_capture
                    )
                )

                left_latencies.append(
                    left_receive -
                    left_capture
                )

                right_latencies.append(
                    right_receive -
                    right_capture
                )

                #
                # Timestamp wrap detection
                #

                if (
                    previous_left_capture is not None
                    and left_capture < previous_left_capture
                ):
                    report.csv.timestamp_wraps += 1

                if (
                    previous_right_capture is not None
                    and right_capture < previous_right_capture
                ):
                    report.csv.timestamp_wraps += 1

                previous_left_capture = left_capture
                previous_right_capture = right_capture

        self._compute_frame_statistics(
            left_frames,
            right_frames,
            report,
        )

        self._compute_sync_statistics(
            capture_deltas,
            left_latencies,
            right_latencies,
            report,
        )

    def _compute_frame_statistics(
        self,
        left_frames: list[int],
        right_frames: list[int],
        report: ValidationReport,
    ) -> None:
        """
        Compute duplicate and missing frame counts.
        """

        report.csv.duplicate_frames = (
            len(left_frames)
            - len(set(left_frames))
        )

        if left_frames:

            expected = (
                max(left_frames)
                - min(left_frames)
                + 1
            )

            report.csv.missing_frames = (
                expected
                - len(set(left_frames))
            )

    def _compute_sync_statistics(
        self,
        capture_deltas: list[int],
        left_latencies: list[int],
        right_latencies: list[int],
        report: ValidationReport,
    ) -> None:
        """
        Compute synchronization statistics.
        """

        if not capture_deltas:
            return

        sync = report.synchronization

        sync.pair_count = len(capture_deltas)

        sync.average_delta_ms = statistics.mean(
            capture_deltas
        )

        sync.median_delta_ms = statistics.median(
            capture_deltas
        )

        sync.minimum_delta_ms = min(
            capture_deltas
        )

        sync.maximum_delta_ms = max(
            capture_deltas
        )

        if len(capture_deltas) > 1:

            sync.stddev_delta_ms = (
                statistics.stdev(
                    capture_deltas
                )
            )

        sorted_values = sorted(capture_deltas)

        sync.percentile95_ms = self._percentile(
            sorted_values,
            95,
        )

        sync.percentile99_ms = self._percentile(
            sorted_values,
            99,
        )

        sync.average_left_latency_ms = (
            statistics.mean(left_latencies)
        )

        sync.average_right_latency_ms = (
            statistics.mean(right_latencies)
        )

    def _percentile(
        self,
        values: list[float],
        percentile: float,
    ) -> float:
        """
        Compute a percentile from sorted values.
        """

        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        index = (
            percentile / 100
        ) * (len(values) - 1)

        lower = int(index)
        upper = min(
            lower + 1,
            len(values) - 1,
        )

        fraction = index - lower

        return (
            values[lower]
            + (
                values[upper]
                - values[lower]
            )
            * fraction
        )