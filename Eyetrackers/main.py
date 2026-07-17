from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import Eyetrackers.Core.config as config

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Core.tracker_types import CameraConfig
from Eyetrackers.Core.output_worker import OutputWorker, BufferMode

from Eyetrackers.Outputs.recorder import Recorder
from Eyetrackers.Outputs.csvlogger import CSVLogger
from Eyetrackers.Outputs.display import Display


# ==========================================================
# Application Container
# ==========================================================

@dataclass
class Application:

    patient_directory: Path

    left_camera: Camera
    right_camera: Camera

    left_recorder: Recorder
    right_recorder: Recorder

    left_csvlogger: CSVLogger
    right_csvlogger: CSVLogger

    display: Display

    left_display_worker: OutputWorker
    right_display_worker: OutputWorker

    left_recorder_worker: OutputWorker
    right_recorder_worker: OutputWorker

    left_csv_worker: OutputWorker
    right_csv_worker: OutputWorker


# ==========================================================
# Recording Directory
# ==========================================================

def create_patient_directory() -> Path:
    """
    Temporary recording directory creator.

    This will eventually be replaced by the Experiment
    Manager, which will provide the directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    directory = config.RAW_VIDEO_DIR / f"Patient_{timestamp}"

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ==========================================================
# Component Construction
# ==========================================================

def create_components() -> Application:

    patient_directory = create_patient_directory()

    left_video = patient_directory / "left_eye.mp4"
    right_video = patient_directory / "right_eye.mp4"

    csv_file = patient_directory / "eyetracker.csv"

    left_camera = Camera(config.LEFT_CAMERA)
    right_camera = Camera(config.RIGHT_CAMERA)

    frame_size = (
        config.VIDEO_WIDTH,
        config.VIDEO_HEIGHT,
    )

    left_recorder = Recorder(
        filename=str(left_video),
        fps=config.OUTPUT_FPS,
        frame_size=frame_size,
    )

    right_recorder = Recorder(
        filename=str(right_video),
        fps=config.OUTPUT_FPS,
        frame_size=frame_size,
    )

    left_csvlogger = CSVLogger(
        str(patient_directory / "left_eye.csv")
    )

    right_csvlogger = CSVLogger(
        str(patient_directory / "right_eye.csv")
    )

    display = Display()

    left_display_worker = OutputWorker(
        callback=display.render_left,
        mode=BufferMode.LATEST,
        name="DisplayWorker",
    )

    right_display_worker = OutputWorker(
        callback=display.render_right,
        mode=BufferMode.LATEST,
        name="DisplayWorker",
    )

    left_recorder_worker = OutputWorker(
        callback=left_recorder.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="RecorderWorker",
    )

    right_recorder_worker = OutputWorker(
        callback=right_recorder.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="RecorderWorker",
    )

    left_csv_worker = OutputWorker(
        callback=left_csvlogger.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="CSVWorker",
    )

    right_csv_worker = OutputWorker(
        callback=right_csvlogger.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="CSVWorker",
    )

    print(f"Recording directory: {patient_directory}")

    return Application(
        patient_directory=patient_directory,
        left_camera=left_camera,
        right_camera=right_camera,
        left_recorder=left_recorder,
        right_recorder=right_recorder,
        left_csvlogger=left_csvlogger,
        right_csvlogger=right_csvlogger,
        display=display,
        left_display_worker=left_display_worker,
        right_display_worker=right_display_worker,
        left_recorder_worker=left_recorder_worker,
        right_recorder_worker=right_recorder_worker,
        left_csv_worker=left_csv_worker,
        right_csv_worker=right_csv_worker,
    )


# ==========================================================
# Startup
# ==========================================================

def start(app: Application) -> None:
    """
    Start camera acquisition and wait for both
    cameras to become ready.
    """

    print("Starting cameras...")

    app.left_display_worker.start()
    app.right_display_worker.start()
    app.left_recorder_worker.start()
    app.right_recorder_worker.start()

    app.left_csv_worker.start()
    app.right_csv_worker.start()

    app.left_camera.start()
    app.right_camera.start()

    print("Waiting for first frames...")

    if not app.left_camera.wait_until_ready():
        raise RuntimeError(
            "Left camera failed to initialize."
        )

    if not app.right_camera.wait_until_ready():
        raise RuntimeError(
            "Right camera failed to initialize."
        )

    print("Both cameras synchronized.")


# ==========================================================
# Shutdown
# ==========================================================

def shutdown(app: Application) -> None:
    """
    Cleanly stop every subsystem.
    """

    print("\nStopping Eyetrackers...")

    try:
        app.left_camera.stop()
    except Exception:
        pass

    try:
        app.right_camera.stop()
    except Exception:
        pass

    try:
        app.left_display_worker.stop()
    except Exception:
        pass

    try:
        app.right_display_worker.stop()
    except Exception:
        pass

    try:
        app.left_recorder_worker.stop()
    except Exception:
        pass 

    try:
        app.right_recorder_worker.stop()
    except Exception:
        pass 

    try:
        app.left_csv_worker.stop()
    except Exception:
        pass 

    try:
        app.right_csv_worker.stop()
    except Exception:
        pass 

    try:
        app.left_recorder.close()
    except Exception:
        pass

    try:
        app.right_recorder.close()
    except Exception:
        pass

    try:
        app.left_csvlogger.close()
    except Exception:
        pass

    try:
        app.right_csvlogger.close()
    except Exception:
        pass

    try:
        app.display.close()
    except Exception:
        pass

    print(
        f"Recording complete.\n"
        f"Output directory:\n"
        f"    {app.patient_directory}"
    )


# ==========================================================
# Main Processing Loop
# ==========================================================

def run(app: Application) -> None:
    """
    Continuously process synchronized frame pairs.
    """

    print("Recording started.")
    print("Press Q or Esc to stop.\n")

    last_status = time.time()

    while True:

        if app.display.should_close:
            break

        left = app.left_camera.consume_oldest()

        if left is not None:

            app.left_recorder_worker.submit(left)

            app.left_csv_worker.submit(left)

            app.left_display_worker.submit(left)

        right = app.right_camera.consume_oldest()

        if right is not None:

            app.right_recorder_worker.submit(right)

            app.right_csv_worker.submit(right)

            app.right_display_worker.submit(right)

        if time.time() - last_status >= 5:

            print(
                "\nWorker Status"
            )

            print(
                f"Left Recorder processed={app.left_recorder_worker.processed} "
                f"dropped={app.left_recorder_worker.dropped}"
            )

            print(
                f"Right Display processed={app.right_display_worker.processed} "
                f"dropped={app.right_display_worker.dropped}"
            ) 

            print(
                f"Left CSV processed={app.left_csv_worker.processed} "
                f"dropped={app.left_csv_worker.dropped}"
            )

            print(
                f"Right CSV processed={app.right_csv_worker.processed} "
                f"dropped={app.right_csv_worker.dropped}"
            )

            print(
                f"Left Display processed={app.left_display_worker.processed} "
                f"dropped={app.left_display_worker.dropped}"
            )

            print(
                f"Right Display processed={app.right_display_worker.processed} "
                f"dropped={app.right_display_worker.dropped}"
            )

            last_status = time.time()

        app.display.present()


# ==========================================================
# Entry Point
# ==========================================================

def main() -> None:

    app = create_components()

    try:

        start(app)

        run(app)

    except KeyboardInterrupt:

        print("\nKeyboard interrupt received.")

    finally:

        shutdown(app)


if __name__ == "__main__":
    main()
