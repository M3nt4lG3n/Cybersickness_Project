from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import Eyetrackers.Core.config as config

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Core.sync import StereoSynchronizer
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

    synchronizer: StereoSynchronizer

    recorder: Recorder
    csvlogger: CSVLogger
    display: Display

    display_worker: OutputWorker
    recorder_worker: OutputWorker
    csv_worker: OutputWorker


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

    synchronizer = StereoSynchronizer(
        left_camera=left_camera,
        right_camera=right_camera,
        tolerance_ms=config.SYNC_TOLERANCE_MS,
    )

    frame_size = (
        config.VIDEO_WIDTH,
        config.VIDEO_HEIGHT,
    )

    recorder = Recorder(
        left_filename=str(left_video),
        right_filename=str(right_video),
        fps=config.OUTPUT_FPS,
        frame_size=frame_size,
    )

    csvlogger = CSVLogger(
        str(csv_file)
    )

    display = Display()

    display_worker = OutputWorker(
        callback=display.render,
        mode=BufferMode.LATEST,
        name="DisplayWorker",
    )

    recorder_worker = OutputWorker(
        callback=recorder.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="RecorderWorker",
    )

    csv_worker = OutputWorker(
        callback=csvlogger.write,
        mode=BufferMode.FIFO,
        queue_size=120,
        name="CSVWorker",
    )

    print(f"Recording directory: {patient_directory}")

    return Application(
        patient_directory=patient_directory,
        left_camera=left_camera,
        right_camera=right_camera,
        synchronizer=synchronizer,
        recorder=recorder,
        csvlogger=csvlogger,
        display=display,
        display_worker=display_worker,
        recorder_worker=recorder_worker,
        csv_worker=csv_worker,
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

    app.display_worker.start()
    app.recorder_worker.start()
    app.csv_worker.start()

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
        app.display_worker.stop()
    except Exception:
        pass

    try:
        app.recorder_worker.stop()
    except Exception:
        pass 

    try:
        app.csv_worker.stop()
    except Exception:
        pass 

    try:
        app.recorder.close()
    except Exception:
        pass

    try:
        app.csvlogger.close()
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
        
        t0 = time.perf_counter()

        pair = app.synchronizer.get_pair()

        if pair is None:
            time.sleep(0.001)
            continue

        app.recorder_worker.submit(pair)

        app.csv_worker.submit(pair)

        app.display_worker.submit(pair)

        app.display.present()

        print(
            f"Left buffer : "
            f"{app.left_camera.buffer.size()}/"
            f"{app.left_camera.buffer.capacity}"
        )

        print(
            f"Right buffer: "
            f"{app.right_camera.buffer.size()}/"
            f"{app.right_camera.buffer.capacity}"
        )

        print(
            f"Right span: "
            f"{app.right_camera.buffer.age_span_ms()} ms"
        )

        print(
            f"Largest match index: "
            f"{app.right_camera.buffer.max_match_index}"
        )

        if time.time() - last_status >= 5:

            print(
                "\nWorker Status"
            )

            print(
                f"Recorder processed={app.recorder_worker.processed} "
                f"dropped={app.recorder_worker.dropped}"
            )

            print(
                f"CSV processed={app.csv_worker.processed} "
                f"dropped={app.csv_worker.dropped}"
            )

            print(
                f"Display processed={app.display_worker.processed} "
                f"dropped={app.display_worker.dropped}"
            ) 

            

            last_status = time.time()


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
