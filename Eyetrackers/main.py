"""
main.py

Launcher for the Eyetrackers subsystem.

Responsibilities
----------------
- Create application components
- Start camera acquisition
- Wait for cameras to become ready
- Process synchronized frame pairs
- Cleanly shut everything down

Synchronization, buffering, recording, and display logic
remain inside their respective modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config

from Eyetrackers.Core.camera import Camera
from Eyetrackers.Outputs.csvlogger import CSVLogger
from Eyetrackers.Outputs.display import Display
from Eyetrackers.Outputs.recorder import Recorder
from Eyetrackers.Core.sync import StereoSynchronizer
from Eyetrackers.Core.tracker_types import CameraConfig


# ==========================================================
# Application Container
# ==========================================================

@dataclass
class Application:
    left_camera: Camera
    right_camera: Camera

    synchronizer: StereoSynchronizer

    recorder: Recorder
    csvlogger: CSVLogger
    display: Display


# ==========================================================
# Component Construction
# ==========================================================

def create_components() -> Application:
    """
    Construct every application component.
    """

    left_config = CameraConfig(
        name="LEFT",
        stream_url=config.LEFT_CAMERA_URL,
        brightness=config.BRIGHTNESS,
        contrast=config.CONTRAST,
    )

    right_config = CameraConfig(
        name="RIGHT",
        stream_url=config.RIGHT_CAMERA_URL,
        brightness=config.BRIGHTNESS,
        contrast=config.CONTRAST,
    )

    left_camera = Camera(left_config)
    right_camera = Camera(right_config)

    synchronizer = StereoSynchronizer(
        left_camera=left_camera,
        right_camera=right_camera,
        tolerance_ms=config.SYNC_TOLERANCE_MS,
    )

    #
    # Adjust constructor arguments below if these
    # classes currently require additional parameters.
    #
    recorder = Recorder()

    csvlogger = CSVLogger()

    display = Display()

    return Application(
        left_camera=left_camera,
        right_camera=right_camera,
        synchronizer=synchronizer,
        recorder=recorder,
        csvlogger=csvlogger,
        display=display,
    )


# ==========================================================
# Startup
# ==========================================================

def start(app: Application) -> None:
    """
    Start camera acquisition.
    """

    print("Starting cameras...")

    app.left_camera.start()
    app.right_camera.start()

    print("Waiting for cameras...")

    if not app.left_camera.wait_until_ready():
        raise RuntimeError("Left camera failed to initialize.")

    if not app.right_camera.wait_until_ready():
        raise RuntimeError("Right camera failed to initialize.")

    print("Both cameras ready.")


# ==========================================================
# Shutdown
# ==========================================================

def shutdown(app: Application) -> None:
    """
    Stop every subsystem.
    """

    print("\nStopping...")

    #
    # Stop cameras
    #
    app.left_camera.stop()
    app.right_camera.stop()

    #
    # Close recorder
    #
    if hasattr(app.recorder, "close"):
        app.recorder.close()

    #
    # Close CSV logger
    #
    if hasattr(app.csvlogger, "close"):
        app.csvlogger.close()

    #
    # Close display
    #
    if hasattr(app.display, "close"):
        app.display.close()

    print("Shutdown complete.")


# ==========================================================
# Main Processing Loop
# ==========================================================

def run(app: Application) -> None:
    """
    Process synchronized frame pairs.
    """

    while True:

        pair = app.synchronizer.get_pair()

        if pair is None:
            time.sleep(0.001)
            continue

        #
        # Recorder
        #
        app.csvlogger.write(pair)

        #
        # CSV Logger
        #
        app.recorder.write(pair)

        #
        # Display
        #
        app.display.show(pair)


# ==========================================================
# Entry Point
# ==========================================================

def main():

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
