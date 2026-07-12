# Eyetrackers Pipeline

**Architecture Version:** 1.0\
**Last Updated:** 2026-07-11\
**Status:** Active

------------------------------------------------------------------------

# Purpose

This document defines the processing pipeline for the Eyetrackers
subsystem. Each stage has a single responsibility and a well-defined
input and output.

------------------------------------------------------------------------

# End-to-End Pipeline

``` text
ESP32 Firmware
    │
    ▼
MJPEGStream
    │  (FramePacket)
    ▼
Camera
    │  (FrameBuffer)
    ▼
StereoSynchronizer
    │  (SyncPair)
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Recorder      CSVLogger      Display
```

------------------------------------------------------------------------

# Stage Contracts

  --------------------------------------------------------------------------
  Stage                Input             Output            Responsibility
  -------------------- ----------------- ----------------- -----------------
  ESP32 Firmware       Sensor            MJPEG stream      Capture image and
                                                           embed metadata

  MJPEGStream          HTTP MJPEG        FramePacket       Decode JPEG and
                                                           parse metadata

  Camera               FramePacket       FrameBuffer       Acquisition,
                                                           buffering,
                                                           runtime state

  StereoSynchronizer   Two FrameBuffers  SyncPair          Match left/right
                                                           frames

  Recorder             SyncPair          MP4               Save synchronized
                                                           video

  CSVLogger            SyncPair          CSV               Save
                                                           synchronization
                                                           metadata

  Display              SyncPair          OpenCV Window     Visualize
                                                           synchronized
                                                           frames
  --------------------------------------------------------------------------

------------------------------------------------------------------------

# Pipeline Rules

1.  Data always flows forward.
2.  FramePacket objects are created only by MJPEGStream.
3.  SyncPair objects are created only by StereoSynchronizer.
4.  Recorder, CSVLogger, and Display never modify images or metadata.
5.  Synchronization is based on ESP32 timestamps, not host receive time.

------------------------------------------------------------------------

## Recording Lifecycle

1. Experiment Manager creates the patient directory.
2. Experiment Manager creates subsystem output directories.
3. Cameras connect.
4. Recorder opens output files.
5. CSVLogger opens output CSV.
6. Acquisition begins.
7. StereoSynchronizer produces SyncPairs.
8. Recorder writes original images.
9. CSVLogger writes synchronization metadata.
10. Display renders overlays.
11. Recorder closes.
12. CSVLogger closes.
13. Validation tools analyze the completed recording.

Validation is not part of acquisition.

Validation tools never modify recorded data.

Validation tools only read completed recordings.

------------------------------------------------------------------------

# Future Pipeline Extensions

Possible future stages:

-   Eye detection
-   Pupil tracking
-   Gaze estimation
-   Blink detection
-   Unity integration
-   Real-time analytics
