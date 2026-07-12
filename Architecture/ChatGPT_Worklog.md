# Eyetrackers Development Worklog

**Project:** Cybersickness_Project / Eyetrackers

**Started:** 2026-07-11

**Purpose**

This document provides a running history of architectural changes, implementation progress, and future work. It serves as the handoff document between development sessions.

---

# Session 001

## Focus

Begin migrating the Eyetrackers pipeline from anonymous dictionaries to strongly typed dataclasses.

---

### Architectural Goal

Replace structures such as

```python
{
    "left": left,
    "right": right,
    "timestamp_ms": ...,
    "delta_ms": ...
}
```

with

```python
SyncPair
```

throughout the entire project.

---

## Completed

### tracker_types.py

- Introduced dataclasses.
- Established strongly typed pipeline objects.

---

### FrameBuffer

- Migrated to storing FramePacket objects.
- Centralized buffering behavior.

---

### StereoSynchronizer

- Produces SyncPair objects.
- Synchronization now operates on dataclasses instead of dictionaries.

---

### Recorder

- Updated to consume SyncPair.
- Access pattern changed from

```python
pair["left"]
```

to

```python
pair.left.image
```

---

### CSV Logger

- Updated to consume SyncPair.
- Removed dictionary-based field access.

---

### Display

- Began migration.
- Updated to consume SyncPair.
- Existing display behavior preserved where practical.

---

### MJPEGStream

Current design:

- Parses multipart MJPEG stream.
- Decodes JPEG images.
- Extracts ESP32 metadata.
- Produces FramePacket objects.

FramePacket creation now belongs exclusively to MJPEGStream.

---

### Camera

Migration in progress.

Current direction:

- Treat MJPEGStream as the owner of FramePacket creation.
- Camera becomes responsible only for

    - acquisition
    - buffering
    - runtime statistics
    - connection management

Remaining cleanup includes removing legacy buffer manipulation.

---

# Firmware

ESP32 firmware now embeds

- frame number
- Unix timestamp (milliseconds)

inside the MJPEG multipart header.

These timestamps are considered authoritative.

---

# Current Pipeline

ESP32 Firmware

↓

MJPEGStream

↓

FramePacket

↓

FrameBuffer

↓

StereoSynchronizer

↓

SyncPair

↓

Recorder

CSV Logger

Display

---

# Architectural Decisions Implemented

- Strongly typed dataclasses replace anonymous dictionaries.
- ESP32 timestamps are the synchronization source.
- MJPEGStream creates FramePacket objects.
- StereoSynchronizer creates SyncPair objects.
- Recorder receives original images only.
- Display overlays must never modify recorded images.

---

## Major Refactor

Completed

- Camera cleanup completed.
- Synchronizer migrated fully to SyncPair.
- FrameBuffer updated to support consume_closest().
- Launcher-style main.py generated.
- Comprehensive regression test suite designed.
- Session validation tool designed.

Architectural Changes

- Experiment Manager introduced above Eyetrackers.
- Validation separated from acquisition.
- Patient recording directories adopted.
- Output structure expanded to support future Unity and LabScribe integration.

---

# Remaining Work

## High Priority

- Finish camera.py
- Finish launcher-style main.py
- End-to-end pipeline verification

## Medium Priority

- Regression test harness
- Full integration testing
- Runtime diagnostics

## Future Features

- Pupil detection
- Eye tracking
- Blink detection
- Gaze estimation
- Unity integration
- Live statistics

---

# Notes

The architecture documents are now considered the source of truth for future development.

Future code changes should follow the documented architecture rather than introducing new design patterns.

---

# Next Session

1. Complete camera.py migration.
2. Verify FrameBuffer integration.
3. Finish launcher-style main.py.
4. Build developer regression test suite.
