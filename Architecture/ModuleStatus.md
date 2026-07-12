# Eyetrackers Module Status

**Architecture Version:** 1.0\
**Last Updated:** 2026-07-11\
**Status:** Living Document

------------------------------------------------------------------------

# Purpose

This document tracks the implementation status of every module in the
Eyetrackers subsystem.

Unlike TODO.md, this file reflects the current architectural state of
the codebase. It should be updated whenever a module changes state.

------------------------------------------------------------------------

# Status Legend

  Symbol   Meaning
  -------- ------------------
  ✅       Complete
  🟡       In Progress
  🔴       Not Started
  ⚠️       Needs Review
  🔄       Planned Refactor

------------------------------------------------------------------------

# Core Modules

  -----------------------------------------------------------------------
  Module                    Status                    Notes
  ------------------------- ------------------------- -------------------
  tracker_types.py          ✅                        Dataclass migration
                                                      complete. Public
                                                      data contracts
                                                      established.

  mjpeg.py                  ✅                        Produces
                                                      FramePacket objects
                                                      directly from MJPEG
                                                      stream.

  framebuffer.py            ✅                        Thread-safe
                                                      FramePacket storage
                                                      implemented.

  sync.py                   ✅                        Produces SyncPair
                                                      objects.

  recorder.py               ✅                        Consumes SyncPair
                                                      objects for
                                                      synchronized
                                                      recording.

  csvlogger.py              ✅                        Consumes SyncPair
                                                      objects for
                                                      metadata logging.

  display.py                🟡                        Migrated to
                                                      SyncPair. Requires
                                                      final verification
                                                      with live cameras.

  camera.py                 🟡                        Final cleanup in
                                                      progress.
                                                      Converting
                                                      remaining logic to
                                                      FrameBuffer-based
                                                      API.

  config.py                 ✅                        Stable.

  main.py                   🔄                        Planned launcher
                                                      redesign.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Firmware

  -----------------------------------------------------------------------
  Component                     Status                  Notes
  ----------------------------- ----------------------- -----------------
  CameraWebServer               ✅                      Stable.

  app_httpd.cpp                 ✅                      Embeds frame
                                                        number and Unix
                                                        timestamp in
                                                        MJPEG headers.

  board_config.h                ✅                      Stable.

  camera_pins.h                 ✅                      Stable.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Architectural Migration

## Dictionary Removal

  Area           Status
  -------------- --------
  Recorder       ✅
  CSV Logger     ✅
  Synchronizer   ✅
  Display        ✅
  Camera         🟡

Goal:

Replace anonymous dictionaries with strongly typed dataclasses
throughout the pipeline.

------------------------------------------------------------------------

# Remaining Refactor Tasks

## High Priority

-   Finish camera.py migration
-   Complete launcher-style main.py
-   Verify full pipeline with both ESP32 cameras

## Medium Priority

-   Developer regression test harness
-   End-to-end integration testing
-   Runtime statistics improvements

## Future Features

-   Pupil detection
-   Eye tracking
-   Blink detection
-   Gaze estimation
-   Unity integration
-   Live visualization improvements

------------------------------------------------------------------------

# Verification Checklist

  Item                                       Status
  ------------------------------------------ --------
  FramePacket used everywhere                🟡
  SyncPair used everywhere                   🟡
  No anonymous dictionaries                  🟡
  ESP timestamps authoritative               ✅
  Display overlays isolated from recording   ✅
  Recorder receives original images          ✅

------------------------------------------------------------------------

## Validation Tools

## Status

Planned / In Progress

## Modules

validate_session.py

## Purpose

Post-acquisition quality assurance.

## Input

Completed recordings.

## Output

Human-readable validation report.

------------------------------------------------------------------------

## plot_session.py

## Status

Planned

## Purpose

Visualize synchronization quality.

------------------------------------------------------------------------

## playback.py

## Status

Planned

## Purpose

Replay synchronized recordings with metadata overlays.

------------------------------------------------------------------------

# Definition of Done

The Eyetrackers refactor will be considered complete when:

-   All modules communicate exclusively through dataclasses.
-   Camera no longer contains legacy buffering logic.
-   Launcher starts the complete acquisition pipeline.
-   Regression tests pass.
-   No remaining dictionary-based interfaces exist.
