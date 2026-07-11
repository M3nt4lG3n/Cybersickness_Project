# Eyetrackers Architectural Decisions

**Architecture Version:** 1.0\
**Last Updated:** 2026-07-11\
**Status:** Living Document

------------------------------------------------------------------------

# Purpose

This document records significant architectural decisions made during
development of the Eyetrackers subsystem.

Each decision explains **what** was decided, **why** it was chosen, and
the expected impact. New architectural decisions should be appended
rather than replacing previous entries.

------------------------------------------------------------------------

# ADR-001 --- Dataclasses Replace Anonymous Dictionaries

**Status:** Accepted

## Decision

All inter-module communication uses strongly typed dataclasses.

Examples:

-   ESP32Metadata
-   FramePacket
-   SyncPair

Anonymous dictionaries are no longer used as module interfaces.

## Rationale

-   Better type safety
-   Easier refactoring
-   IDE autocompletion
-   Clear contracts between modules
-   Reduced runtime errors caused by missing dictionary keys

## Consequences

Every pipeline stage has a well-defined public interface.

------------------------------------------------------------------------

# ADR-002 --- ESP32 Is the Authoritative Clock

**Status:** Accepted

## Decision

Synchronization is based on timestamps embedded by the ESP32 firmware.

Host receive timestamps are used only for:

-   Latency estimation
-   Network diagnostics
-   Performance analysis

## Rationale

The ESP32 capture timestamp represents the actual image acquisition time
and is independent of network latency.

## Consequences

Stereo synchronization is resilient to Wi-Fi jitter.

------------------------------------------------------------------------

# ADR-003 --- MJPEGStream Owns FramePacket Creation

**Status:** Accepted

## Decision

Only `mjpeg.py` creates `ESP32Metadata` and `FramePacket` instances.

## Rationale

The MJPEG parser already has all required information:

-   JPEG image
-   Frame number
-   Capture timestamp
-   Receive timestamp

Creating packets elsewhere duplicates parsing logic.

## Consequences

Camera receives complete FramePacket objects.

------------------------------------------------------------------------

# ADR-004 --- StereoSynchronizer Owns SyncPair Creation

**Status:** Accepted

## Decision

Only `sync.py` creates SyncPair objects.

## Rationale

Synchronization policy belongs in one location.

Recorder, Display, and CSVLogger should never attempt to match frames.

## Consequences

A single synchronization algorithm governs the entire pipeline.

------------------------------------------------------------------------

# ADR-005 --- Recorder Saves Original Images

**Status:** Accepted

## Decision

Recorder writes the original images contained in each FramePacket.

Display overlays are applied only to temporary display images.

## Rationale

Debug overlays should never contaminate recorded data.

## Consequences

Recordings remain suitable for future computer vision processing.

------------------------------------------------------------------------

# ADR-006 --- Camera Is an Acquisition Manager

**Status:** Accepted

## Decision

Camera is responsible for:

-   Connection management
-   Acquisition thread
-   Runtime statistics
-   Buffer ownership

Camera is not responsible for:

-   JPEG parsing
-   Metadata creation
-   Stereo synchronization

## Rationale

Separating concerns keeps Camera focused on acquisition.

------------------------------------------------------------------------

# ADR-007 --- FrameBuffer Is the Buffer Authority

**Status:** Accepted

## Decision

FrameBuffer owns all buffering behavior.

Camera should not manipulate internal containers directly.

## Rationale

Centralizing buffering logic avoids duplicate implementations and
simplifies maintenance.

------------------------------------------------------------------------

# ADR-008 --- Data Flows Forward Only

**Status:** Accepted

## Decision

Objects move through the pipeline in one direction.

    ESP32
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
    Consumers

## Rationale

Forward-only flow simplifies reasoning and debugging.

------------------------------------------------------------------------

# ADR-009 --- Single Ownership

**Status:** Accepted

## Decision

Each core object has exactly one creator.

  Object          Creator
  --------------- --------------------
  ESP32Metadata   MJPEGStream
  FramePacket     MJPEGStream
  SyncPair        StereoSynchronizer

## Rationale

Single ownership eliminates ambiguity and duplicated construction logic.

------------------------------------------------------------------------

# Proposed Future Decisions

These topics may require future ADRs:

-   Multi-camera synchronization (3+ cameras)
-   Hardware trigger support
-   Recording file format changes
-   Real-time pupil detection pipeline
-   GPU acceleration
-   Distributed acquisition

------------------------------------------------------------------------

# How to Add a New Decision

Each new entry should include:

1.  Decision ID (ADR-010, ADR-011, ...)
2.  Status
3.  Decision
4.  Rationale
5.  Consequences

Architectural decisions should be documented before large-scale
refactors whenever practical.
