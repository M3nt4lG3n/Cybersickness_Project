"""
Eyetrackers Regression Test Runner.
"""

import time

from test_tracker_types import run as tracker_types_test
from test_framebuffer import run as framebuffer_test
from test_sync import (
    run as sync_test,
    failure_test,
    no_reuse_test,
)
from test_pipeline import (
    run as pipeline_test,
    jitter_test,
    dropped_frame_test,
    statistics_test,
    long_session_test,
    buffer_cleanup_test,
    out_of_order_test,
)
from test_recorder import run as recorder_test
from test_csvlogger import run as csvlogger_test
from test_display import run as display_test


def main():

    print("=" * 50)
    print("Eyetrackers Regression Test Suite")
    print("=" * 50)

    start = time.perf_counter()

    tests = [
        tracker_types_test,
        framebuffer_test,
        sync_test,
        failure_test,
        no_reuse_test,
        pipeline_test,
        jitter_test,
        dropped_frame_test,
        statistics_test,
        long_session_test,
        buffer_cleanup_test,
        out_of_order_test,
        recorder_test,
        csvlogger_test,
        display_test,
    ]

    for test in tests:
        test()

    elapsed = time.perf_counter() - start

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()