from __future__ import annotations

import queue
import threading
import traceback
from enum import Enum, auto
from typing import Any, Callable, Optional


# ==========================================================
# Buffer Modes
# ==========================================================

class BufferMode(Enum):
    """
    Determines how the worker buffers items.
    """

    FIFO = auto()
    LATEST = auto()


# ==========================================================
# Output Worker
# ==========================================================

class OutputWorker:
    """
    Runs an output module on its own thread.

    The worker accepts items from the main acquisition
    loop and forwards them to the supplied callback.

    Supported modes:

        FIFO
            Every item is processed.

        LATEST
            Only the newest item is kept.
            Older frames are discarded automatically.
    """

    def __init__(
        self,
        callback: Callable[[Any], None],
        *,
        mode: BufferMode = BufferMode.FIFO,
        queue_size: int = 60,
        name: str = "OutputWorker",
    ):

        self.callback = callback
        self.mode = mode

        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=name,
        )

        #
        # FIFO mode
        #
        self.queue: queue.Queue[Optional[Any]] = queue.Queue(
            maxsize=queue_size
        )

        #
        # Latest mode
        #
        self._latest_item: Optional[Any] = None
        self._latest_lock = threading.Lock()
        self._latest_event = threading.Event()

        #
        # Statistics
        #
        self._submitted = 0
        self._processed = 0
        self._dropped = 0

    # ==========================================================
    # Public API
    # ==========================================================

    def start(self) -> None:
        """
        Start the worker thread.

        Calling start() multiple times has no effect.
        """

        if self.thread.is_alive():
            return

        self.thread.start()


    def stop(self) -> None:
        """
        Stop the worker thread and wait for it to exit.
        """

        self.stop_event.set()

        #
        # Wake whichever buffering mode is active.
        #
        if self.mode == BufferMode.FIFO:
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                #
                # Queue is full. Make room for the sentinel.
                #
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

                self.queue.put_nowait(None)

        else:
            self._latest_event.set()

        self.thread.join()


    def submit(
        self,
        item: Any,
    ) -> None:
        """
        Submit a synchronized frame pair for processing.
        """
        self._submitted += 1

        if self.stop_event.is_set():
            return

        #
        # Recorder / CSVLogger
        #
        if self.mode == BufferMode.FIFO:

            #
            # Block if necessary.
            #
            self.queue.put(item)

            return

        #
        # Display
        #
        with self._latest_lock:
            #
            # If another frame hasn't been displayed yet,
            # it is intentionally discarded.
            #
            if self._latest_item is not None:
                self._dropped += 1

            self._latest_item = item

        self._latest_event.set()

    # ==========================================================
    # Worker Thread
    # ==========================================================

    def _run(self) -> None:
        """
        Worker thread entry point.
        """

        if self.mode == BufferMode.FIFO:
            self._run_fifo()
        else:
            self._run_latest()


    def _run_fifo(self) -> None:
        """
        Process every submitted item in order.
        """

        while True:

            item = self.queue.get()

            #
            # Sentinel used during shutdown.
            #
            if item is None:
                break

            try:
                self.callback(item)
                self._processed += 1
            except Exception:

                print(
                    f"[{self.thread.name}] Worker exception:"
                )

                traceback.print_exc()


    def _run_latest(self) -> None:
        """
        Continuously process only the newest item.
        Older pairs are automatically discarded.
        """

        while not self.stop_event.is_set():

            #
            # Sleep until a new frame arrives.
            #
            self._latest_event.wait()

            self._latest_event.clear()

            if self.stop_event.is_set():
                break

            with self._latest_lock:
                item = self._latest_item
                self._latest_item = None

            if item is None:
                continue

            try:
                self.callback(item)
                self._processed += 1
            except Exception as exc:
                print(
                    f"[{self.thread.name}] "
                    f"Worker error: {exc}"
                )
    
    @property
    def submitted(self) -> int:
        """Total items submitted."""
        return self._submitted


    @property
    def processed(self) -> int:
        """Total items processed."""
        return self._processed


    @property
    def dropped(self) -> int:
        """Total items discarded."""
        return self._dropped


    @property
    def backlog(self) -> int:
        """
        Number of items waiting to be processed.
        """

        if self.mode == BufferMode.FIFO:
            return self.queue.qsize()

        return max(
            0,
            self._submitted -
            self._processed -
            self._dropped,
        )