"""Shared marker coordination primitives for sync and async trace-marker execution.

This module contains the low-level synchronization classes used by both
:mod:`frontrun.trace_markers` (threads) and :mod:`frontrun.async_trace_markers`
(async tasks running in threads).  Both modules re-export these names so the
historical public API (``from frontrun.trace_markers import MarkerRegistry,
ThreadCoordinator``) continues to work.
"""

from __future__ import annotations

import io
import linecache
import re
import threading
import tokenize
from collections.abc import Callable
from typing import Any

from frontrun._cooperative import real_condition, real_lock
from frontrun._threaded_runner import join_threads_with_deadline
from frontrun.common import Schedule

# Docs promise ``# frontrun: <name>``: the name is the whole whitespace-free
# token, not just its leading ``\w+`` run. Truncating (`read-balance` -> `read`)
# made hyphenated schedules stall and let a prefix-colliding sibling marker
# silently consume another step's turn.
MARKER_PATTERN = re.compile(r"#\s*frontrun:\s*(\S+)")


def _validate_schedule_marker_names(schedule: Schedule) -> None:
    """Reject schedule steps whose marker name no comment can ever produce."""
    for step in schedule.steps:
        name = step.marker_name
        if not name or any(ch.isspace() for ch in name):
            raise ValueError(
                f"Step({step.execution_name!r}, {name!r}): marker names are single whitespace-free tokens "
                "(the <name> in '# frontrun: <name>'), so this step could never match a marker"
            )


def finalize_marker_executor_run(
    *,
    threads: list[threading.Thread],
    timeout: float | None,
    task_errors: dict[str, BaseException],
    coordinator: ThreadCoordinator,
    timeout_message: Callable[[list[threading.Thread]], str],
) -> None:
    """Join worker threads and validate post-conditions for a marker-based run.

    Shared between the sync :class:`~frontrun.trace_markers.TraceExecutor`
    and the async :class:`~frontrun.async_trace_markers.AsyncTraceExecutor`
    (which runs each async task inside its own worker thread).

    Raises:
        TimeoutError: If any worker is still alive after ``timeout``,
            using ``timeout_message`` to format the message; or if the
            schedule was partially consumed but not completed.
        Exception: The first exception found in ``task_errors``.
    """
    alive = join_threads_with_deadline(threads, timeout)
    if alive:
        raise TimeoutError(timeout_message(alive))

    if task_errors:
        raise next(iter(task_errors.values()))

    # Surface a coordinator-level error (e.g. a schedule stall detected inside
    # wait_for_turn).  The trace function swallows this exception so the worker
    # thread does not crash mid-trace, but the run is NOT valid: the threads ran
    # under an uncontrolled interleaving.  This must be raised — including the
    # step-0 stall case, which the heuristic below (current_step > 0) misses.
    if coordinator.error is not None:
        raise coordinator.error

    # If at least one step was processed (so the schedule was in use) but
    # the full schedule wasn't completed, it means the schedule references
    # markers that no worker reached. If zero steps were consumed, the
    # markers were simply never hit -- which could be a different issue
    # (wrong file, exec'd code, etc.) and is not necessarily an error for the
    # low-level executor. The exhaustive exploration API validates step zero.
    if (
        coordinator.current_step > 0
        and coordinator.current_step < len(coordinator.schedule.steps)
        and not coordinator.completed
    ):
        remaining = coordinator.schedule.steps[coordinator.current_step :]
        step_strs = [f"Step({s.execution_name!r}, {s.marker_name!r})" for s in remaining]
        raise TimeoutError(f"Schedule incomplete: {len(remaining)} step(s) were never reached: {', '.join(step_strs)}")


class MarkerRegistry:
    """Tracks marker locations in source code for efficient lookup.

    This class scans source files to find lines with frontrun markers and
    maintains a mapping from (filename, line_number) to marker names.
    """

    def __init__(self):
        self._markers: dict[tuple[str, int], str] = {}  # (filename, lineno) -> marker_name
        self._scanned_files: set[str] = set()
        self._lock = real_lock()

    def scan_frame(self, frame: Any) -> None:  # type: ignore[name-defined]
        """Scan the source file for the given frame to find all markers.

        Args:
            frame: A Python frame object from the trace function
        """
        filename = frame.f_code.co_filename

        # Fast path: Skip if already scanned (no lock needed)
        if filename in self._scanned_files:
            return

        # Double-checked locking: acquire lock and re-check
        with self._lock:
            # Re-check inside the lock in case another thread finished scanning
            if filename in self._scanned_files:
                return

            # Read all lines from the file and tokenize so that marker-like
            # text is only recognized inside real comments.  A bare regex over
            # raw source would also match ``# frontrun: x`` embedded in a
            # string literal on an executable line, registering a phantom
            # marker that is never in the schedule and stalls the run.
            try:
                # Use linecache to read the file
                linecache.checkcache(filename)
                lines: list[str] = []
                line_num = 1
                while True:
                    line = linecache.getline(filename, line_num)
                    if not line:
                        break
                    lines.append(line)
                    line_num += 1

                readline = io.StringIO("".join(lines)).readline
                try:
                    for tok in tokenize.generate_tokens(readline):
                        if tok.type != tokenize.COMMENT:
                            continue
                        match = MARKER_PATTERN.search(tok.string)
                        if match:
                            marker_name = match.group(1)
                            self._markers[(filename, tok.start[0])] = marker_name
                except tokenize.TokenError:
                    # Truncated/incomplete source: keep whatever markers were
                    # tokenized before the error.
                    pass
            except Exception:
                # If we can't read the file, just skip it
                pass

            # Mark as scanned AFTER we've populated all markers
            self._scanned_files.add(filename)

    def get_marker(self, filename: str, lineno: int) -> str | None:
        """Get the marker name for a specific file location.

        Args:
            filename: The source file path
            lineno: The line number

        Returns:
            The marker name if found, None otherwise
        """
        return self._markers.get((filename, lineno))


class ThreadCoordinator:
    """Coordinates thread execution according to a schedule.

    This class manages the synchronization between threads, ensuring that
    each thread executes markers in the order specified by the schedule.
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        _validate_schedule_marker_names(schedule)
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self.current_step = 0
        self.lock = real_lock()
        self.condition = real_condition(self.lock)
        self.completed = False
        self.error: Exception | None = None
        # Execution serialization lock: ensures only one thread runs between
        # markers, replicating GIL-like serialization needed on free-threaded
        # Python where threads truly run in parallel.
        self._execution_lock = real_lock()

    def wait_for_turn(self, execution_name: str, marker_name: str, *, _reacquire_execution_lock: bool = False):
        """Block until it's this execution unit's turn to execute this marker.

        When *_reacquire_execution_lock* is ``True`` (used by the trace
        executors), ``_execution_lock`` is acquired while the condition lock
        is still held, before returning.  This prevents other threads from
        racing ahead between being notified and the caller resuming execution.
        The caller must have already released ``_execution_lock`` before
        calling this method.

        Args:
            execution_name: The name of the calling execution unit
            marker_name: The marker that was hit
        """
        with self.condition:
            while True:
                # Check if we're done or had an error
                if self.completed or self.error:
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    return

                # Check if we've exceeded the schedule
                if self.current_step >= len(self.schedule.steps):
                    self.completed = True
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

                # Get the current expected step
                expected_step = self.schedule.steps[self.current_step]

                # Is it our turn?
                if expected_step.execution_name == execution_name and expected_step.marker_name == marker_name:
                    # It's our turn! Advance, optionally acquire execution lock, notify.
                    self.current_step += 1
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

                # Not our turn — wait with a fallback timeout so that
                # incorrect schedules (referencing a marker that no thread
                # ever reaches) get diagnosed promptly instead of blocking
                # until the outer thread.join(timeout) fires.
                if not self.condition.wait(timeout=self.deadlock_timeout):
                    # Re-check terminal conditions before indexing: another
                    # thread may have completed the schedule (advancing
                    # current_step to len(steps)) or reported an error while we
                    # were blocked in wait() (9a).
                    if self.completed or self.error or self.current_step >= len(self.schedule.steps):
                        continue
                    expected = self.schedule.steps[self.current_step]
                    self.error = TimeoutError(
                        f"Schedule stall: waiting for Step({expected.execution_name!r}, "
                        f"{expected.marker_name!r}) at step {self.current_step}/"
                        f"{len(self.schedule.steps)}, but no thread has reached it"
                    )
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

    def report_error(self, error: Exception):
        """Report an error and wake up all waiting threads.

        Args:
            error: The exception that occurred
        """
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def is_finished(self) -> bool:
        """Check if the schedule has completed or encountered an error."""
        with self.condition:
            return self.completed or self.error is not None
