"""Bridge between the LD_PRELOAD I/O interception library and frontrun.

The preload library (``libfrontrun_io.so``) intercepts libc I/O functions
and writes events to a log file (``FRONTRUN_IO_LOG``).  This module:

1. Creates a temporary log file and sets the env var before managed
   threads start.
2. Provides :func:`read_io_events` to parse the log after execution.
3. Provides a background reader that can feed events to the DPOR
   scheduler in real-time (via a polling thread).

Event log format (tab-separated)::

    <kind>\\t<resource_id>\\t<fd>\\t<pid>\\t<tid>

Where *kind* is one of: ``connect``, ``read``, ``write``, ``close``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


@dataclass
class PreloadIOEvent:
    """A single I/O event captured by the preload library."""

    kind: str  # "connect", "read", "write", "close"
    resource_id: str  # e.g. "socket:127.0.0.1:5432", "file:/tmp/data.db"
    fd: int
    pid: int
    tid: int


def setup_io_log() -> str:
    """Create a temporary log file and set ``FRONTRUN_IO_LOG``.

    Returns the path to the log file.  Call :func:`read_io_events` after
    execution to parse the events.
    """
    fd, path = tempfile.mkstemp(prefix="frontrun_io_", suffix=".log")
    os.close(fd)
    os.environ["FRONTRUN_IO_LOG"] = path
    return path


def cleanup_io_log(path: str) -> None:
    """Remove the temporary I/O log file and unset the env var."""
    os.environ.pop("FRONTRUN_IO_LOG", None)
    try:
        os.unlink(path)
    except OSError:
        pass


def read_io_events(path: str) -> list[PreloadIOEvent]:
    """Parse I/O events from the preload library's log file.

    Returns a list of :class:`PreloadIOEvent` in chronological order.
    Skips malformed lines silently.
    """
    events: list[PreloadIOEvent] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                try:
                    events.append(
                        PreloadIOEvent(
                            kind=parts[0],
                            resource_id=parts[1],
                            fd=int(parts[2]),
                            pid=int(parts[3]),
                            tid=int(parts[4]),
                        )
                    )
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return events


def filter_user_io_events(events: list[PreloadIOEvent]) -> list[PreloadIOEvent]:
    """Filter out Python startup / import I/O noise.

    Keeps only events for:
    - Socket connections (``socket:`` prefix)
    - User files (not under ``/usr/``, ``/lib/``, ``site-packages/``, etc.)
    """
    filtered: list[PreloadIOEvent] = []
    for ev in events:
        resource = ev.resource_id
        # Always keep socket events
        if resource.startswith("socket:"):
            filtered.append(ev)
            continue
        # Keep file events only for user paths
        if resource.startswith("file:"):
            path = resource[5:]
            # Skip stdlib, site-packages, and other system paths
            if any(
                seg in path
                for seg in (
                    "/usr/lib/python",
                    "/usr/local/lib/python",
                    "site-packages/",
                    "__pycache__",
                    ".pyc",
                    "/proc/",
                    "/sys/",
                    "/dev/",
                )
            ):
                continue
            filtered.append(ev)
    return filtered
