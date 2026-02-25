"""Bridge between the LD_PRELOAD I/O interception library and frontrun.

The preload library (``libfrontrun_io.so`` / ``libfrontrun_io.dylib``)
intercepts libc I/O functions and reports events back to Python.

Two transport mechanisms are supported:

**Pipe transport (preferred):**  :class:`IOEventDispatcher` creates an
``os.pipe()``, passes the write-end fd to the preload library via
``FRONTRUN_IO_FD``, and reads events from the read end in a background
thread.  Registered listener callbacks are invoked for each event as it
arrives.  The pipe's FIFO ordering provides a natural total order without
timestamps — events are delivered in the exact order the C library
produced them.  Since the DPOR scheduler controls which thread runs at
any given moment, the ``tid`` field on each event is sufficient to
attribute it to the correct schedule step (no timestamp-based merging
required).

**Log-file transport (legacy):**  :func:`setup_io_log` creates a temp
file and sets ``FRONTRUN_IO_LOG``.  After execution,
:func:`read_io_events` parses the log.  This incurs an open/close per
event on the Rust side and only supports batch (not streaming) reads.

Event format (tab-separated, same for both transports)::

    <kind>\\t<resource_id>\\t<fd>\\t<pid>\\t<tid>

Where *kind* is one of: ``connect``, ``read``, ``write``, ``close``.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import select as _select_mod
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass

from frontrun._cooperative import real_lock


@dataclass
class PreloadIOEvent:
    """A single I/O event captured by the preload library."""

    kind: str  # "connect", "read", "write", "close"
    resource_id: str  # e.g. "socket:127.0.0.1:5432", "file:/tmp/data.db"
    fd: int
    pid: int
    tid: int


# Callback type for I/O event listeners.
# Signature: (event: PreloadIOEvent) -> None
IOEventListener = Callable[[PreloadIOEvent], None]


def _parse_event_line(line: str) -> PreloadIOEvent | None:
    """Parse a single tab-separated event line, or return None if malformed."""
    line = line.rstrip("\n")
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 5:
        return None
    try:
        return PreloadIOEvent(
            kind=parts[0],
            resource_id=parts[1],
            fd=int(parts[2]),
            pid=int(parts[3]),
            tid=int(parts[4]),
        )
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Direct pipe-fd setter via ctypes (bypasses cached env-var lookup)
# ---------------------------------------------------------------------------


def _set_preload_pipe_fd(fd: int) -> bool:
    """Directly set the pipe fd in the LD_PRELOAD library.

    The Rust preload library caches ``FRONTRUN_IO_FD`` on first use, so
    setting the env var after any I/O has occurred (e.g. during Python
    startup) has no effect.  This function calls the exported
    ``frontrun_io_set_pipe_fd`` symbol to update the cached value directly.

    Returns ``True`` if the call succeeded, ``False`` if the library
    is not loaded or the symbol is unavailable.
    """
    try:
        lib = ctypes.CDLL(None)
        func = lib.frontrun_io_set_pipe_fd
        func.argtypes = [ctypes.c_int]
        func.restype = None
        func(fd)
        return True
    except (OSError, AttributeError):
        return False


def _set_preload_pipe_read_fd(fd: int) -> bool:
    """Tell the LD_PRELOAD library which fd is the pipe read end.

    Reads on this fd bypass interception entirely — no ``ensure_fd_mapped``
    overhead, no ``FD_MAP`` contention.  This prevents deadlocks where the
    pipe reader thread blocks on FD_MAP inside LD_PRELOAD's ``read()`` hook
    while holding ``_pipe_lock``.
    """
    try:
        lib = ctypes.CDLL(None)
        func = lib.frontrun_io_set_pipe_read_fd
        func.argtypes = [ctypes.c_int]
        func.restype = None
        func(fd)
        return True
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Pipe-based transport with listener callbacks
# ---------------------------------------------------------------------------


class IOEventDispatcher:
    """Stream I/O events from the preload library via a pipe.

    Creates an ``os.pipe()`` and sets ``FRONTRUN_IO_FD`` to the write-end
    fd so the Rust preload library writes events there instead of to a
    log file.  A daemon thread reads the pipe and dispatches events to
    registered listeners.

    Usage::

        dispatcher = IOEventDispatcher()
        dispatcher.add_listener(lambda ev: print(ev))
        dispatcher.start()
        # ... run code under LD_PRELOAD / DYLD_INSERT_LIBRARIES ...
        dispatcher.stop()

    Or as a context manager::

        with IOEventDispatcher() as dispatcher:
            dispatcher.add_listener(my_callback)
            # ... run code ...
    """

    def __init__(self) -> None:
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._listeners: list[IOEventListener] = []
        # Use real (non-cooperative) locks so the background reader thread
        # is never affected by frontrun's cooperative lock patching.
        self._lock = real_lock()
        self._pipe_lock = real_lock()  # serialises pipe reads between reader thread and poll()
        self._buf = b""  # partial-line buffer, protected by _pipe_lock
        self._started = False
        self._stopped = False
        self._events: list[PreloadIOEvent] = []

    def add_listener(self, listener: IOEventListener) -> None:
        """Register a callback invoked for each arriving I/O event.

        Listeners are called from the reader thread in the order events
        arrive through the pipe.  Keep listener work minimal to avoid
        back-pressure on the pipe.
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: IOEventListener) -> None:
        """Unregister a previously registered listener."""
        with self._lock:
            self._listeners.remove(listener)

    @property
    def events(self) -> list[PreloadIOEvent]:
        """All events received so far (thread-safe snapshot)."""
        with self._lock:
            return list(self._events)

    def start(self) -> None:
        """Create the pipe, set the env var, and start the reader thread."""
        if self._started:
            return
        r, w = os.pipe()
        # Make the read end non-blocking so poll() and the reader thread
        # can share it safely under _pipe_lock without blocking each other.
        flags = fcntl.fcntl(r, fcntl.F_GETFL)
        fcntl.fcntl(r, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        self._read_fd = r
        self._write_fd = w
        os.environ["FRONTRUN_IO_FD"] = str(w)
        # Also clear FRONTRUN_IO_LOG so the Rust side prefers the pipe.
        os.environ.pop("FRONTRUN_IO_LOG", None)
        # Directly update the cached pipe fd in the LD_PRELOAD library.
        # The Rust side caches FRONTRUN_IO_FD on first use, so setting the
        # env var alone is not enough if any I/O occurred during startup.
        _set_preload_pipe_fd(w)
        # Tell the LD_PRELOAD library which fd is the pipe read end so it
        # skips interception on reads from it (avoids ensure_fd_mapped
        # overhead and FD_MAP contention that can deadlock the reader thread).
        _set_preload_pipe_read_fd(r)

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="frontrun-io-reader",
            daemon=True,
        )
        self._started = True
        self._reader_thread.start()

    def stop(self) -> None:
        """Close the write end and wait for the reader thread to drain."""
        if self._stopped:
            return
        self._stopped = True

        # Reset the cached pipe fds in the LD_PRELOAD library before closing,
        # so the Rust side stops writing to the about-to-be-closed fd.
        _set_preload_pipe_fd(-1)
        _set_preload_pipe_read_fd(-1)

        # Close write end so the reader sees EOF.
        if self._write_fd is not None:
            try:
                os.close(self._write_fd)
            except OSError:
                pass
            self._write_fd = None

        os.environ.pop("FRONTRUN_IO_FD", None)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)

        # Close read end after reader exits.
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
            self._read_fd = None

    def _read_parse_dispatch(self) -> tuple[list[PreloadIOEvent], bool]:
        """Read pipe data, parse events, and dispatch to listeners — all under ``_pipe_lock``.

        Holding ``_pipe_lock`` across the entire read→parse→dispatch cycle
        ensures there is no window where an event has been read from the pipe
        but not yet delivered to listeners.  Without this, ``poll()`` (called
        from the DPOR scheduling path) could see an empty pipe while the
        reader thread is mid-dispatch, causing I/O events to be attributed
        to a later scheduling step and triggering exponential DPOR path
        explosion on free-threaded Python.

        Returns ``(events, got_data)`` where *got_data* is True when at
        least one byte was read from the pipe (used for EOF detection).
        """
        with self._pipe_lock:
            assert self._read_fd is not None
            got_data = False
            while True:
                try:
                    chunk = os.read(self._read_fd, 65536)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not chunk:
                    break  # EOF
                got_data = True
                self._buf += chunk
            events: list[PreloadIOEvent] = []
            while b"\n" in self._buf:
                line_bytes, self._buf = self._buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace")
                event = _parse_event_line(line)
                if event is not None:
                    events.append(event)
            # Dispatch while still holding _pipe_lock so that poll() callers
            # are guaranteed to see all dispatched events in listener buffers
            # by the time they acquire _pipe_lock themselves.
            for event in events:
                with self._lock:
                    self._events.append(event)
                    listeners = list(self._listeners)
                for listener in listeners:
                    listener(event)
        return events, got_data

    def _reader_loop(self) -> None:
        """Background thread: wait for pipe data via select, then read and dispatch."""
        assert self._read_fd is not None
        read_fd = self._read_fd
        while True:
            try:
                ready, _, _ = _select_mod.select([read_fd], [], [], 0.05)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            events, got_data = self._read_parse_dispatch()
            if not events and not got_data and self._stopped:
                break

    def poll(self) -> None:
        """Synchronously read all available pipe data and dispatch events.

        On free-threaded Python (3.13t+) the background reader thread may
        not have processed pipe data by the time the calling thread reaches
        its next scheduling point.  Calling ``poll()`` from the DPOR drain
        path ensures events are available immediately.

        Uses ``select()`` as a fast guard: ``select`` is not intercepted by
        the LD_PRELOAD library, whereas ``os.read`` is — each intercepted
        read triggers ``getpeername`` + ``readlink`` syscalls on the pipe fd.
        Skipping ``os.read`` when the pipe is empty avoids that overhead on
        every opcode boundary.
        """
        if self._read_fd is None or self._stopped:
            return
        try:
            ready, _, _ = _select_mod.select([self._read_fd], [], [], 0)
        except (OSError, ValueError):
            return
        if not ready:
            return
        self._read_parse_dispatch()

    # Context manager support
    def __enter__(self) -> IOEventDispatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Legacy file-based transport
# ---------------------------------------------------------------------------


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
                event = _parse_event_line(line)
                if event is not None:
                    events.append(event)
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
