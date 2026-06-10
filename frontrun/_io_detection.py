"""Auto-detection of I/O operations (sockets, files) for concurrency testing.

Provides two layers of I/O detection:

**Layer 1 — Socket/file monkey-patching:**
Patches ``socket.socket`` methods and ``builtins.open`` to report resource
accesses to the scheduler. This follows the same pattern as
``_cooperative.py``'s monkey-patching of threading primitives.

**Layer 1.5 — ``sys.setprofile`` C-call detection:**
Installs per-thread profile functions that detect C-level socket/file
operations invisible to ``sys.settrace``.  Coexists with ``sys.settrace``
and ``sys.monitoring`` without interference.

Both layers report accesses through a per-thread callback stored in TLS,
which the scheduler (bytecode or DPOR) provides when setting up each thread.

Resource identity is derived from the socket's peer address ``(host, port)``
or the file's resolved path.  Two threads accessing the same endpoint or
file are reported as conflicting; different endpoints are independent.
"""

from __future__ import annotations

import builtins
import contextvars
import os
import socket
import sys
import threading
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Per-thread IO reporter callback (set by scheduler)
# ---------------------------------------------------------------------------

_io_tls = threading.local()

# ---------------------------------------------------------------------------
# Task-aware DPOR context (contextvars) — takes precedence over _io_tls.
#
# Async DPOR runs every task on the SAME event-loop thread, so per-thread
# storage (``_io_tls``) cannot distinguish tasks: after all tasks start,
# the threading.local DPOR thread-id is permanently the last task's id.
# Contextvars ARE per-task (asyncio copies the context for each Task), so
# the async scheduler sets these per task and they resolve correctly.
#
# Sync paths never set these (each worker thread is a real OS thread with
# its own ``_io_tls``), so the ``_UNSET`` default makes ``get_*`` fall back
# to ``_io_tls`` — keeping sync behaviour bit-identical and zero-risk.
# ---------------------------------------------------------------------------

_UNSET: Any = object()

_dpor_scheduler_var: contextvars.ContextVar[Any] = contextvars.ContextVar("_dpor_scheduler_var", default=_UNSET)
_dpor_thread_id_var: contextvars.ContextVar[Any] = contextvars.ContextVar("_dpor_thread_id_var", default=_UNSET)

# Callback signature: (resource_id: str, kind: str) -> None
#   resource_id: e.g. "socket:127.0.0.1:5432" or "file:/tmp/data.db"
#   kind: "read" or "write"
IOReporter = Callable[[str, str], None]


def get_io_reporter() -> IOReporter | None:
    """Return the per-thread IO reporter, or ``None``."""
    return getattr(_io_tls, "io_reporter", None)


def set_io_reporter(reporter: IOReporter | None) -> None:
    """Install a per-thread IO reporter (or clear with ``None``)."""
    _io_tls.io_reporter = reporter


def get_dpor_scheduler() -> Any:
    """Return the active DPOR scheduler reference, or ``None``.

    Prefers the task-aware contextvar when it has been set (async DPOR);
    otherwise falls back to the per-thread ``_io_tls`` value (sync DPOR).
    """
    scheduler = _dpor_scheduler_var.get()
    if scheduler is not _UNSET:
        return scheduler
    return getattr(_io_tls, "_dpor_scheduler", None)


def set_dpor_scheduler(scheduler: Any) -> None:
    """Install a per-thread DPOR scheduler reference (or clear with ``None``)."""
    _io_tls._dpor_scheduler = scheduler


def set_dpor_scheduler_task(scheduler: Any) -> None:
    """Install the DPOR scheduler in the task-aware contextvar (async DPOR)."""
    _dpor_scheduler_var.set(scheduler)


def get_dpor_thread_id() -> int | None:
    """Return the active DPOR thread/task ID, or ``None``.

    Prefers the task-aware contextvar when it has been set (async DPOR, where
    every task shares one OS thread); otherwise falls back to the per-thread
    ``_io_tls`` value (sync DPOR).
    """
    thread_id = _dpor_thread_id_var.get()
    if thread_id is not _UNSET:
        return thread_id
    return getattr(_io_tls, "_dpor_thread_id", None)


def set_dpor_thread_id(thread_id: int | None) -> None:
    """Install a per-thread DPOR thread ID (or clear with ``None``)."""
    _io_tls._dpor_thread_id = thread_id


def set_dpor_thread_id_task(thread_id: int | None) -> None:
    """Install the DPOR thread/task ID in the task-aware contextvar (async DPOR)."""
    _dpor_thread_id_var.set(thread_id)


# ---------------------------------------------------------------------------
# Task-aware SQL transaction state
# ---------------------------------------------------------------------------
#
# SQL transaction state (``_in_transaction``, ``_tx_buffer``, ``_tx_savepoints``,
# ``_pending_row_locks``, …) lives on ``_io_tls`` for sync DPOR.  Async DPOR
# shares one event-loop thread across all tasks, so that state must instead be
# isolated per task.  ``set_tx_store_task`` installs a fresh per-task object in
# a contextvar; ``tx_store`` returns it when set, else falls back to ``_io_tls``.


class _TxStore:
    """Per-task attribute bag for SQL transaction state (async DPOR).

    Mirrors the attribute-access surface of ``_io_tls`` used by
    ``_sql_transactions`` / ``_sql_row_locks`` so those modules can target
    either store transparently via :func:`tx_store`.
    """


_tx_store_var: contextvars.ContextVar[Any] = contextvars.ContextVar("_tx_store_var", default=_UNSET)


def tx_store() -> Any:
    """Return the active SQL transaction-state store.

    The task-aware contextvar store (async DPOR) when set, else ``_io_tls``
    (sync DPOR and the unpatched default).
    """
    store = _tx_store_var.get()
    if store is not _UNSET:
        return store
    return _io_tls


def set_tx_store_task() -> Any:
    """Install a fresh per-task transaction-state store and return it."""
    store = _TxStore()
    _tx_store_var.set(store)
    return store


def get_dpor_context() -> tuple[Any, int] | None:
    """Return ``(scheduler, thread_id)`` if DPOR is active, else ``None``.

    Convenience wrapper used by SQL and Redis interception modules to
    obtain the active scheduler and logical thread ID in a single call.
    """
    scheduler = get_dpor_scheduler()
    if scheduler is None:
        return None
    thread_id = get_dpor_thread_id()
    if thread_id is None:
        return None
    return scheduler, thread_id


# ---------------------------------------------------------------------------
# Resource identity helpers
# ---------------------------------------------------------------------------


def _socket_resource_id(sock: socket.socket) -> str | None:
    """Derive a resource ID from a socket's peer address."""
    try:
        peer = sock.getpeername()
        if isinstance(peer, tuple) and len(peer) >= 2:
            return f"socket:{peer[0]}:{peer[1]}"
        return f"socket:{peer}"
    except (OSError, AttributeError):
        # Not connected yet or already closed
        return None


def _file_resource_id(path: str) -> str:
    """Derive a resource ID from a file path."""
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        resolved = path
    return f"file:{resolved}"


# ---------------------------------------------------------------------------
# Layer 1: Socket monkey-patching
# ---------------------------------------------------------------------------

# Save real methods before patching
_real_socket_connect = socket.socket.connect
_real_socket_send = socket.socket.send
_real_socket_sendall = socket.socket.sendall
_real_socket_sendto = socket.socket.sendto
_real_socket_recv = socket.socket.recv
_real_socket_recv_into = socket.socket.recv_into
_real_socket_recvfrom = socket.socket.recvfrom


def _report_socket_io(sock: socket.socket, kind: str) -> None:
    """Report a socket I/O event to the per-thread reporter, if installed."""
    # Skip if SQL-level or Redis-level detection already reported for this call
    if getattr(_io_tls, "_sql_suppress", False) or getattr(_io_tls, "_redis_suppress", False):
        return
    reporter = get_io_reporter()
    if reporter is not None:
        resource_id = _socket_resource_id(sock)
        if resource_id is not None:
            reporter(resource_id, kind)


def _make_traced_socket_method(
    real_method: Callable[..., Any],
    kind: str,
    *,
    report_after: bool = False,
) -> Callable[..., Any]:
    """Create a traced wrapper for a ``socket.socket`` method.

    *real_method* is the saved original (e.g. ``_real_socket_send``).
    *kind* is ``"read"`` or ``"write"``.  When *report_after* is true the
    report fires after the real call (needed for ``connect``, which must
    complete before ``getpeername()`` works).
    """

    def traced(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
        if not report_after:
            _report_socket_io(self, kind)
        result = real_method(self, *args, **kwargs)
        if report_after:
            _report_socket_io(self, kind)
        return result

    return traced


_traced_connect = _make_traced_socket_method(_real_socket_connect, "write", report_after=True)
_traced_send = _make_traced_socket_method(_real_socket_send, "write")
_traced_sendall = _make_traced_socket_method(_real_socket_sendall, "write")
_traced_sendto = _make_traced_socket_method(_real_socket_sendto, "write")
_traced_recv = _make_traced_socket_method(_real_socket_recv, "read")
_traced_recv_into = _make_traced_socket_method(_real_socket_recv_into, "read")
_traced_recvfrom = _make_traced_socket_method(_real_socket_recvfrom, "read")


# ---------------------------------------------------------------------------
# Layer 1: File open monkey-patching
# ---------------------------------------------------------------------------

_real_open = builtins.open


def _traced_open(*args: Any, **kwargs: Any) -> Any:
    result = _real_open(*args, **kwargs)
    reporter = get_io_reporter()
    if reporter is not None:
        # Determine the file path from args
        file_arg = args[0] if args else kwargs.get("file")
        if file_arg is not None and isinstance(file_arg, (str, bytes, os.PathLike)):
            path = os.fsdecode(file_arg)
            resource_id = _file_resource_id(path)
            # Determine read vs write from mode
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                reporter(resource_id, "write")
            else:
                reporter(resource_id, "read")
    return result


# ---------------------------------------------------------------------------
# Layer 1.5: sys.setprofile C-call detection
# ---------------------------------------------------------------------------

# Set of C functions we consider I/O-related for profiling detection.
# Identity comparison (``arg is func``) is used in the profile callback.
_SOCKET_WRITE_FUNCS: frozenset[Any] = frozenset(
    {
        socket.socket.send,
        socket.socket.sendall,
        socket.socket.sendto,
        socket.socket.connect,
    }
)
_SOCKET_READ_FUNCS: frozenset[Any] = frozenset(
    {
        socket.socket.recv,
        socket.socket.recv_into,
        socket.socket.recvfrom,
    }
)

# Collect qualnames for fallback matching (C builtins may not match by identity)
_SOCKET_WRITE_NAMES: frozenset[str] = frozenset(
    getattr(f, "__qualname__", getattr(f, "__name__", "")) for f in _SOCKET_WRITE_FUNCS
)
_SOCKET_READ_NAMES: frozenset[str] = frozenset(
    getattr(f, "__qualname__", getattr(f, "__name__", "")) for f in _SOCKET_READ_FUNCS
)


def make_io_profile_func(reporter: IOReporter) -> Callable[[Any, str, Any], None]:
    """Create a sys.setprofile callback that detects C-level I/O calls.

    The returned function should be installed with ``sys.setprofile()`` on
    each managed thread.  It coexists with ``sys.settrace`` without
    interference (profile fires for C calls, trace fires for opcodes).
    """

    def profile_func(frame: Any, event: str, arg: Any) -> None:
        if event != "c_call":
            return
        qualname = getattr(arg, "__qualname__", getattr(arg, "__name__", ""))
        if qualname in _SOCKET_WRITE_NAMES:
            # Try to get the socket object from the frame's locals
            # The first argument to a socket method is `self`
            resource_id = _guess_socket_resource_from_frame(frame)
            if resource_id is not None:
                reporter(resource_id, "write")
        elif qualname in _SOCKET_READ_NAMES:
            resource_id = _guess_socket_resource_from_frame(frame)
            if resource_id is not None:
                reporter(resource_id, "read")

    return profile_func


def _guess_socket_resource_from_frame(frame: Any) -> str | None:
    """Try to find a socket object in the frame's locals and get its resource ID."""
    # In a method call like sock.send(data), `self` is the socket
    local_self = frame.f_locals.get("self")
    if isinstance(local_self, socket.socket):
        return _socket_resource_id(local_self)
    # Fall back to searching locals for any socket
    for val in frame.f_locals.values():
        if isinstance(val, socket.socket):
            return _socket_resource_id(val)
    return None


# ---------------------------------------------------------------------------
# Patching / unpatching API
# ---------------------------------------------------------------------------

_io_patched = False


def patch_io() -> None:
    """Replace socket and open with traced versions.

    Call this before running managed threads.  Call :func:`unpatch_io` to
    restore originals.
    """
    global _io_patched  # noqa: PLW0603
    if _io_patched:
        return
    socket.socket.connect = _traced_connect  # type: ignore[assignment]
    socket.socket.send = _traced_send  # type: ignore[assignment]
    socket.socket.sendall = _traced_sendall  # type: ignore[assignment]
    socket.socket.sendto = _traced_sendto  # type: ignore[assignment]
    socket.socket.recv = _traced_recv  # type: ignore[assignment]
    socket.socket.recv_into = _traced_recv_into  # type: ignore[assignment]
    socket.socket.recvfrom = _traced_recvfrom  # type: ignore[assignment]
    builtins.open = _traced_open  # type: ignore[assignment]
    _io_patched = True


def unpatch_io() -> None:
    """Restore original socket and open implementations."""
    global _io_patched  # noqa: PLW0603
    if not _io_patched:
        return
    socket.socket.connect = _real_socket_connect  # type: ignore[assignment]
    socket.socket.send = _real_socket_send  # type: ignore[assignment]
    socket.socket.sendall = _real_socket_sendall  # type: ignore[assignment]
    socket.socket.sendto = _real_socket_sendto  # type: ignore[assignment]
    socket.socket.recv = _real_socket_recv  # type: ignore[assignment]
    socket.socket.recv_into = _real_socket_recv_into  # type: ignore[assignment]
    socket.socket.recvfrom = _real_socket_recvfrom  # type: ignore[assignment]
    builtins.open = _real_open  # type: ignore[assignment]
    _io_patched = False


def install_io_profile(reporter: IOReporter) -> Callable[[Any, str, Any], None]:
    """Create and install a sys.setprofile callback for the current thread.

    Returns the profile function so the caller can remove it later with
    ``sys.setprofile(None)``.
    """
    prof = make_io_profile_func(reporter)
    sys.setprofile(prof)
    return prof


def uninstall_io_profile() -> None:
    """Remove the current thread's sys.setprofile callback."""
    sys.setprofile(None)
