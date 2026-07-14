"""LD_PRELOAD endpoint/thread suppression for SQL drivers.

The SQL interception layer reports DB I/O at table/row granularity. The
LD_PRELOAD bridge would otherwise also report the underlying socket
events for the same calls, which would be redundant and noisy. This
module owns the per-thread and per-endpoint suppression registries the
bridge listener consults.

Two suppression mechanisms exist:

* **Per-thread (transient)** — :func:`_suppress_endpoint_io` flips a
  thread-local + a TID set for the duration of a single ``execute()``
  call.  Used to drop socket events synchronously emitted while the
  SQL call is on the stack.
* **Per-endpoint (persistent)** — :func:`suppress_sql_endpoint` records
  the connection's socket peer (e.g. ``socket:127.0.0.1:5432``) so
  events that travel through the async pipe and are read after the
  context manager exits are still suppressed.

This module also tracks the most recent SQL statement seen on each
native thread, which the LD_PRELOAD bridge uses to render trace
context for socket I/O it does NOT suppress.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import sys
import threading
from collections.abc import Generator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._io_detection import _io_tls
from frontrun._trace_format import build_call_chain
from frontrun._tracing import should_trace_file as _should_trace_file

try:
    from frontrun._sql_params import resolve_parameters
except ImportError:

    def resolve_parameters(sql: str, parameters: Any, paramstyle: str) -> str:  # type: ignore[misc]
        return sql


__all__ = [
    "_set_active_sql_io_context",
    "_suppress_endpoint_io",
    "_suppress_lock",
    "_suppress_tids",
    "clear_permanent_suppressions",
    "get_active_sql_io_context",
    "is_sql_endpoint_suppressed",
    "is_sql_write_suppressed",
    "is_tid_suppressed",
    "suppress_sql_endpoint",
    "suppress_sql_write",
    "suppress_tid_permanently",
    "unsuppress_tid_permanently",
]


# ---------------------------------------------------------------------------
# Suppression infrastructure
# ---------------------------------------------------------------------------

# OS thread IDs currently inside a patched execute call.
# The LD_PRELOAD bridge listener checks this to skip endpoint-level reports.
_suppress_tids: set[int] = set()
_suppress_lock = _rt.lock()  # Real lock (not cooperative)
_ACTIVE_SQL_IO_CONTEXTS: dict[int, tuple[str | None, list[str] | None]] = {}

# Persistent suppression: SQL socket endpoints whose LD_PRELOAD events
# should be suppressed because the SQL layer reports at a higher granularity
# (table/row level).  Keyed by resource_id (e.g. "socket:127.0.0.1:5432",
# "socket:unix:/var/run/postgresql/.s.PGSQL.5432").
#
# The temporary _suppress_endpoint_io() context manager has a timing
# problem: LD_PRELOAD events travel through an async pipe, so by the time
# they're read the context has exited.  Permanent endpoint suppression
# persists across the entire DPOR execution.
_suppressed_sql_endpoints: set[str] = set()
_suppressed_sql_writes: set[str] = set()

_permanently_suppressed_tids: set[int] = set()


def _summarize_sql_for_trace(operation: Any, parameters: Any, paramstyle: str) -> str | None:
    """Return a short SQL summary suitable for trace output."""
    if not isinstance(operation, str):
        return None
    sql = operation
    try:
        if parameters is not None:
            sql = resolve_parameters(operation, parameters, paramstyle)
    except Exception:
        sql = operation
    sql = " ".join(sql.split())
    if len(sql) > 160:
        sql = f"{sql[:157]}..."
    return f"SQL: {sql}"


def _current_user_call_chain() -> list[str] | None:
    """Return a best-effort call chain rooted at the first traced user frame."""
    frame = sys._getframe(1)
    while frame is not None and not _should_trace_file(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None:
        return None
    return build_call_chain(frame, filter_fn=_should_trace_file)


def _set_active_sql_io_context(operation: Any, parameters: Any, paramstyle: str) -> None:
    """Remember the current SQL statement for C-level socket I/O trace rendering."""
    tid = threading.get_native_id()
    summary = _summarize_sql_for_trace(operation, parameters, paramstyle)
    chain = _current_user_call_chain()
    with _suppress_lock:
        _ACTIVE_SQL_IO_CONTEXTS[tid] = (summary, chain)


def get_active_sql_io_context(tid: int) -> tuple[str | None, list[str] | None]:
    """Return the most recent SQL trace context for a native thread id."""
    with _suppress_lock:
        return _ACTIVE_SQL_IO_CONTEXTS.get(tid, (None, None))


@contextlib.contextmanager
def _suppress_endpoint_io() -> Generator[None, None, None]:
    """Temporarily suppress endpoint-level I/O for the current thread."""
    tid = threading.get_native_id()
    _io_tls._sql_suppress = True
    with _suppress_lock:
        _suppress_tids.add(tid)
    try:
        yield
    finally:
        with _suppress_lock:
            _suppress_tids.discard(tid)
        _io_tls._sql_suppress = False


def is_tid_suppressed(tid: int) -> bool:
    """Check if a thread ID is currently suppressed (for LD_PRELOAD bridge)."""
    with _suppress_lock:
        return tid in _suppress_tids or tid in _permanently_suppressed_tids


def _socket_resource_id_from_peer(peer: object) -> str | None:
    """Return the resource identity emitted by the preload socket interceptor."""
    if isinstance(peer, str):
        # Unix domain socket — peer is a path string.
        return f"socket:unix:{peer}" if peer else None
    if not isinstance(peer, tuple) or len(peer) < 2:
        return None

    host, port = peer[:2]
    if not isinstance(host, str):
        return None
    if ":" in host:
        # Rust formats the raw 16 IPv6 address bytes as eight zero-padded,
        # lowercase groups.  getpeername() normally compresses them (``::1``),
        # so canonicalize here to keep persistent endpoint suppression exact.
        try:
            host = f"[{ipaddress.IPv6Address(host.split('%', 1)[0]).exploded}]"
        except ipaddress.AddressValueError:
            return None
    return f"socket:{host}:{port}"


def _socket_resource_id_from_fd(fd: int) -> str | None:
    """Derive the LD_PRELOAD-style resource_id from a socket file descriptor.

    Returns e.g. ``"socket:127.0.0.1:5432"`` for TCP or
    ``"socket:unix:/var/run/postgresql/.s.PGSQL.5432"`` for Unix domain sockets.
    Returns ``None`` if the fd is not a connected socket.
    """
    import socket as _socket

    # Duplicate the fd so we don't accidentally close the connection's socket
    # when the temporary socket object is garbage-collected.
    dup_fd = os.dup(fd)
    try:
        sock = _socket.socket(fileno=dup_fd)
        try:
            peer = sock.getpeername()
        except (OSError, ValueError):
            return None
        finally:
            sock.detach()  # detach so sock.__del__ doesn't close dup_fd
    finally:
        os.close(dup_fd)
    return _socket_resource_id_from_peer(peer)


def _resource_id_from_connection(conn: Any) -> str | None:
    """Extract the LD_PRELOAD-compatible socket resource_id from a DB connection."""
    fileno_fn = getattr(conn, "fileno", None)
    if fileno_fn is None:
        return None
    try:
        fd = fileno_fn()
    except Exception:
        return None
    if not isinstance(fd, int) or fd < 0:
        return None
    return _socket_resource_id_from_fd(fd)


def suppress_sql_endpoint(conn: Any) -> None:
    """Register a SQL connection's socket endpoint for LD_PRELOAD suppression."""
    resource_id = _resource_id_from_connection(conn)
    if resource_id is not None:
        with _suppress_lock:
            _suppressed_sql_endpoints.add(resource_id)


def suppress_sql_write(operation: Any, parameters: Any = None, paramstyle: str = "format") -> None:
    """Register raw PostgreSQL wire SQL text that the SQL layer already models."""
    if not isinstance(operation, str):
        return
    candidates = {operation}
    try:
        candidates.add(resolve_parameters(operation, parameters, paramstyle))
    except Exception:
        pass
    with _suppress_lock:
        _suppressed_sql_writes.update(candidates)


def suppress_tid_permanently(tid: int | None = None) -> None:
    """Mark a thread as permanently suppressed for LD_PRELOAD events.

    .. deprecated::
        Prefer :func:`suppress_sql_endpoint` which suppresses by socket
        endpoint rather than by thread, so non-SQL file I/O remains visible.
        Kept for the connect-time path where the connection is not yet
        established and we must suppress by thread temporarily.
    """
    if tid is None:
        tid = threading.get_native_id()
    with _suppress_lock:
        _permanently_suppressed_tids.add(tid)


def unsuppress_tid_permanently(tid: int | None = None) -> None:
    """Remove connect-time native-thread suppression after endpoint discovery."""
    if tid is None:
        tid = threading.get_native_id()
    with _suppress_lock:
        _permanently_suppressed_tids.discard(tid)


def is_sql_endpoint_suppressed(resource_id: str) -> bool:
    """Check if a resource_id matches a known SQL socket endpoint."""
    with _suppress_lock:
        return resource_id in _suppressed_sql_endpoints


def is_sql_write_suppressed(resource_id: str) -> bool:
    """Check if a raw ``sql_write`` resource has a SQL-level model."""
    with _suppress_lock:
        return resource_id in _suppressed_sql_writes


def clear_permanent_suppressions() -> None:
    """Clear all permanent suppressions (between DPOR executions)."""
    with _suppress_lock:
        _permanently_suppressed_tids.clear()
        _suppressed_sql_endpoints.clear()
        _suppressed_sql_writes.clear()
