# ruff: noqa: F403, F405
# pyright: reportUnusedClass=false

from __future__ import annotations

from ._shared import *
from ._shared import _make_object_key


class _PreloadBridge:
    """Routes I/O events from the LD_PRELOAD library to DPOR threads.

    The LD_PRELOAD library intercepts C-level ``send()``/``recv()`` calls
    (e.g. from psycopg2's libpq) and writes events to a pipe.  An
    :class:`~frontrun._preload_io.IOEventDispatcher` reads the pipe in a
    background thread and invokes :meth:`listener` for each event.

    This bridge maps OS thread IDs to DPOR logical thread IDs and buffers
    events in per-thread lists.  The DPOR scheduler drains these buffers
    at each scheduling point via :meth:`drain`.
    """

    def __init__(self, dispatcher: Any = None) -> None:
        self._lock = real_lock()
        self._tid_to_dpor: dict[int, int] = {}
        self._pending: dict[int, list[tuple[int, str, str, str | None, list[str] | None]]] = {}
        self._active = False
        self._dispatcher = dispatcher  # IOEventDispatcher (for poll())

    def register_thread(self, os_tid: int, dpor_id: int) -> None:
        """Map an OS thread ID to a DPOR logical thread ID."""
        with self._lock:
            self._tid_to_dpor[os_tid] = dpor_id
            self._pending.setdefault(dpor_id, [])
            self._active = True

    def unregister_thread(self, os_tid: int) -> None:
        """Remove an OS thread ID mapping."""
        with self._lock:
            self._tid_to_dpor.pop(os_tid, None)
            if not self._tid_to_dpor:
                self._active = False

    def clear(self) -> None:
        """Clear all mappings and pending events (between executions)."""
        with self._lock:
            self._tid_to_dpor.clear()
            self._pending.clear()
            self._active = False

    def listener(self, event: Any) -> None:
        """IOEventDispatcher callback — buffer the event for the right thread."""
        if not self._active:
            return
        # Skip close events — closing a file descriptor doesn't mutate the
        # external resource and creates many spurious conflict points that
        # force DPOR to explore uninteresting interleavings first.
        if event.kind == "close":
            return
        # Skip events to known SQL/Redis socket endpoints — the cursor/client
        # layer already reports at a higher granularity (table/row/key level).
        # Also skip socket events from permanently-suppressed SQL threads (covers
        # the race window where connect() events arrive before the endpoint is
        # registered).  Non-socket events (file I/O) always pass through even
        # for SQL threads, so DPOR can detect non-SQL conflicts.
        is_socket = event.resource_id.startswith("socket:")
        if is_sql_endpoint_suppressed(event.resource_id):
            return
        if is_socket and is_tid_suppressed(event.tid):
            return
        if is_socket and is_redis_tid_suppressed(event.tid):
            return
        with self._lock:
            dpor_id = self._tid_to_dpor.get(event.tid)
            if dpor_id is None:
                return
            # Map libc I/O operations to DPOR access kinds.  Using the
            # actual send/recv distinction (write/read) is critical: the
            # DPOR engine's ObjectState tracks per-thread latest-read and
            # latest-write separately.  If we treated all socket I/O as
            # "write", only the LAST write per thread would be tracked,
            # and early access positions (e.g. a SELECT recv) would be
            # overwritten by later ones (e.g. a COMMIT recv).  With
            # read/write distinction, DPOR iteratively explores wakeup tree
            # branches through the send/recv pairs to reach the critical interleaving.
            kind = "write" if event.kind == "write" else "read"
            obj_key = _make_object_key(hash(event.resource_id), event.resource_id)
            detail, call_chain = get_active_sql_io_context(event.tid)
            self._pending.setdefault(dpor_id, []).append((obj_key, kind, event.resource_id, detail, call_chain))

    def drain(self, dpor_id: int) -> list[tuple[int, str, str, str | None, list[str] | None]]:
        """Return and clear buffered events for a DPOR thread.

        Each item is ``(object_key, kind, resource_id, detail, call_chain)``.

        On free-threaded Python the background reader may not have
        processed pipe data yet, so we poll the dispatcher first to
        flush any pending bytes into listener callbacks.
        """
        if self._dispatcher is not None:
            self._dispatcher.poll()
        with self._lock:
            events = self._pending.get(dpor_id)
            if events:
                self._pending[dpor_id] = []
                return events
            return []


# ---------------------------------------------------------------------------
# DPOR Opcode Scheduler
# ---------------------------------------------------------------------------


_SENTINEL = object()
