"""DPOR row-lock acquire/release helpers for SQL interception.

Row locks are scheduler-level locks (in the DPOR engine) tracking which
rows a transaction holds. They are populated by ``_report_or_buffer``
inside ``_sql_transactions`` (which writes ``_io_tls._pending_row_locks``)
and drained here at the next scheduling point.

Kept separate from ``_sql_cursor.py`` so the DPOR-context glue is
isolated from cursor patching.
"""

from __future__ import annotations

from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._io_detection import tx_store

__all__ = ["_acquire_pending_row_locks", "_release_dpor_row_locks"]


def _acquire_pending_row_locks() -> None:
    """Drain pending row-lock resources from TLS and acquire them on the scheduler."""
    store = tx_store()
    lock_resources = getattr(store, "_pending_row_locks", None)
    if lock_resources:
        store._pending_row_locks = []
        lock_resources = list(dict.fromkeys(lock_resources))
        ctx = _get_dpor_context()
        if ctx is not None:
            acquired = ctx[0].acquire_row_locks(ctx[1], lock_resources)
            if acquired is None:
                acquired = lock_resources
            held = getattr(store, "_held_row_locks", None)
            if held is None:
                held = set()
                store._held_row_locks = held
            held.update(acquired)


def _release_dpor_row_locks() -> None:
    """Release any DPOR row locks held by the current thread."""
    store = tx_store()
    if hasattr(store, "_held_row_locks"):
        store._held_row_locks = set()
    ctx = _get_dpor_context()
    if ctx is not None:
        ctx[0].release_row_locks(ctx[1])
