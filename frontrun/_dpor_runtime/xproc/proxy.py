"""Worker-side remote scheduler for cross-process DPOR exploration.

A :class:`SchedulerProxy` stands in for the in-process ``DporScheduler`` inside
a spawned worker process. The SQL/Redis interception layers fetch it from
thread-local context (``set_dpor_scheduler``) and call the *same* methods they
call on the real scheduler; each call is forwarded to the coordinator over a
socket and blocks until the coordinator grants the turn. Workers therefore run
no opcode tracing — only the external-access patches are active, and between
accesses the worker's own code runs uncontrolled (independent by construction,
since separate processes share no Python memory).

This is the worker half of Phase 1 (``ideas/cross_process_exploration.md``),
scoped to the SQL hot path:

* :meth:`report_and_wait` — force a scheduling point at a SQL statement.
* :meth:`acquire_row_locks` / :meth:`release_row_locks` — ``SELECT FOR UPDATE``
  / write row locks, keyed by the same ``sql:<table>:<pred>`` resource strings
  the in-process scheduler uses.
* :meth:`io_report` — the io-reporter callable (installed via
  ``set_io_reporter``) that funnels ``(resource_id, kind)`` access reports.

Redis (``before_io`` / ``after_io``) and async (``pause``) are Phase 2.
"""

from __future__ import annotations

import socket
from typing import Any

from frontrun._deadlock import SchedulerAbort

from . import protocol as proto


class SchedulerProxy:
    """Forwards the interception layer's scheduler calls to the coordinator."""

    # Worker processes deliberately scrub LD_PRELOAD, so SQL statements the
    # semantic parser cannot resolve need a conservative modeled fallback.
    requires_semantic_io_fallback = True

    def __init__(self, sock: socket.socket, worker_id: int) -> None:
        self._sock = sock
        self._worker_id = worker_id
        # Latched once the coordinator sends ABORT (exploration finished or a
        # peer failed). Further scheduling points then short-circuit instead of
        # sending into a socket the coordinator has stopped reading.
        self._aborted = False

    def hello(self) -> None:
        """Announce this worker's id to the coordinator (first frame after connect)."""
        proto.send_msg(self._sock, {"t": proto.HELLO, "w": self._worker_id})

    def reset(self) -> None:
        """Clear the abort latch so a reused worker can run the next iteration."""
        self._aborted = False

    # --- io-reporter callable: installed via set_io_reporter(proxy.io_report) ---

    def io_report(self, resource_id: str, kind: str) -> None:
        """Report one external access. Fire-and-forget; never blocks the worker."""
        if self._aborted:
            return
        proto.send_msg(self._sock, {"t": proto.ACCESS, "w": self._worker_id, "rid": resource_id, "kind": kind})

    # --- scheduler interface used by the SQL interception layer ---

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Force a scheduling point; block until granted. ``False`` once aborted.

        ``frame`` is always ``None`` on this path and ``thread_id`` equals this
        worker's id; both are accepted only to match the in-process signature.
        """
        if self._aborted:
            return False
        proto.send_msg(self._sock, {"t": proto.REPORT_AND_WAIT, "w": self._worker_id})
        return self._await_grant()

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Block until *resource_ids* can be held. Raise ``SchedulerAbort`` if aborted."""
        if self._aborted:
            raise SchedulerAbort("cross-process scheduler aborted")
        proto.send_msg(self._sock, {"t": proto.ACQUIRE_LOCKS, "w": self._worker_id, "res": list(resource_ids)})
        if not self._await_grant():
            raise SchedulerAbort("cross-process scheduler aborted while acquiring row locks")

    def release_row_locks(self, thread_id: int, resources: list[str] | None = None) -> None:
        """Drop selected row locks, or all locks on COMMIT/ROLLBACK. Fire-and-forget."""
        if self._aborted:
            return
        msg: dict[str, Any] = {"t": proto.RELEASE_LOCKS, "w": self._worker_id}
        if resources is not None:
            msg["res"] = list(resources)
        proto.send_msg(self._sock, msg)

    def before_io(self, thread_id: int, resource_id: str) -> bool:
        """Enter a two-phase IO boundary (Redis); block until granted the turn.

        Returns ``True`` on GRANT. ``False`` means the coordinator aborted the
        run: the caller must not perform the real command (the Redis envelope
        raises ``SchedulerAbort``, matching the SQL path).
        """
        if self._aborted:
            return False
        proto.send_msg(self._sock, {"t": proto.BEFORE_IO, "w": self._worker_id, "rid": resource_id})
        return self._await_grant()

    def after_io(self, thread_id: int, resource_id: str) -> None:
        """Exit the IO boundary and release the turn. Fire-and-forget."""
        if self._aborted:
            return
        proto.send_msg(self._sock, {"t": proto.AFTER_IO, "w": self._worker_id, "rid": resource_id})

    # --- worker lifecycle (called by the worker bootstrap, not interception) ---

    def mark_done(self) -> None:
        """Tell the coordinator this worker has finished."""
        if self._aborted:
            return
        proto.send_msg(self._sock, {"t": proto.DONE, "w": self._worker_id})

    def report_error(self, message: str) -> None:
        """Tell the coordinator this worker raised an unhandled exception."""
        if self._aborted:
            return
        proto.send_msg(self._sock, {"t": proto.ERROR, "w": self._worker_id, "msg": message})

    def _await_grant(self) -> bool:
        """Block on the coordinator's reply. ``True`` only for GRANT.

        Anything else — ABORT, EOF, or an out-of-band control frame such as
        ITER_START/SHUTDOWN that arrives because a prior handshake was
        abandoned — latches the abort so a reused worker can never mistake a
        control frame for a grant and run its next statement off by one frame.
        """
        msg = proto.recv_msg(self._sock)
        if msg is not None and msg.get("t") == proto.GRANT:
            return True
        self._aborted = True
        return False
