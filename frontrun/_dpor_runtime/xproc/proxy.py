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
from collections.abc import Generator
from contextlib import contextmanager
from threading import get_ident
from typing import Any

from frontrun import _real_threading as _rt
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
        # The proxy models ONE logical worker: its frames must form a single
        # sequential stream, and GRANT replies carry no addressee, so two
        # threads blocked in recv on the same socket would be woken
        # arbitrarily — silent nondeterminism. Concurrent use is rejected
        # loudly via this non-blocking guard; *sequential* cross-thread
        # hand-off (spawn a thread for one statement, join it, continue)
        # remains supported. A real lock, never a patched one: the guard runs
        # inside interception machinery.
        self._use_lock = _rt.lock()
        # OS thread owning the complete semantic operation: ACCESS reports,
        # grant, physical I/O, and explicit completion.
        self._operation_owner: int | None = None
        # Concurrent use is raised in the offending thread, which user code may
        # join without propagating that exception.  Latch it so the bootstrap
        # thread reports ERROR instead of a false DONE after the target returns.
        self._fatal_error: str | None = None

    @contextmanager
    def _exclusive(self, *, blocking: bool = False) -> Generator[None, None, None]:
        if not self._use_lock.acquire(blocking=blocking):
            raise self._concurrent_use_error()
        try:
            yield
        finally:
            self._use_lock.release()

    def _concurrent_use_error(self) -> RuntimeError:
        message = (
            "SchedulerProxy used concurrently from multiple threads: cross-process "
            "workers must perform scheduled external accesses (SQL/Redis statements) "
            "from one thread at a time — the coordinator schedules each worker "
            "process as a single logical actor"
        )
        self._fatal_error = message
        return RuntimeError(message)

    @contextmanager
    def _wire_access(self, *, blocking: bool = False) -> Generator[None, None, None]:
        """Serialize a frame exchange, reusing operation-scoped ownership."""
        if self._operation_owner == get_ident():
            yield
            return
        with self._exclusive(blocking=blocking):
            yield

    def begin_external_operation(self) -> None:
        """Own this one-actor proxy until :meth:`end_external_operation`."""
        ident = get_ident()
        if self._operation_owner == ident:
            raise RuntimeError("nested SchedulerProxy external operation")
        if not self._use_lock.acquire(blocking=False):
            raise self._concurrent_use_error()
        self._operation_owner = ident

    def end_external_operation(self) -> None:
        """Release operation-scoped ownership from the owning OS thread."""
        if self._operation_owner != get_ident():
            raise RuntimeError("SchedulerProxy external operation ended by a non-owner thread")
        self._operation_owner = None
        self._use_lock.release()

    def hello(self) -> None:
        """Announce this worker's id to the coordinator (first frame after connect)."""
        proto.send_msg(self._sock, {"t": proto.HELLO, "w": self._worker_id})

    def reset(self) -> None:
        """Clear the abort latch so a reused worker can run the next iteration."""
        self._aborted = False
        self._fatal_error = None

    @property
    def fatal_error(self) -> str | None:
        """Concurrent-use failure that must be reported by the bootstrap thread."""
        return self._fatal_error

    # --- io-reporter callable: installed via set_io_reporter(proxy.io_report) ---

    def io_report(self, resource_id: str, kind: str) -> None:
        """Report one external access. Fire-and-forget; never blocks the worker."""
        if self._aborted:
            return
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.ACCESS, "w": self._worker_id, "rid": resource_id, "kind": kind})

    # --- scheduler interface used by the SQL interception layer ---

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Force a scheduling point; block until granted. ``False`` once aborted.

        ``frame`` is always ``None`` on this path and ``thread_id`` equals this
        worker's id; both are accepted only to match the in-process signature.
        """
        if self._aborted:
            return False
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.REPORT_AND_WAIT, "w": self._worker_id})
            return self._await_grant()

    def before_sync_retry(self, thread_id: int) -> bool:
        """Acquire and hold the coordinator's SQL/sync turn."""
        return self.report_and_wait(None, thread_id)

    def after_sync_retry(self, thread_id: int) -> None:
        """Explicitly release the coordinator's held SQL/sync turn."""
        if self._aborted:
            return
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.AFTER_SYNC, "w": self._worker_id})

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Block until *resource_ids* can be held. Raise ``SchedulerAbort`` if aborted."""
        if self._aborted:
            raise SchedulerAbort("cross-process scheduler aborted")
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.ACQUIRE_LOCKS, "w": self._worker_id, "res": list(resource_ids)})
            granted = self._await_grant()
        if not granted:
            raise SchedulerAbort("cross-process scheduler aborted while acquiring row locks")

    def release_row_locks(self, thread_id: int, resources: list[str] | None = None) -> None:
        """Drop selected row locks, or all locks on COMMIT/ROLLBACK. Fire-and-forget."""
        if self._aborted:
            return
        msg: dict[str, Any] = {"t": proto.RELEASE_LOCKS, "w": self._worker_id}
        if resources is not None:
            msg["res"] = list(resources)
        with self._wire_access():
            proto.send_msg(self._sock, msg)

    def before_io(self, thread_id: int, resource_id: str) -> bool:
        """Enter a two-phase IO boundary (Redis); block until granted the turn.

        Returns ``True`` on GRANT. ``False`` means the coordinator aborted the
        run: the caller must not perform the real command (the Redis envelope
        raises ``SchedulerAbort``, matching the SQL path).
        """
        if self._aborted:
            return False
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.BEFORE_IO, "w": self._worker_id, "rid": resource_id})
            return self._await_grant()

    def after_io(self, thread_id: int, resource_id: str) -> None:
        """Exit the IO boundary and release the turn. Fire-and-forget."""
        if self._aborted:
            return
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.AFTER_IO, "w": self._worker_id, "rid": resource_id})

    # --- worker lifecycle (called by the worker bootstrap, not interception) ---

    def mark_done(self) -> None:
        """Tell the coordinator this worker has finished."""
        if self._aborted:
            return
        with self._wire_access():
            proto.send_msg(self._sock, {"t": proto.DONE, "w": self._worker_id})

    def report_error(self, message: str) -> None:
        """Tell the coordinator this worker raised an unhandled exception."""
        if self._aborted:
            return
        # A child thread may still be unwinding the scheduling call that
        # latched this error.  Wait for that real lock rather than losing the
        # coordinator-visible ERROR to a second concurrent-use exception.
        with self._wire_access(blocking=True):
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
