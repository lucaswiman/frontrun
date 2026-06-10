"""Regression tests for row-lock engine blocking (finding 1).

When a thread S blocks on a DPOR row lock held by another thread H, the engine
must be told that S is blocked (``execution.block_thread(S)``), so the engine
never schedules S while it is waiting. Previously the row-lock path only set
``_row_lock_blocked`` and relied on a holder-substitution hack in
``_schedule_next`` that ran H's accesses *after* the engine had already
committed a step labelled with S — corrupting per-thread bookkeeping.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from frontrun._cooperative import real_condition, real_lock
from frontrun._dpor_core import RowLockRegistry
from frontrun.dpor import DporScheduler


class _FakeExecution:
    """Tracks which threads the engine considers blocked."""

    def __init__(self) -> None:
        self.blocked: set[int] = set()

    def block_thread(self, thread_id: int) -> None:
        self.blocked.add(thread_id)

    def unblock_thread(self, thread_id: int) -> None:
        self.blocked.discard(thread_id)


class _FakeEngine:
    def report_sync(self, *args: Any, **kwargs: Any) -> None:
        return None


def _make_host() -> Any:
    class RowLockHost:
        def __init__(self) -> None:
            self.deadlock_timeout = 1.0
            self._lock = real_lock()
            self._condition = real_condition(self._lock)
            self._engine_lock = real_lock()
            self._finished = False
            self._error: Exception | None = None
            self._current_thread: int | None = None
            self.execution = _FakeExecution()
            self.engine = _FakeEngine()
            self._row_lock_registry = RowLockRegistry()
            self._active_row_locks = self._row_lock_registry._active_row_locks
            self._thread_row_locks = self._row_lock_registry._task_row_locks
            self._row_lock_ids = self._row_lock_registry._row_lock_ids
            self._row_lock_blocked: dict[int, int] = {}

        acquire_row_locks = DporScheduler.acquire_row_locks
        release_row_locks = DporScheduler.release_row_locks
        _release_row_locks_unlocked = DporScheduler._release_row_locks_unlocked
        _row_lock_int_id = DporScheduler._row_lock_int_id

    return RowLockHost()


def test_blocked_thread_marked_blocked_in_engine() -> None:
    """While S waits for a row lock, the engine must see S as blocked."""
    host = _make_host()
    host.acquire_row_locks(0, ["sql:users:(('id', 42))"])

    started = threading.Event()
    acquired = threading.Event()

    def waiter() -> None:
        started.set()
        host.acquire_row_locks(1, ["sql:users:(('id', 42))"])
        acquired.set()

    t = threading.Thread(target=waiter)
    t.start()
    try:
        started.wait(timeout=2.0)
        # Let the waiter reach the blocking wait.
        time.sleep(0.2)

        assert not acquired.is_set(), "waiter should still be blocked"
        assert 1 in host.execution.blocked, "engine must see the waiting thread as blocked"
    finally:
        # Always release so the waiter unblocks and the thread can join,
        # even if an assertion above failed.
        host.release_row_locks(0)
        t.join(timeout=2.0)
    assert acquired.is_set(), "waiter should acquire after release"
    assert 1 not in host.execution.blocked, "engine must unblock the thread after acquire"


def test_blocked_thread_unblocked_on_timeout() -> None:
    """A thread is blocked during the wait and unblocked after a timeout."""
    host = _make_host()
    host.deadlock_timeout = 0.3
    host.acquire_row_locks(0, ["sql:users:(('id', 42))"])

    saw_blocked = threading.Event()
    done = threading.Event()

    def waiter() -> None:
        # The holder never releases, so this blocks then times out.
        host.acquire_row_locks(1, ["sql:users:(('id', 42))"])
        done.set()

    t = threading.Thread(target=waiter)
    t.start()
    # Poll for the engine-blocked state while the waiter is in its wait.
    for _ in range(50):
        if 1 in host.execution.blocked:
            saw_blocked.set()
            break
        time.sleep(0.01)

    t.join(timeout=2.0)
    assert saw_blocked.is_set(), "engine must see the thread blocked during the wait"
    assert done.is_set(), "waiter should return after timeout"
    assert 1 not in host.execution.blocked, "timed-out thread must be unblocked in the engine"
