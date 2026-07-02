"""Persistent worker reuse (Phase 3) — functional coverage.

Reuse mode connects each worker once and re-runs the target per iteration
(ITER_START / SHUTDOWN) instead of respawning, avoiding spawn cost. These tests
drive it in-process via PersistentThreadLauncher so the persistent protocol and
per-iteration reset are exercised without subprocesses, and confirm reuse
reaches the same verdicts and execution counts as spawn-per-iteration.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.worker import PersistentThreadLauncher, ThreadLauncher


class _DB:
    def __init__(self) -> None:
        self.balance = 0

    def reset(self) -> None:
        self.balance = 0


def _rmw_worker(db: _DB):
    def worker(proxy) -> None:
        proxy.report_and_wait(None, 0)
        current = db.balance
        proxy.io_report("sql:accounts:id=1", "read")
        proxy.report_and_wait(None, 0)
        db.balance = current + 100
        proxy.io_report("sql:accounts:id=1", "write")

    return worker


def test_reuse_finds_lost_update() -> None:
    db = _DB()
    worker = _rmw_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, reuse_workers=True)
    result = coord.explore(
        worker_set=PersistentThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None


def test_reuse_matches_spawn_execution_count() -> None:
    # Reuse must explore the same DPOR search as spawn-per-iteration.
    db_spawn = _DB()
    spawn = DporCrossProcessCoordinator(num_workers=2, stop_on_first=False).explore(
        worker_set=ThreadLauncher([_rmw_worker(db_spawn), _rmw_worker(db_spawn)]),
        setup=db_spawn.reset,
        invariant=lambda: True,
    )
    db_reuse = _DB()
    reuse = DporCrossProcessCoordinator(num_workers=2, stop_on_first=False, reuse_workers=True).explore(
        worker_set=PersistentThreadLauncher([_rmw_worker(db_reuse), _rmw_worker(db_reuse)]),
        setup=db_reuse.reset,
        invariant=lambda: True,
    )
    assert spawn.exhausted and reuse.exhausted
    assert reuse.iterations == spawn.iterations


def test_reuse_stops_safely_on_deadlock() -> None:
    # A deadlock aborts a worker mid-iteration, leaving its persistent socket at
    # an unknown frame boundary. Reuse must report the deadlock and stop rather
    # than send the next ITER_START into a poisoned socket (desync) or hang.
    row1, row2 = "sql:acct:id=1", "sql:acct:id=2"

    def locker(first: str, second: str):
        def worker(proxy) -> None:
            proxy.acquire_row_locks(0, [first])
            proxy.report_and_wait(None, 0)
            proxy.acquire_row_locks(0, [second])
            proxy.report_and_wait(None, 0)
            proxy.release_row_locks(0)

        return worker

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0, reuse_workers=True, stop_on_first=False)
    result = coord.explore(
        worker_set=PersistentThreadLauncher([locker(row1, row2), locker(row2, row1)]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "deadlock"


def test_reuse_no_state_leak_across_iterations() -> None:
    # A safe atomic increment must hold across every reused iteration; a leak
    # would surface as balance > 200 (extra increments from a prior run).
    import threading

    db = _DB()
    lock = threading.Lock()

    def atomic(proxy) -> None:
        proxy.report_and_wait(None, 0)
        with lock:
            db.balance += 100
        proxy.io_report("sql:accounts:id=1", "write")

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, reuse_workers=True)
    result = coord.explore(
        worker_set=PersistentThreadLauncher([atomic, atomic]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert result.ok, f"unexpected {result.failure_kind}: {result.failure!r}"
    assert result.exhausted


# --- Fix 2: a serialisation failure in worker_set.launch must surface as a
# structured worker_error result, not escape as a bare exception. ---


class _RaisingLauncher:
    """Fake WorkerSet whose ``launch`` fails the way ``_dumps_worker`` would."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def launch(self, targets: Any) -> Any:
        raise self._exc

    def join(self, handles: Any, timeout: float) -> list[Any]:
        return []


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "exc",
    [TypeError("cannot pickle <thread.lock>"), ImportError("dill is required for subprocess workers")],
)
def test_serialization_failure_returns_worker_error(reuse: bool, exc: Exception) -> None:
    coord = DporCrossProcessCoordinator(num_workers=2, reuse_workers=reuse)
    result = coord.explore(
        worker_set=_RaisingLauncher(exc),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert result.exhausted is False
    # The clear message must still be surfaced, not swallowed.
    assert str(exc) in (result.failure or "")


# --- Fix 1: a relay thread still alive after its join budget must raise a loud
# TimeoutError (converting a silent concurrent-engine data race into a catchable
# failure) instead of letting the coordinator move on. ---


class _FakeScheduler:
    """Minimal stand-in exposing only what ``_relay_loop`` touches."""

    def __init__(self) -> None:
        self.engine = None
        self.execution = None
        self._lock_depth_by_thread: dict[int, int] = {}
        self._pending_io_by_thread: dict[int, list[Any]] = {}
        self.done: list[int] = []

    def mark_done(self, worker_id: int) -> None:
        self.done.append(worker_id)


def test_drive_relays_raises_on_hung_relay() -> None:
    coord = DporCrossProcessCoordinator(num_workers=1)
    # Force a non-blocking join budget so the still-alive relay is detected at
    # once rather than after the default deadlock_timeout*2+10s. Thread.join
    # clamps negative timeouts to 0, so join_budget == 0 makes the join a
    # non-blocking liveness check.
    coord.deadlock_timeout = -5.0

    peer, relay_end = socket.socketpair()
    scheduler = _FakeScheduler()
    try:
        # The relay blocks forever in recv_msg (peer never sends), so it is
        # still alive after the zero-budget join -> loud TimeoutError.
        with pytest.raises(TimeoutError, match="did not terminate"):
            coord._drive_relays(scheduler, {0: relay_end}, [], {}, set())
    finally:
        # Unblock the hung daemon relay so it can unwind.
        peer.close()
        relay_end.close()
