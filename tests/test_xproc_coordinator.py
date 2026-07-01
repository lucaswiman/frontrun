"""Functional tests for the cross-process exploration coordinator.

These drive the *real* coordinator, protocol, and ``SchedulerProxy`` over a
real ``AF_UNIX`` listener, but run the workers as in-process threads (via
``ThreadLauncher``) contending on a shared in-memory dict standing in for the
database. That exercises 100% of the coordinator's scheduling, row-lock
arbitration, and invariant machinery deterministically, without spawning
subprocesses or needing a real DB (that is the e2e layer).
"""

from __future__ import annotations

import threading

from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher


class _DB:
    """A trivially-shared 'database': one integer balance."""

    def __init__(self) -> None:
        self.balance = 0

    def reset(self) -> None:
        self.balance = 0


def _incrementer_without_locks(db: _DB):
    """Read-modify-write with a scheduling point before each access, no locking.

    Models the classic lost-update: SELECT balance; balance += 100; UPDATE.
    """

    def worker(proxy) -> None:
        proxy.report_and_wait(None, 0)  # scheduling point before the read
        current = db.balance
        proxy.io_report("sql:accounts:id=1", "read")
        proxy.report_and_wait(None, 0)  # scheduling point before the write
        db.balance = current + 100
        proxy.io_report("sql:accounts:id=1", "write")

    return worker


def _incrementer_with_row_lock(db: _DB):
    """Same increment, but guarded by a SELECT FOR UPDATE row lock."""

    def worker(proxy) -> None:
        proxy.acquire_row_locks(0, ["sql:accounts:id=1"])
        proxy.report_and_wait(None, 0)
        current = db.balance
        proxy.io_report("sql:accounts:id=1", "read")
        proxy.report_and_wait(None, 0)
        db.balance = current + 100
        proxy.io_report("sql:accounts:id=1", "write")
        proxy.release_row_locks(0)

    return worker


def test_finds_lost_update_race() -> None:
    db = _DB()
    coord = CrossProcessCoordinator(num_workers=2)
    worker = _incrementer_without_locks(db)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
        max_iterations=100,
    )
    assert not result.ok
    assert result.failing_schedule is not None
    # The invariant must actually have been violated (a real lost update),
    # not merely a worker crash.
    assert result.failure_kind == "invariant"


def test_row_lock_prevents_lost_update() -> None:
    db = _DB()
    coord = CrossProcessCoordinator(num_workers=2)
    worker = _incrementer_with_row_lock(db)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
        max_iterations=100,
    )
    assert result.ok, f"unexpected failure: {result.failure!r} at {result.failing_schedule!r}"
    assert result.iterations >= 1


def test_explores_all_interleavings_and_reports_count() -> None:
    # Two workers with two scheduling points each and no locking: the number of
    # interleavings is C(4,2) = 6. Every one must be explored when the invariant
    # never fails (make it always true so exploration runs to exhaustion).
    db = _DB()
    coord = CrossProcessCoordinator(num_workers=2)
    worker = _incrementer_without_locks(db)
    seen: list[int] = []
    lock = threading.Lock()

    def record_true() -> bool:
        with lock:
            seen.append(db.balance)
        return True

    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=record_true,
        max_iterations=100,
    )
    assert result.ok
    assert result.exhausted
    assert result.iterations == 6


def test_unsupported_before_io_frame_is_reported_not_hung() -> None:
    # The exhaustive coordinator does not speak Redis's two-phase before_io.
    # It must surface an unexpected frame as a worker error rather than swallow
    # it and leave the worker blocked awaiting a grant.
    def redis_style(proxy) -> None:
        proxy.before_io(0, "redis:GET:key")

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([redis_style]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "unsupported frame" in (result.failure or "")


def test_reports_worker_error() -> None:
    db = _DB()

    def boom(proxy) -> None:
        proxy.report_and_wait(None, 0)
        raise ValueError("worker blew up")

    coord = CrossProcessCoordinator(num_workers=1)
    result = coord.explore(
        worker_set=ThreadLauncher([boom]),
        setup=db.reset,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "worker blew up" in (result.failure or "")
