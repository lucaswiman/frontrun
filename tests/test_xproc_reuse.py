"""Persistent worker reuse (Phase 3) — functional coverage.

Reuse mode connects each worker once and re-runs the target per iteration
(ITER_START / SHUTDOWN) instead of respawning, avoiding spawn cost. These tests
drive it in-process via PersistentThreadLauncher so the persistent protocol and
per-iteration reset are exercised without subprocesses, and confirm reuse
reaches the same verdicts and execution counts as spawn-per-iteration.
"""

from __future__ import annotations

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
