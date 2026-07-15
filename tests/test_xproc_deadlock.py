"""Cross-process row-lock deadlock detection (Phase 2b).

Two workers acquire two row locks in opposite order. Under the interleaving
where each holds one lock and then requests the other, the wait-for graph in the
reused DporScheduler detects the cycle and the coordinator reports a
deterministic deadlock — the same machinery that catches in-process SELECT FOR
UPDATE deadlocks, now spanning worker "processes".

Run in-process (ThreadLauncher) because SQLite has no SELECT FOR UPDATE; the
same path applies to real Postgres subprocesses (integration).
"""

from __future__ import annotations

from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher

ROW1 = "sql:accounts:id=1"
ROW2 = "sql:accounts:id=2"


def _locker(first: str, second: str):
    def worker(proxy) -> None:
        proxy.acquire_row_locks(0, [first])
        proxy.report_and_wait(None, 0)  # let the other worker grab its first lock
        proxy.acquire_row_locks(0, [second])
        proxy.report_and_wait(None, 0)
        proxy.release_row_locks(0)

    return worker


def test_detects_cross_worker_lock_cycle() -> None:
    # A merely redirected trace is not replay-exact and therefore fails
    # closed. Keep exploring so a later concrete lock cycle can supersede it.
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0, stop_on_first=False)
    result = coord.explore(
        worker_set=ThreadLauncher([_locker(ROW1, ROW2), _locker(ROW2, ROW1)]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "deadlock"
    assert result.failing_schedule is not None


def test_same_order_locking_fails_closed_without_claiming_deadlock() -> None:
    # Both workers take locks in the same order, so there is no cycle. Physical
    # lock redirection still makes the committed engine trace inexact, however,
    # and must not be certified as an exhaustive successful exploration.
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([_locker(ROW1, ROW2), _locker(ROW1, ROW2)]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "nondeterministic"
