"""Functional tests for the engine-driven (DPOR) cross-process coordinator.

Drives the real DporCrossProcessCoordinator (Rust engine + DporScheduler +
relays) over a real AF_UNIX socket, with workers run in-process as threads
(ThreadLauncher) contending on a shared in-memory dict. This exercises the
relay/engine integration deterministically without spawning subprocesses.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import pytest

from frontrun._dpor_runtime.xproc import protocol as proto
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator, _relay_loop
from frontrun._dpor_runtime.xproc.launch import WorkerSerializationError
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher


class _DB:
    def __init__(self) -> None:
        self.balance = 0

    def reset(self) -> None:
        self.balance = 0


def _rmw_worker(db: _DB):
    """Read-modify-write with a scheduling point before each access (no locks)."""

    def worker(proxy) -> None:
        proxy.report_and_wait(None, 0)
        current = db.balance
        proxy.io_report("sql:accounts:id=1", "read")
        proxy.report_and_wait(None, 0)
        db.balance = current + 100
        proxy.io_report("sql:accounts:id=1", "write")

    return worker


def _row_locked_worker(db: _DB):
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


def test_dpor_finds_lost_update() -> None:
    db = _DB()
    worker = _rmw_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None


def test_dpor_row_lock_prevents_lost_update() -> None:
    db = _DB()
    worker = _row_locked_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert result.ok, f"unexpected failure {result.failure!r} at {result.failing_schedule!r}"


def test_dpor_reports_worker_error() -> None:
    db = _DB()

    def boom(proxy) -> None:
        proxy.report_and_wait(None, 0)
        raise ValueError("kaboom")

    coord = DporCrossProcessCoordinator(num_workers=1, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([boom]),
        setup=db.reset,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "kaboom" in (result.failure or "")


def test_dpor_reports_worker_disconnect_after_hello() -> None:
    # A worker can connect and then die before DONE/ERROR (os._exit, SIGKILL,
    # segfault, etc.). That must be a worker failure, not a successful run.
    def disconnect(proxy) -> None:
        proxy._sock.close()

    coord = DporCrossProcessCoordinator(num_workers=1, deadlock_timeout=0.2)
    result = coord.explore(
        worker_set=ThreadLauncher([disconnect]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "disconnected" in (result.failure or "")


def test_dpor_stop_on_first_false_still_reports_race() -> None:
    # Regression: with stop_on_first=False the coordinator must still report the
    # invariant violation, not silently explore to exhaustion and return ok.
    db = _DB()
    worker = _rmw_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, stop_on_first=False)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None
    assert result.exhausted  # whole space explored, yet the failure is still surfaced


def _unlocked_write_then_lock_worker(db: _DB):
    """RMW whose unlocked write is reported just before acquiring an unrelated row lock.

    Mirrors the ordering real SQL interception produces: an access is reported
    after its statement executes (e.g. a post-INSERT indexical write) and the
    NEXT statement is ``SELECT ... FOR UPDATE``. The write sits in the relay's
    ``pending_io`` when ACQUIRE_LOCKS runs, so a pre-fix relay attributes it
    inside the (later) critical section.
    """

    def worker(proxy) -> None:
        proxy.report_and_wait(None, 0)
        current = db.balance
        proxy.io_report("sql:balance:id=1", "read")
        proxy.report_and_wait(None, 0)
        db.balance = current + 100
        proxy.io_report("sql:balance:id=1", "write")  # buffered; no flush before acquire
        proxy.acquire_row_locks(0, ["sql:acct:id=1"])
        proxy.report_and_wait(None, 0)
        proxy.release_row_locks(0)

    return worker


def test_dpor_flushes_pending_io_before_row_lock_acquire() -> None:
    # Regression: an unlocked write reported just before ACQUIRE_LOCKS must be
    # attributed OUTSIDE the (later) critical section. Otherwise two workers'
    # unlocked writes look lock-synchronized, DPOR prunes the racing
    # interleaving, and the lost update is missed (false ok=True / exhausted).
    #
    # Pre-fix this false negative occurs on ~29/30 runs (relay-timing dependent),
    # so assert the violation is found across several fresh searches: pre-fix at
    # least one search returns ok=True (fails), post-fix every search finds it.
    for _ in range(5):
        db = _DB()
        worker = _unlocked_write_then_lock_worker(db)
        coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, stop_on_first=False)
        result = coord.explore(
            worker_set=ThreadLauncher([worker, worker]),
            setup=db.reset,
            invariant=lambda: db.balance == 200,
        )
        assert not result.ok, "DPOR missed the lost update (pre-lock write attributed inside the lock)"
        assert result.failure_kind == "invariant"


def test_dpor_reduces_interleavings_vs_exhaustive() -> None:
    # For two read-modify-write workers the exhaustive strategy explores all
    # C(4,2)=6 interleavings; DPOR prunes equivalent ones and explores fewer.
    from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator

    db_ex = _DB()
    w_ex = _rmw_worker(db_ex)
    exhaustive = CrossProcessCoordinator(num_workers=2).explore(
        worker_set=ThreadLauncher([w_ex, w_ex]),
        setup=db_ex.reset,
        invariant=lambda: True,
        max_iterations=100,
    )

    db_dp = _DB()
    w_dp = _rmw_worker(db_dp)
    dpor = DporCrossProcessCoordinator(num_workers=2, stop_on_first=False).explore(
        worker_set=ThreadLauncher([w_dp, w_dp]),
        setup=db_dp.reset,
        invariant=lambda: True,
    )

    assert exhaustive.exhausted and dpor.exhausted
    assert exhaustive.iterations == 6
    assert dpor.iterations < exhaustive.iterations


def _ordered_locker(first: str, second: str):
    def worker(proxy) -> None:
        proxy.acquire_row_locks(0, [first])
        proxy.report_and_wait(None, 0)
        proxy.acquire_row_locks(0, [second])
        proxy.report_and_wait(None, 0)
        proxy.release_row_locks(0)

    return worker


def test_dpor_deadlock_does_not_claim_exhausted() -> None:
    # Regression: a deadlock-aborted execution unwinds its workers via
    # SchedulerAbort before their remaining accesses are reported, so the engine
    # never seeds the wakeup tree from that trace and next_execution() returns
    # False with deadlock-avoiding interleavings still unexplored. The coordinator
    # must NOT then claim exhausted=True (over-claiming coverage) with
    # stop_on_first=False; only max_executions/total_timeout previously demoted it.
    row1, row2 = "sql:accounts:id=1", "sql:accounts:id=2"
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0, stop_on_first=False)
    result = coord.explore(
        worker_set=ThreadLauncher([_ordered_locker(row1, row2), _ordered_locker(row2, row1)]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert result.failure_kind == "deadlock"
    assert not result.exhausted  # search aborted at a deadlock; space not fully covered

    # Control: same-order locking (no deadlock) with a conflicting write inside
    # the critical section explores >1 interleaving, proving the deadlock run
    # left reachable orderings unexplored. Pure lock/unlock workers with no data
    # accesses are all Mazurkiewicz-equivalent, so the write is what forces the
    # engine to reverse the acquisition order.
    def writing_locker(proxy) -> None:
        proxy.acquire_row_locks(0, [row1])
        proxy.io_report(row1, "write")
        proxy.report_and_wait(None, 0)
        proxy.release_row_locks(0)

    control = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0, stop_on_first=False)
    control_result = control.explore(
        worker_set=ThreadLauncher([writing_locker, writing_locker]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert control_result.iterations > 1


def test_dpor_lock_first_workers_explore_deterministically() -> None:
    # Regression: acquire_row_locks is not gated on the engine's scheduling
    # turn, so when two workers' first frames are both ACQUIRE_LOCKS the relay
    # threads race: whichever wins reports lock_acquire (and its subsequent
    # write) at the step the engine committed to the OTHER thread. The engine's
    # recorded schedule then disagrees with the actual executor, corrupting
    # backtracking: the reachable acquisition-order reversal is silently pruned
    # and the run reports iterations=1 with exhausted=True (over-claiming
    # coverage). The in-process path cannot race here because opcode tracing
    # means the acquiring thread already holds the turn.
    row1 = "sql:accounts:id=1"

    def writing_locker(proxy) -> None:
        proxy.acquire_row_locks(0, [row1])
        proxy.io_report(row1, "write")
        proxy.report_and_wait(None, 0)
        proxy.release_row_locks(0)

    # Two same-lock workers with conflicting writes have exactly two
    # Mazurkiewicz classes (the two acquisition orders). Pre-fix the race made
    # this 1-or-2 nondeterministically; it must be 2 every time.
    for attempt in range(8):
        coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0, stop_on_first=False)
        result = coord.explore(
            worker_set=ThreadLauncher([writing_locker, writing_locker]),
            setup=lambda: None,
            invariant=lambda: True,
        )
        assert result.ok
        assert result.iterations == 2, (
            f"attempt {attempt}: expected both lock-acquisition orders explored, "
            f"got iterations={result.iterations} (exhausted={result.exhausted})"
        )


def test_dpor_single_statement_conflict_found_deterministically() -> None:
    # Regression: workers report their accesses BEFORE executing the statement
    # (ACCESS frames precede REPORT_AND_WAIT), so when worker B's ACCESS frame
    # wins the OS race against worker A's grant, the scheduler's cross-thread
    # pending-io flush attributes B's not-yet-executed access at A's step. The
    # engine trace desyncs, the write-write reversal is never seeded into the
    # wakeup tree, and the search ends after 1 iteration with ok=True and
    # exhausted=True — a false negative. Pre-fix this misses on most runs, so
    # loop enough fresh searches that at least one reliably misses.
    class _Counter:
        def __init__(self) -> None:
            self.value = 1
            self.lock = threading.Lock()

        def reset(self) -> None:
            self.value = 1

    def make_worker(db: _Counter, fn):
        def worker(proxy) -> None:
            proxy.io_report("sql:counter:id=1", "write")
            proxy.report_and_wait(None, 0)
            with db.lock:
                db.value = fn(db.value)

        return worker

    # value starts at 1; mul2-then-add3 gives 5, add3-then-mul2 gives 8.
    # Only the second order violates the invariant, so DPOR must explore both
    # write-write orders every search: found at iteration 2, deterministically.
    for attempt in range(15):
        db = _Counter()
        coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
        result = coord.explore(
            worker_set=ThreadLauncher([make_worker(db, lambda v: v * 2), make_worker(db, lambda v: v + 3)]),
            setup=db.reset,
            invariant=lambda db=db: db.value != 8,
        )
        assert not result.ok, (
            f"attempt {attempt}: DPOR missed the order-dependent conflict "
            f"(iterations={result.iterations}, exhausted={result.exhausted})"
        )
        assert result.failure_kind == "invariant"
        assert result.iterations == 2, (
            f"attempt {attempt}: expected the reversal on the second execution, got {result.iterations}"
        )


def test_dpor_scheduler_timeout_is_a_failure_not_a_pass() -> None:
    # Regression: a deadlock_timeout expiry (unmodeled DB-level blocking, or a
    # statement slower than deadlock_timeout) aborts all workers and skips the
    # invariant — but used to count as a clean pass, so an exploration whose
    # invariant is ALWAYS False could report ok=True with exhausted=True. A
    # scheduler timeout must surface as a structured failing result instead.
    def slow(proxy) -> None:
        proxy.io_report("sql:t:id=1", "write")
        proxy.report_and_wait(None, 0)
        time.sleep(1.5)  # "statement" slower than deadlock_timeout

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=0.5)
    result = coord.explore(
        worker_set=ThreadLauncher([slow, slow]),
        setup=lambda: None,
        invariant=lambda: False,
    )
    # The stall is now diagnosed at ~deadlock_timeout, i.e. faster than the
    # workers' 1.5s sleep finishes — and ThreadLauncher cannot force-kill a
    # thread the way the process launchers kill children, so reap the
    # stragglers here before the leak checker runs.
    for t in threading.enumerate():
        if t.name.startswith("xproc-worker-"):
            t.join(timeout=10.0)
    assert not result.ok, "scheduler timeout was scored as a pass despite an always-False invariant"
    assert result.failure_kind == "timeout"
    assert "deadlock_timeout" in (result.failure or "")
    assert not result.exhausted


def test_dpor_stop_on_first_false_collects_all_failures() -> None:
    # Regression (api-shape): the coordinator kept only first_failure, so with
    # stop_on_first=False every subsequent failing schedule was discarded. The
    # result must carry every failing execution as (execution_number, schedule)
    # pairs, mirroring thread-mode InterleavingResult.failures.
    db = _DB()
    worker = _rmw_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, stop_on_first=False)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: False,  # every completed execution fails
    )
    assert not result.ok
    assert result.failure_kind == "invariant"
    assert len(result.failures) == result.iterations
    assert len(result.failures) >= 1
    exec_no, schedule = result.failures[0]
    assert exec_no == 1
    assert schedule == result.failing_schedule
    # Execution numbers are 1-based and strictly increasing.
    assert [n for n, _ in result.failures] == list(range(1, result.iterations + 1))


@pytest.mark.parametrize("late_error", [OSError("connection lost"), WorkerSerializationError("cannot pickle")])
def test_dpor_late_launch_error_preserves_prior_failures(late_error: Exception) -> None:
    db = _DB()
    worker = _rmw_worker(db)
    delegate = ThreadLauncher([worker, worker])

    class FailsSecondLaunch:
        launches = 0

        def launch(self, targets: Any) -> Any:
            self.launches += 1
            if self.launches == 2:
                raise late_error
            return delegate.launch(targets)

        def join(self, handles: Any, timeout: float) -> list[Any]:
            return delegate.join(handles, timeout)

    result = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=1.0, stop_on_first=False).explore(
        worker_set=FailsSecondLaunch(),
        setup=db.reset,
        invariant=lambda: False,
    )

    assert result.failure_kind == "worker_error"
    assert len(result.failures) == 1
    assert result.failures[0][0] == 1


def test_dpor_stop_on_first_true_still_records_its_failure() -> None:
    # With stop_on_first=True the single failing execution appears in failures
    # too (thread-mode records the failure it stops on).
    db = _DB()
    worker = _rmw_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert not result.ok
    assert len(result.failures) == 1
    assert result.failures[0] == (result.iterations, result.failing_schedule)


class _NoopEngine:
    """Engine stub for driving _relay_loop without the Rust extension."""

    def report_synced_io_access(self, *args: Any) -> None:
        pass

    def report_io_access(self, *args: Any) -> None:
        pass

    def register_resource_group(self, *args: Any) -> None:
        pass


class _TurnOrderScheduler:
    """Fake scheduler capturing the shared accesses list at turn release.

    Exposes exactly what ``_relay_loop`` touches. ``after_sync_retry`` snapshots
    the accesses list so the test can assert the relay appended its worker's
    ACCESS entry BEFORE releasing the turn.
    """

    def __init__(self, accesses: list[tuple[int, str, str]]) -> None:
        self.engine = _NoopEngine()
        self.execution = None
        self.deadlock_timeout = 5.0
        self._engine_lock = threading.Lock()
        self._lock_depth_by_thread: dict[int, int] = {}
        self._pending_io_by_thread: dict[int, list[Any]] = {}
        self._accesses = accesses
        self.accesses_at_release: list[tuple[int, str, str]] | None = None

    def before_sync_retry(self, worker_id: int) -> bool:
        return True

    def after_sync_retry(self, worker_id: int) -> None:
        if self.accesses_at_release is None:
            self.accesses_at_release = list(self._accesses)

    def mark_done(self, worker_id: int) -> None:
        pass

    def report_error(self, error: Exception) -> None:
        pass


def test_relay_appends_access_before_releasing_turn() -> None:
    # Regression (exactness): the relay released the sync turn (after_sync_retry)
    # BEFORE appending the worker's ACCESS frame to the shared accesses list, so
    # the next scheduled relay could interleave ITS appends first depending on
    # OS thread timing — the human-facing access trace of the same DPOR
    # counterexample differed run to run. The access must be recorded while the
    # owning worker still holds the turn.
    accesses: list[tuple[int, str, str]] = []
    scheduler = _TurnOrderScheduler(accesses)
    coord_end, worker_end = socket.socketpair()
    relay = threading.Thread(
        target=_relay_loop,
        args=(scheduler, 0, coord_end, accesses, threading.Lock(), {}, set()),
        name="xproc-relay-ordering-test",
        daemon=True,
    )
    relay.start()
    try:
        proto.send_msg(worker_end, {"t": proto.REPORT_AND_WAIT, "w": 0})
        grant = proto.recv_msg(worker_end)
        assert grant is not None and grant["t"] == proto.GRANT
        proto.send_msg(worker_end, {"t": proto.ACCESS, "w": 0, "rid": "redis:k1", "kind": "write"})
        proto.send_msg(worker_end, {"t": proto.DONE, "w": 0})
        relay.join(10.0)
        assert not relay.is_alive()
    finally:
        worker_end.close()
        coord_end.close()
    assert scheduler.accesses_at_release == [(0, "redis:k1", "write")], (
        f"turn released before the worker's access was recorded: {scheduler.accesses_at_release!r}"
    )


def test_dpor_no_race_when_safe() -> None:
    # Each worker does a single atomic increment under a scheduling point.
    db = _DB()
    lock = threading.Lock()

    def atomic(proxy) -> None:
        proxy.report_and_wait(None, 0)
        with lock:
            db.balance += 100
        proxy.io_report("sql:accounts:id=1", "write")

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([atomic, atomic]),
        setup=db.reset,
        invariant=lambda: db.balance == 200,
    )
    assert result.ok
    assert result.exhausted
