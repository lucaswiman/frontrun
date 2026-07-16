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

import pytest

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


def test_exhaustive_failure_populates_failures_like_dpor() -> None:
    # CrossProcessResult.failures exists regardless of strategy, and the DPOR
    # coordinator records its failing (execution_number, schedule) there even
    # with the default stop_on_first. A strategy-agnostic caller iterating
    # result.failures must not silently see [] for the identical bug just
    # because it picked strategy='exhaustive' as the reduction-free cross-check.
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
    assert result.failures == [(result.iterations, list(result.failing_schedule))]


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


def test_replay_divergence_reported_as_nondeterministic() -> None:
    # A nondeterministic workload can make a recorded prefix choice ungrantable
    # on replay. That is a divergent/nondeterministic schedule, NOT a genuine
    # cross-worker deadlock, and must be reported as its own failure_kind so the
    # message is not misleading for a non-deadlocking workload.
    lock = threading.Lock()
    calls = {"n": 0}

    def stable(proxy) -> None:
        proxy.report_and_wait(None, 0)

    def flaky(proxy) -> None:
        # First exploration run: take a scheduling point (so the coordinator
        # records "grant worker 1 here" as a branch). Later runs skip it, so
        # replaying that recorded choice finds worker 1 no longer grantable.
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        if first:
            proxy.report_and_wait(None, 0)

    coord = CrossProcessCoordinator(num_workers=2, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([stable, flaky]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=50,
    )
    assert not result.ok
    assert result.failure_kind == "nondeterministic"
    assert result.failure_kind != "deadlock"
    assert "reproducible" in (result.failure or "")


def test_active_lock_owner_public_accessor() -> None:
    # Fix 2: _grantable must read row-lock ownership through a public accessor
    # rather than reaching into RowLockRegistry._active_row_locks.
    from frontrun._dpor_core.row_locks import RowLockRegistry

    reg = RowLockRegistry()
    assert reg.active_lock_owner("sql:accounts:id=1") is None
    reg.record_acquire(7, "sql:accounts:id=1", None)
    assert reg.active_lock_owner("sql:accounts:id=1") == 7
    reg.pop_all(7, None)
    assert reg.active_lock_owner("sql:accounts:id=1") is None


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


def test_accept_hello_treats_pre_hello_death_as_connection_failure() -> None:
    """A worker that connects then dies before sending HELLO must surface as a
    connection failure (OSError, which the coordinators catch), not an uncaught
    RuntimeError that escapes explore().  The accepted socket must be closed.
    """
    import socket

    from frontrun._dpor_runtime.xproc import protocol as proto
    from frontrun._dpor_runtime.xproc.coordinator import accept_hello

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        # Case 1: EOF before any HELLO frame.
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        client.close()
        with pytest.raises(OSError):
            accept_hello(listener, timeout=2.0)

        # Case 2: a well-formed frame that is not a valid HELLO (missing "w").
        client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client2.connect(("127.0.0.1", port))
        proto.send_msg(client2, {"t": proto.HELLO})  # no worker id
        with pytest.raises(OSError):
            accept_hello(listener, timeout=2.0)
        client2.close()
    finally:
        listener.close()


def test_accept_hello_restores_full_per_frame_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """HELLO latency must not consume the later per-frame silence budget."""
    from frontrun._dpor_runtime.xproc import coordinator
    from frontrun._dpor_runtime.xproc import protocol as proto

    class FakeSocket:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def settimeout(self, timeout: float) -> None:
            self.timeouts.append(timeout)

    sock = FakeSocket()

    class FakeListener:
        def accept(self):
            return sock, None

    times = iter([10.0, 10.75])
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(proto, "recv_msg", lambda _sock: {"t": proto.HELLO, "w": 0})

    accepted, worker_id = coordinator.accept_hello(FakeListener(), timeout=1.0)  # type: ignore[arg-type]

    assert accepted is sock
    assert worker_id == 0
    assert sock.timeouts == [0.25, 1.0]
