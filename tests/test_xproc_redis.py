"""Cross-process Redis path (Phase 2c) — functional coverage.

Redis interception schedules via the scheduler's two-phase before_io/after_io
(the worker holds the turn through the command) plus the io-reporter for key
accesses. These tests drive that path in-process (ThreadLauncher) against a
shared dict standing in for Redis, so the relay's BEFORE_IO/AFTER_IO handling is
exercised without a Redis server (that is the integration e2e).
"""

from __future__ import annotations

import socket

import pytest

from frontrun._deadlock import SchedulerAbort
from frontrun._dpor_runtime.xproc import protocol as proto
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.proxy import SchedulerProxy
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher

KEY = "redis:counter"


class _Store:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> None:
        self.value = 0


def _incr_worker(store: _Store):
    """GET then SET around two-phase IO boundaries (a Redis lost update)."""

    def worker(proxy) -> None:
        proxy.before_io(0, "redis:GET:counter")
        current = store.value
        proxy.io_report(KEY, "read")
        proxy.after_io(0, "redis:GET:counter")

        proxy.before_io(0, "redis:SET:counter")
        store.value = current + 1
        proxy.io_report(KEY, "write")
        proxy.after_io(0, "redis:SET:counter")

    return worker


def test_dpor_finds_redis_lost_update() -> None:
    store = _Store()
    worker = _incr_worker(store)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=store.reset,
        invariant=lambda: store.value == 2,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None


def test_proxy_before_io_returns_grant_outcome() -> None:
    # Regression: before_io discarded _await_grant()'s result, so an aborted
    # worker proceeded to run its Redis command unscheduled. The proxy must
    # surface the outcome (True on GRANT, False on ABORT, latched thereafter).
    coord_sock, worker_sock = socket.socketpair()
    try:
        proxy = SchedulerProxy(worker_sock, 0)
        proto.send_msg(coord_sock, {"t": proto.GRANT})
        assert proxy.before_io(0, "redis:GET:k") is True
        proto.send_msg(coord_sock, {"t": proto.ABORT})
        assert proxy.before_io(0, "redis:GET:k") is False
        # The abort latch short-circuits every later boundary without I/O.
        assert proxy.before_io(0, "redis:GET:k") is False
    finally:
        coord_sock.close()
        worker_sock.close()


def test_aborted_before_io_does_not_execute_redis_command() -> None:
    # Regression: _run_sync_dpor_envelope executed the Redis command even when
    # before_io was denied — an aborted cross-process worker kept mutating the
    # real Redis outside any schedule. A denied grant must raise SchedulerAbort
    # (matching the SQL path) without running the command.
    from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id
    from frontrun._redis_client import _run_sync_dpor_envelope

    class _AbortingScheduler:
        def __init__(self) -> None:
            self.after_io_resources: list[str] = []

        def before_io(self, thread_id: int, resource_id: str) -> bool:
            return False

        def after_io(self, thread_id: int, resource_id: str) -> None:
            self.after_io_resources.append(resource_id)

    executed: list[int] = []
    scheduler = _AbortingScheduler()
    set_dpor_scheduler(scheduler)
    set_dpor_thread_id(0)
    try:
        with pytest.raises(SchedulerAbort):
            _run_sync_dpor_envelope(lambda: executed.append(1), "redis\x1fSET\x1fk\x1f", False, True)
    finally:
        set_dpor_scheduler(None)
        set_dpor_thread_id(None)
    assert not executed
    # The boundary was never entered, so it must not be exited either.
    assert scheduler.after_io_resources == []


def test_dpor_redis_atomic_incr_has_no_race() -> None:
    store = _Store()

    def atomic(proxy) -> None:
        # A single atomic INCR: one IO boundary, no read-modify-write gap.
        proxy.before_io(0, "redis:INCR:counter")
        store.value += 1
        proxy.io_report(KEY, "write")
        proxy.after_io(0, "redis:INCR:counter")

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, preemption_bound=None)
    result = coord.explore(
        worker_set=ThreadLauncher([atomic, atomic]),
        setup=store.reset,
        invariant=lambda: store.value == 2,
    )
    assert result.ok, f"unexpected {result.failure_kind}: {result.failure!r}"
    assert result.exhausted
