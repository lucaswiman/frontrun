"""Cross-process Redis path (Phase 2c) — functional coverage.

Redis interception schedules via the scheduler's two-phase before_io/after_io
(the worker holds the turn through the command) plus the io-reporter for key
accesses. These tests drive that path in-process (ThreadLauncher) against a
shared dict standing in for Redis, so the relay's BEFORE_IO/AFTER_IO handling is
exercised without a Redis server (that is the integration e2e).
"""

from __future__ import annotations

from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
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


def test_dpor_redis_atomic_incr_has_no_race() -> None:
    store = _Store()

    def atomic(proxy) -> None:
        # A single atomic INCR: one IO boundary, no read-modify-write gap.
        proxy.before_io(0, "redis:INCR:counter")
        store.value += 1
        proxy.io_report(KEY, "write")
        proxy.after_io(0, "redis:INCR:counter")

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0)
    result = coord.explore(
        worker_set=ThreadLauncher([atomic, atomic]),
        setup=store.reset,
        invariant=lambda: store.value == 2,
    )
    assert result.ok, f"unexpected {result.failure_kind}: {result.failure!r}"
    assert result.exhausted
