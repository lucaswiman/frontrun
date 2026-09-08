"""Shared async row-lock protocol regressions for exploration and replay."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

import frontrun._async_cooperative as async_cooperative
from frontrun._async_dpor_replay import _ReplayAsyncScheduler
from frontrun._deadlock import DeadlockError, WaitForGraph
from frontrun._virtual_clock import VirtualClock
from frontrun.async_dpor import AsyncDporScheduler
from frontrun.cli import require_active


def _exploration_scheduler() -> AsyncDporScheduler:
    from frontrun._dpor import PyDporEngine

    engine = PyDporEngine(3)
    return AsyncDporScheduler(engine, engine.begin_execution(), 3)


def _replay_scheduler() -> _ReplayAsyncScheduler:
    return _ReplayAsyncScheduler([1, 0, 2], 3)


@pytest.fixture(params=[_exploration_scheduler, _replay_scheduler], ids=["exploration", "replay"])
def scheduler(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    require_active("async_row_lock_protocol")
    scheduler = request.param()

    async def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(scheduler, "kick_stalled_schedule", no_op)
    monkeypatch.setattr(scheduler, "wait_until_scheduled_after_block", no_op)
    return scheduler


def test_cancelled_multirow_wait_releases_partial_acquisition(scheduler: Any) -> None:
    """Cancellation before the SQL call must roll back locks acquired by that call."""

    async def scenario() -> None:
        scheduler._row_lock_registry.record_acquire(0, "held", None)
        scheduler._row_lock_registry.record_acquire(1, "prior", None)

        acquire = asyncio.create_task(scheduler.acquire_row_locks_async(1, ["prior", "partial", "held"]))
        await asyncio.sleep(0)
        assert scheduler._active_row_locks["partial"] == 1
        assert "held" in scheduler._row_lock_protocol.waiters

        acquire.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await acquire

        assert "partial" not in scheduler._active_row_locks
        assert scheduler._active_row_locks["prior"] == 1
        assert scheduler._row_lock_protocol.waiters == {}
        assert 1 not in scheduler._event_blocked
        assert 1 not in scheduler._lock_blocked

    asyncio.run(scenario())


def test_partial_release_only_wakes_waiters_for_released_rows(scheduler: Any) -> None:
    async def scenario() -> None:
        scheduler._row_lock_registry.record_acquire(0, "A", None)
        scheduler._row_lock_registry.record_acquire(0, "B", None)
        acquire_a = asyncio.create_task(scheduler.acquire_row_locks_async(1, ["A"]))
        acquire_b = asyncio.create_task(scheduler.acquire_row_locks_async(2, ["B"]))
        await asyncio.sleep(0)

        scheduler.release_row_locks(0, ["A"])
        assert await acquire_a == ["A"]
        assert not acquire_b.done()
        assert scheduler._active_row_locks == {"A": 1, "B": 0}

        scheduler.release_row_locks(0, ["B"])
        assert await acquire_b == ["B"]

    asyncio.run(scenario())


def test_row_lock_deadlock_preserves_prior_ownership(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        graph = WaitForGraph()
        monkeypatch.setattr(async_cooperative, "_async_wait_graph", graph)

        async def no_op(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(scheduler, "_report_error", no_op)
        scheduler._row_lock_registry.record_acquire(0, "A", graph)
        scheduler._row_lock_registry.record_acquire(1, "B", graph)
        lock_b = scheduler._row_lock_registry._row_lock_int_id("B")
        assert graph.add_waiting(0, lock_b, kind="row_lock") is None

        with pytest.raises(DeadlockError, match="Row-lock deadlock detected"):
            await scheduler.acquire_row_locks_async(1, ["A"])

        assert scheduler._active_row_locks == {"A": 0, "B": 1}
        assert scheduler._row_lock_protocol.waiters == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["exploration", "replay"])
def test_timeout_guarded_row_lock_wait_is_not_a_deadlock(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Exploration and replay must share timeout-aware wait-for graph semantics."""

    async def run() -> None:
        graph = WaitForGraph()
        monkeypatch.setattr(async_cooperative, "_async_wait_graph", graph)
        clock = VirtualClock()
        if mode == "exploration":
            from frontrun._dpor import PyDporEngine

            engine = PyDporEngine(3)
            scheduler = AsyncDporScheduler(
                engine, engine.begin_execution(), 2, virtual_clock=clock, clock_actor_id=2, clock_mode="virtual"
            )
        else:
            scheduler = _ReplayAsyncScheduler([1, 0, 1], 2, virtual_clock=clock, clock_actor_id=2)

        async def no_op(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(scheduler, "kick_stalled_schedule", no_op)
        monkeypatch.setattr(scheduler, "wait_until_scheduled_after_block", no_op)

        scheduler._row_lock_registry.record_acquire(0, "A", graph)
        scheduler._row_lock_registry.record_acquire(1, "B", graph)
        lock_b = scheduler._row_lock_registry._row_lock_int_id("B")
        assert graph.add_waiting(0, lock_b, kind="row_lock") is None
        scheduler.add_timeout_deadline(1, clock.now() + 1.0, object())

        acquire = asyncio.create_task(scheduler.acquire_row_locks_async(1, ["A"]))
        try:
            await asyncio.sleep(0)
            assert not acquire.done(), "the pending timeout makes this wait recoverable"

            scheduler.release_row_locks(0, ["A"])
            assert await acquire == ["A"]
        finally:
            if not acquire.done():
                acquire.cancel()
            with contextlib.suppress(asyncio.CancelledError, DeadlockError):
                await acquire

    asyncio.run(run())
