"""Shared async row-lock protocol regressions for exploration and replay."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest

from frontrun._async_dpor_replay import _ReplayAsyncScheduler
from frontrun.async_dpor import AsyncDporScheduler
from frontrun.cli import require_active


def _exploration_scheduler() -> AsyncDporScheduler:
    from frontrun._dpor import PyDporEngine

    engine = PyDporEngine(2)
    return AsyncDporScheduler(engine, engine.begin_execution(), 2)


def _replay_scheduler() -> _ReplayAsyncScheduler:
    return _ReplayAsyncScheduler([1, 0, 1], 2)


@pytest.mark.parametrize("make_scheduler", [_exploration_scheduler, _replay_scheduler], ids=["exploration", "replay"])
def test_cancelled_multirow_wait_releases_partial_acquisition(
    monkeypatch: pytest.MonkeyPatch, make_scheduler: Callable[[], Any]
) -> None:
    """Cancellation before the SQL call must roll back locks acquired by that call."""
    require_active("test_cancelled_multirow_wait_releases_partial_acquisition")

    async def scenario() -> None:
        scheduler = make_scheduler()

        async def no_op(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(scheduler, "kick_stalled_schedule", no_op)
        monkeypatch.setattr(scheduler, "wait_until_scheduled_after_block", no_op)
        scheduler._row_lock_registry.record_acquire(0, "held", None)

        acquire = asyncio.create_task(scheduler.acquire_row_locks_async(1, ["partial", "held"]))
        await asyncio.sleep(0)
        assert scheduler._active_row_locks["partial"] == 1
        assert "held" in scheduler._row_lock_waiters

        acquire.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await acquire

        assert "partial" not in scheduler._active_row_locks
        assert scheduler._row_lock_waiters == {}
        assert 1 not in scheduler._event_blocked
        assert 1 not in scheduler._lock_blocked

    asyncio.run(scenario())
