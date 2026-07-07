"""Regression tests for async DPOR counterexample replay (findings F3, F5).

F3: ``_ReplayAsyncScheduler`` lacked the ``engine`` / ``execution`` /
``_lock_blocked`` attributes that the patched ``_CooperativeAsyncLock``
references, so every lock-involving counterexample raised an AttributeError
during replay.  That error was swallowed by an over-broad
``except (TimeoutError, Exception)``, so reproduction_successes was always 0.

F5: cooperative-lock global state (wait-for graph, lock owners, held locks)
was cleared only between exploration iterations, never between the up-to-10
replay attempts, so stale edges / id() reuse could cause spurious
DeadlockError or phantom ownership in later replays.
"""

from __future__ import annotations

import asyncio
from typing import Any

from frontrun._async_autopause import _scheduler_var, _task_id_var, wrap_auto_paused_tasks
from frontrun._dpor_core import event_wake_sync_id
from frontrun._opcode_observer import StableObjectIds
from frontrun.async_dpor import (
    _patch_asyncio_event,
    _ReplayAsyncScheduler,
    _reset_async_lock_state,
    _unpatch_asyncio_event,
)
from frontrun.cli import require_active


def test_lock_counterexample_reproduces() -> None:
    """A counterexample whose tasks use asyncio.Lock must reproduce.

    The increment acquires a lock only around the *read*, then releases and
    writes after an await — a lost update is still possible.  DPOR finds the
    failing schedule; replaying it must reproduce the violation rather than
    crash in lock plumbing and report 0 reproductions.
    """
    require_active("test_async_dpor_lock_replay")
    import frontrun

    class Counter:
        def __init__(self) -> None:
            self.value = 0
            self.lock = asyncio.Lock()

        async def increment(self) -> None:
            async with self.lock:
                temp = self.value
            await asyncio.sleep(0)
            self.value = temp + 1

    async def _do_increment(c: Counter) -> None:
        await c.increment()

    result = asyncio.run(
        frontrun.explore(
            setup=Counter,
            workers=[_do_increment, _do_increment],
            invariant=lambda c: c.value == 2,
            strategy="dpor",
            deadlock_timeout=2.0,
            timeout_per_run=3.0,
            reproduce_on_failure=10,
        )
    )

    assert not result.property_holds, "lost update should be found"
    assert result.reproduction_attempts == 10, result.reproduction_attempts
    # The core F3 bug: replay crashed in lock plumbing → 0 successes.
    assert result.reproduction_successes > 0, (
        f"counterexample must reproduce, got {result.reproduction_successes}/{result.reproduction_attempts}"
    )


def test_lock_blocked_holder_counterexample_reproduces() -> None:
    """A counterexample where a task blocks on a lock held across an await must reproduce.

    ``AsyncDporScheduler.should_proceed`` applies a lock-blocked holder override:
    when the engine picks a task blocked on a contended ``asyncio.Lock`` held
    across an await, the scheduler silently runs the *holder* so it can release
    the lock.  ``_ReplayAsyncScheduler`` populates ``_lock_blocked`` (the patched
    lock-acquire path writes it) but never *read* it, so replaying a recorded
    schedule that contains an engine pick of a lock-blocked task deadlocked: the
    picked task is stuck inside the real ``lock.acquire()`` and the holder is
    denied.  Every replay attempt then hung for ``deadlock_timeout`` and was
    scored as a failed reproduction, so a genuine deterministic counterexample
    reported ``reproduction_successes == 0``.

    ``deadlock_timeout`` is kept short so the *pre-fix* hang fails in bounded
    time rather than stalling the suite.
    """
    require_active("test_async_dpor_lock_blocked_replay")
    import frontrun

    class S:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.was_held = False

    async def t0(s: S) -> None:
        async with s.lock:  # holds the lock ACROSS an await
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    async def t1(s: S) -> None:
        await asyncio.sleep(0)
        if s.lock.locked():
            s.was_held = True
        async with s.lock:  # blocks on the held lock → engine picks t1, holder t0 must run
            await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore(
            setup=S,
            workers=[t0, t1],
            invariant=lambda s: not s.was_held,
            strategy="dpor",
            deadlock_timeout=1.0,
            timeout_per_run=3.0,
            reproduce_on_failure=5,
        )
    )

    assert not result.property_holds, "lock-held-across-await violation should be found"
    assert result.reproduction_attempts == 5, result.reproduction_attempts
    # The bug: replay of the recorded schedule deadlocks because
    # _ReplayAsyncScheduler.should_proceed lacks the _lock_blocked holder
    # override, so every attempt hangs deadlock_timeout → 0 successes.
    assert result.reproduction_successes > 0, (
        f"lock-blocked counterexample must reproduce, "
        f"got {result.reproduction_successes}/{result.reproduction_attempts}"
    )


def test_event_blocked_replay_skips_drifted_waiter_slots() -> None:
    """Replay must not stall when a drifted schedule points at an event waiter.

    A cooperative Event.wait() consumes a scheduling point before it parks.
    If the positional replay schedule has an extra slot for that same waiter,
    the waiter is now blocked inside the real event wait and cannot consume it.
    Replay should skip to the setter instead of burning the deadlock timeout.
    """

    async def scenario() -> list[str]:
        _patch_asyncio_event()
        try:
            event = asyncio.Event()
            order: list[str] = []

            async def waiter() -> None:
                await event.wait()
                order.append("waiter")

            async def setter() -> None:
                event.set()
                order.append("setter")

            scheduler = _ReplayAsyncScheduler([0, 0, 0, 1, 0], 2, deadlock_timeout=0.1)
            await scheduler.run_all(wrap_auto_paused_tasks({0: waiter, 1: setter}, scheduler), timeout=0.5)
            assert scheduler._error is None
            return order
        finally:
            _unpatch_asyncio_event()
            _reset_async_lock_state()

    assert asyncio.run(scenario()) == ["setter", "waiter"]


def test_async_event_wake_sync_ids_use_stable_event_ids() -> None:
    """Event wake edges must be keyed by stable event id, not raw id(event)."""

    class Engine:
        def __init__(self) -> None:
            self.syncs: list[tuple[int, str, int]] = []

        def report_sync(self, execution: Any, task_id: int, event: str, object_id: int) -> None:
            self.syncs.append((task_id, event, object_id))

    class Execution:
        def __init__(self) -> None:
            self.unblocked: list[int] = []

        def unblock_thread(self, task_id: int) -> None:
            self.unblocked.append(task_id)

    class Scheduler:
        def __init__(self) -> None:
            self.engine = Engine()
            self.execution = Execution()
            self._stable_ids = StableObjectIds()
            self._error = None

    _patch_asyncio_event()
    try:
        event = asyncio.Event()
        scheduler = Scheduler()
        stable_event_id = scheduler._stable_ids.get(event)
        event._waiters.append(1)  # type: ignore[attr-defined]

        scheduler_token = _scheduler_var.set(scheduler)
        task_token = _task_id_var.set(0)
        try:
            event.set()
        finally:
            _task_id_var.reset(task_token)
            _scheduler_var.reset(scheduler_token)

        assert scheduler.engine.syncs == [(0, "lock_release", event_wake_sync_id(stable_event_id, 1))]
        assert scheduler.execution.unblocked == [1]
    finally:
        _unpatch_asyncio_event()
        _reset_async_lock_state()
