"""Async cooperative primitive wakeups, patch ownership, and execution reset."""

from __future__ import annotations

import asyncio

import pytest

import frontrun._async_cooperative as async_cooperative
from frontrun._async_autopause import _scheduler_var, _task_id_var
from frontrun._dpor_core import event_wake_sync_id
from frontrun._opcode_observer import StableObjectIds
from frontrun.async_dpor import (
    AsyncDporScheduler,
    _async_parked_conditions,
    _async_parked_events,
    _async_parked_queues,
    _CooperativeAsyncCondition,
    _CooperativeAsyncEvent,
    _CooperativeAsyncQueue,
    _patch_asyncio_event,
    _ReplayAsyncScheduler,
    _reset_async_lock_state,
    _unpatch_asyncio_event,
)
from frontrun.async_scheduler import SchedulerTimeoutError


@pytest.mark.parametrize("abort", ["_handle_timeout", "_handle_all_waiting_deadlock"])
def test_scheduler_abort_wakes_parked_primitive_waiters(abort: str) -> None:
    """A watchdog abort (``_handle_timeout``) must wake tasks parked on
    cooperative primitives so they free-run to completion, rather than
    leaving them parked until the outer ``timeout_per_run`` elapses.
    """

    async def scenario() -> bool:
        scheduler = object.__new__(AsyncDporScheduler)
        scheduler._num_tasks = 2
        scheduler._tasks_done = set()
        scheduler._error = None
        scheduler._current_task = 0
        scheduler._condition = asyncio.Condition()
        event = _CooperativeAsyncEvent()
        _async_parked_events.add(event)
        try:
            assert not event._event.is_set()
            async with scheduler._condition:
                getattr(scheduler, abort)(1, marker="x")
            return event._event.is_set()
        finally:
            _async_parked_events.clear()

    assert asyncio.run(scenario()) is True


def test_condition_notify_no_context_wakes_at_most_n_across_both_waiter_sets() -> None:
    """Without scheduler context, ``notify(n)`` must wake at most ``n`` waiters
    total across the real-condition and cooperative waiter populations.

    The no-context path delegates to the wrapped real condition AND then also
    resolves cooperative-waiter futures; with a mix of both, ``notify(1)``
    used to wake two.
    """

    async def scenario() -> int:
        condition = _CooperativeAsyncCondition()
        loop = asyncio.get_running_loop()
        real_fut: asyncio.Future[bool] = loop.create_future()
        coop_fut: asyncio.Future[None] = loop.create_future()
        # One real-condition waiter (what the no-context wait() path registers).
        condition._real_condition._waiters.append(real_fut)  # type: ignore[attr-defined]
        # One cooperative waiter (what the with-context wait() path registers).
        condition._waiters.add(123, coop_fut)
        await condition.acquire()
        try:
            condition.notify(1)
        finally:
            condition.release()
        woke = sum(1 for fut in (real_fut, coop_fut) if fut.done())
        for fut in (real_fut, coop_fut):
            if not fut.done():
                fut.cancel()
        return woke

    assert asyncio.run(scenario()) == 1


def test_reset_async_lock_state_clears_all_parked_primitive_sets() -> None:
    """``_reset_async_lock_state`` must clear the parked-queue and
    parked-condition sets too, not only parked events.  Otherwise a stale
    cooperative queue/condition from a prior execution or replay attempt
    leaks into the next one (only cleared at unpatch).
    """

    async def _make() -> tuple[_CooperativeAsyncQueue[str], _CooperativeAsyncCondition, _CooperativeAsyncEvent]:
        return _CooperativeAsyncQueue(), _CooperativeAsyncCondition(), _CooperativeAsyncEvent()

    queue_obj, condition_obj, event_obj = asyncio.run(_make())
    _async_parked_queues.add(queue_obj)
    _async_parked_conditions.add(condition_obj)
    _async_parked_events.add(event_obj)
    try:
        _reset_async_lock_state()
        assert not _async_parked_events
        assert not _async_parked_queues
        assert not _async_parked_conditions
    finally:
        _async_parked_queues.clear()
        _async_parked_conditions.clear()
        _async_parked_events.clear()


def test_async_event_wake_sync_ids_use_stable_event_ids() -> None:
    """Event wake edges must be keyed by stable event id, not raw id(event)."""

    class Engine:
        pass

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
            self._event_blocked = {1}
            self.syncs: list[tuple[int, str, int]] = []

        def report_task_sync(self, task_id: int, event_type: str, sync_id: int) -> None:
            self.syncs.append((task_id, event_type, sync_id))

        def report_task_access(self, task_id: int, object_id: int, kind: str) -> None:
            pass

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

        assert scheduler.syncs == [(0, "lock_release", event_wake_sync_id(stable_event_id, 1))]
        assert scheduler.execution.unblocked == [1]
        assert scheduler._event_blocked == set()
    finally:
        _unpatch_asyncio_event()
        _reset_async_lock_state()


def test_raw_async_cooperative_patch_helpers_are_reference_counted() -> None:
    patches = [
        (
            async_cooperative._patch_asyncio_lock,
            async_cooperative._unpatch_asyncio_lock,
            lambda: asyncio.Lock,
            async_cooperative._CooperativeAsyncLock,
        ),
        (
            async_cooperative._patch_asyncio_event,
            async_cooperative._unpatch_asyncio_event,
            lambda: asyncio.Event,
            async_cooperative._CooperativeAsyncEvent,
        ),
        (
            async_cooperative._patch_asyncio_queue_condition,
            async_cooperative._unpatch_asyncio_queue_condition,
            lambda: (asyncio.Queue, asyncio.Condition),
            (async_cooperative._CooperativeAsyncQueue, async_cooperative._CooperativeAsyncCondition),
        ),
    ]

    for patch, unpatch, current, replacement in patches:
        original = current()
        patch()
        patch()
        try:
            unpatch()
            assert current() == replacement
        finally:
            unpatch()
        assert current() == original


def test_full_async_queue_put_does_not_repark_after_scheduler_abort() -> None:
    async def scenario() -> None:
        scheduler = _ReplayAsyncScheduler([0, 1], 2)
        queue: async_cooperative._CooperativeAsyncQueue[str] = async_cooperative._CooperativeAsyncQueue(maxsize=1)
        queue.put_nowait("full")
        scheduler_token = _scheduler_var.set(scheduler)
        task_token = _task_id_var.set(0)
        try:
            putter = asyncio.create_task(queue.put("blocked"))
            for _ in range(10):
                await asyncio.sleep(0)
                if queue in async_cooperative._async_parked_queues:
                    break
            assert queue in async_cooperative._async_parked_queues
            scheduler._error = SchedulerTimeoutError("abort")
            scheduler._on_error_set()
            with pytest.raises(SchedulerTimeoutError, match="queue put aborted"):
                await asyncio.wait_for(putter, timeout=0.2)
        finally:
            _task_id_var.reset(task_token)
            _scheduler_var.reset(scheduler_token)
            async_cooperative._async_parked_queues.clear()

    asyncio.run(scenario())
