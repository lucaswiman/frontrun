"""Regressions for deferred async/virtual-clock findings from issue #250."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from frontrun import _async_cooperative
from frontrun import _virtual_clock as virtual_clock_module
from frontrun._async_autopause import _scheduler_var, _task_id_var
from frontrun._async_dpor_replay import _ReplayAsyncScheduler
from frontrun._async_virtual_timeouts import _VirtualLoopDeadline
from frontrun._virtual_clock import VirtualClock
from frontrun.async_shuffler import AwaitScheduler
from frontrun.async_scheduler import SchedulerTimeoutError
from frontrun.bytecode import run_with_schedule
from frontrun.contrib.sqlalchemy._shared import wrap_async_setup


def test_raw_async_cooperative_patch_helpers_are_reference_counted() -> None:
    patches = [
        (
            _async_cooperative._patch_asyncio_lock,
            _async_cooperative._unpatch_asyncio_lock,
            lambda: asyncio.Lock,
            _async_cooperative._CooperativeAsyncLock,
        ),
        (
            _async_cooperative._patch_asyncio_event,
            _async_cooperative._unpatch_asyncio_event,
            lambda: asyncio.Event,
            _async_cooperative._CooperativeAsyncEvent,
        ),
        (
            _async_cooperative._patch_asyncio_queue_condition,
            _async_cooperative._unpatch_asyncio_queue_condition,
            lambda: (asyncio.Queue, asyncio.Condition),
            (_async_cooperative._CooperativeAsyncQueue, _async_cooperative._CooperativeAsyncCondition),
        ),
    ]

    for patch, unpatch, current, replacement in patches:
        patch()
        patch()
        try:
            unpatch()
            assert current() == replacement
        finally:
            unpatch()


def test_async_random_and_replay_record_their_event_loop_thread() -> None:
    current = threading.get_ident()
    random_scheduler = AwaitScheduler([], 1)
    replay_scheduler = _ReplayAsyncScheduler([], 1)

    assert random_scheduler._event_loop_thread_id == current
    assert replay_scheduler._event_loop_thread_id == current


def test_virtual_timeout_deadline_keeps_provenance_through_additive_arithmetic() -> None:
    clock = VirtualClock()
    deadline = _VirtualLoopDeadline(100.0, 1_000_010.0, clock)

    earlier = deadline - 2.0
    later = 2.0 + deadline

    assert isinstance(earlier, _VirtualLoopDeadline)
    assert earlier.virtual_deadline == 1_000_008.0
    assert earlier.clock is clock
    assert isinstance(later, _VirtualLoopDeadline)
    assert later.virtual_deadline == 1_000_012.0
    assert later.clock is clock


def test_async_sqlalchemy_setup_detaches_pool_without_sync_closing_async_connections() -> None:
    class SyncEngine:
        def __init__(self) -> None:
            self.dispose_calls: list[dict[str, Any]] = []

        def dispose(self, **kwargs: Any) -> None:
            self.dispose_calls.append(kwargs)

    class AsyncEngine:
        def __init__(self) -> None:
            self.sync_engine = SyncEngine()

    engine = AsyncEngine()
    setup = wrap_async_setup(engine, lambda: "state")

    assert setup() == "state"
    assert engine.sync_engine.dispose_calls == [{"close": False}]


def test_run_with_schedule_validates_clock_diagnostics() -> None:
    with pytest.raises(ValueError, match="clock_diagnostics"):
        run_with_schedule(
            [],
            object,
            [lambda _state: None],
            clock="real",
            clock_diagnostics=True,
        )


def test_clock_diagnostic_deduplication_uses_its_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingLock:
        enters = 0

        def __enter__(self) -> None:
            self.enters += 1

        def __exit__(self, *_args: object) -> None:
            return None

    class Frame:
        f_code = (lambda: None).__code__
        f_locals: dict[str, object] = {}
        f_globals: dict[str, object] = {}

    lock = CountingLock()
    virtual_clock_module._scanned_code_objects.discard(Frame.f_code)
    monkeypatch.setattr(virtual_clock_module, "_diagnostics_lock", lock)

    virtual_clock_module.warn_if_captured_time_reference(Frame())

    assert lock.enters >= 1


def test_full_async_queue_put_does_not_repark_after_scheduler_abort() -> None:
    async def scenario() -> None:
        scheduler = _ReplayAsyncScheduler([0, 1], 2)
        queue: _async_cooperative._CooperativeAsyncQueue[str] = _async_cooperative._CooperativeAsyncQueue(maxsize=1)
        queue.put_nowait("full")
        scheduler_token = _scheduler_var.set(scheduler)
        task_token = _task_id_var.set(0)
        try:
            putter = asyncio.create_task(queue.put("blocked"))
            for _ in range(10):
                await asyncio.sleep(0)
                if queue in _async_cooperative._async_parked_queues:
                    break
            assert queue in _async_cooperative._async_parked_queues
            scheduler._error = SchedulerTimeoutError("abort")
            scheduler._on_error_set()
            with pytest.raises(SchedulerTimeoutError, match="queue put aborted"):
                await asyncio.wait_for(putter, timeout=0.2)
        finally:
            _task_id_var.reset(task_token)
            _scheduler_var.reset(scheduler_token)
            _async_cooperative._async_parked_queues.clear()

    asyncio.run(scenario())
