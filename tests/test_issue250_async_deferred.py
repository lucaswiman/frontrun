"""Regressions for deferred async/virtual-clock findings from issue #250."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from frontrun import _async_cooperative
from frontrun._async_dpor_replay import _ReplayAsyncScheduler
from frontrun._async_virtual_timeouts import _VirtualLoopDeadline
from frontrun._virtual_clock import VirtualClock
from frontrun.async_shuffler import AwaitScheduler
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
