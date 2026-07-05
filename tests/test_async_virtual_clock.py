"""Async virtual-clock tests (``clock="virtual"`` / ``clock="explored"``).

Async counterpart of ``test_virtual_clock.py``: ``asyncio.sleep`` becomes a
virtual deadline, ``time.monotonic()`` reads virtual time inside explored
tasks, and with ``clock="explored"`` the clock advance is a schedulable
choice for the async DPOR engine.
"""

from __future__ import annotations

import asyncio
import time

import frontrun


class _SleepObserver:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0


def test_async_sleep_advances_virtual_clock_zero_wall_time() -> None:
    async def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        await asyncio.sleep(500.0)
        s.end = time.monotonic()

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_SleepObserver,
            workers=[worker],
            invariant=lambda s: s.end - s.start >= 500.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 60.0


class _TTLCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, object]] = {}
        self.observed_miss_after_expiry = False

    def set(self, key: str, value: object, ttl: float) -> None:
        self._data[key] = (time.monotonic() + ttl, value)

    def get(self, key: str) -> object | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._data[key]
            return None
        return value


def test_async_ttl_cache_expiry_is_reachable() -> None:
    async def worker(cache: _TTLCache) -> None:
        cache.set("k", "v", ttl=1.0)
        assert cache.get("k") == "v"
        await asyncio.sleep(2.0)
        cache.observed_miss_after_expiry = cache.get("k") is None

    result = asyncio.run(
        frontrun.explore(
            setup=_TTLCache,
            workers=[worker],
            invariant=lambda c: c.observed_miss_after_expiry,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


class _WakeOrder:
    def __init__(self) -> None:
        self.order: list[str] = []


def test_async_sleepers_wake_in_deadline_order() -> None:
    async def short(s: _WakeOrder) -> None:
        await asyncio.sleep(1.0)
        s.order.append("short")

    async def long(s: _WakeOrder) -> None:
        await asyncio.sleep(5.0)
        s.order.append("long")

    result = asyncio.run(
        frontrun.explore(
            setup=_WakeOrder,
            workers=[long, short],
            invariant=lambda s: s.order == ["short", "long"],
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


class _RetryRace:
    def __init__(self) -> None:
        self.x = 0


async def _rmw_worker(s: _RetryRace) -> None:
    tmp = s.x
    await asyncio.sleep(0)  # natural await point inside the RMW window
    s.x = tmp + 1


async def _delayed_writer(s: _RetryRace) -> None:
    await asyncio.sleep(1.0)
    s.x = 100


def test_async_autojump_does_not_explore_early_timer_fire() -> None:
    result = asyncio.run(
        frontrun.explore(
            setup=_RetryRace,
            workers=[_rmw_worker, _delayed_writer],
            invariant=lambda s: s.x == 100,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_explored_clock_finds_timer_between_read_and_write() -> None:
    result = asyncio.run(
        frontrun.explore(
            setup=_RetryRace,
            workers=[_rmw_worker, _delayed_writer],
            invariant=lambda s: s.x == 100,
            clock="explored",
        )
    )
    assert not result.property_holds
    assert result.counterexample is not None


def test_async_random_virtual_sleep_zero_wall_time() -> None:
    async def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        await asyncio.sleep(300.0)
        s.end = time.monotonic()

    async def noop(s: _SleepObserver) -> None:
        pass

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_SleepObserver,
            workers=[worker, noop],
            invariant=lambda s: s.end - s.start >= 300.0,
            strategy="random",
            clock="virtual",
            max_attempts=5,
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 60.0


def test_async_random_explored_clock_can_fire_timer_early() -> None:
    result = asyncio.run(
        frontrun.explore(
            setup=_RetryRace,
            workers=[_rmw_worker, _delayed_writer],
            invariant=lambda s: s.x == 100,
            strategy="random",
            clock="explored",
            max_attempts=200,
            seed=1234,
            reproduce_on_failure=0,
        )
    )
    assert not result.property_holds


def test_async_wait_for_stays_on_wall_clock() -> None:
    """Documented limitation: loop timers (asyncio.wait_for) remain real, so
    a short real timeout fires even while virtual sleeps are active."""

    class State:
        def __init__(self) -> None:
            self.timed_out = False

    async def worker(s: State) -> None:
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.2)
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation
