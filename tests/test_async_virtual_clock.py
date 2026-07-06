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


def test_uncaught_async_wait_for_timeout_is_task_crash_not_deadlock() -> None:
    class State:
        pass

    async def worker(s: State) -> None:
        await asyncio.wait_for(asyncio.Event().wait(), timeout=0.05)

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
            deadlock_timeout=0.2,
            timeout_per_run=1.0,
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "Task crash" in result.explanation
    assert "TimeoutError" in result.explanation
    assert "Deadlock detected" not in result.explanation


def test_async_wait_for_lock_timeout_is_not_scored_as_deadlock() -> None:
    class State:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.release = asyncio.Event()
            self.timed_out = False
            self.holder_done = False

    async def holder(s: State) -> None:
        async with s.lock:
            await s.release.wait()
        s.holder_done = True

    async def waiter(s: State) -> None:
        await asyncio.sleep(0)
        try:
            await asyncio.wait_for(s.lock.acquire(), timeout=0.05)
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True
            s.release.set()

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[holder, waiter],
            invariant=lambda s: s.timed_out and s.holder_done,
            clock="virtual",
            reproduce_on_failure=0,
            deadlock_timeout=0.2,
            timeout_per_run=1.0,
        )
    )
    assert result.property_holds, result.explanation


def test_timer_tagging_restores_loop_call_at_instance_override() -> None:
    from frontrun.async_dpor import _install_frontrun_timer_tagging

    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        def custom_call_at(when: float, callback: object, *args: object, context: object = None) -> object:
            raise AssertionError("not called")

        setattr(loop, "call_at", custom_call_at)
        _check_user_timers, uninstall = _install_frontrun_timer_tagging(loop)
        uninstall()

        assert getattr(loop, "call_at") is custom_call_at

    asyncio.run(scenario())


def test_async_runtime_pin_restores_loop_time_instance_override() -> None:
    from frontrun.async_shuffler import _patch_async_runtime

    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        def custom_time() -> float:
            return 123.0

        setattr(loop, "time", custom_time)
        with _patch_async_runtime(virtual_time=True, pin_loop_time=loop):
            assert getattr(loop, "time") is not custom_time

        assert getattr(loop, "time") is custom_time

    asyncio.run(scenario())


class _CrossEventState:
    def __init__(self) -> None:
        self.e1 = asyncio.Event()
        self.e2 = asyncio.Event()


def test_async_event_deadlock_detected_exactly() -> None:
    """Two tasks each waiting on the other's event is a genuine deadlock:
    with a virtual clock and no pending deadline it must be reported via
    exact detection, not by burning the wall-clock fallback timeout."""

    async def w1(s: _CrossEventState) -> None:
        await s.e1.wait()
        s.e2.set()

    async def w2(s: _CrossEventState) -> None:
        await s.e2.wait()
        s.e1.set()

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_CrossEventState,
            workers=[w1, w2],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert "deadlock" in str(result.explanation).lower()
    assert wall_elapsed < 4.0, f"deadlock took {wall_elapsed:.1f}s to report (wall-clock fallback?)"


class _EventHandoffState:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.flag = False
        self.observed = False


def test_async_event_wait_explores_cleanly() -> None:
    """A plain setter/waiter handoff must explore without false deadlocks:
    branches that schedule the waiter before the setter must block the waiter
    in the engine and hand the turn onward, not stall until the wall
    timeout scores a bogus deadlock counterexample."""

    async def waiter(s: _EventHandoffState) -> None:
        await s.event.wait()
        s.observed = s.flag

    async def setter(s: _EventHandoffState) -> None:
        s.flag = True
        s.event.set()

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_EventHandoffState,
            workers=[waiter, setter],
            invariant=lambda s: s.observed,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"event handoff took {wall_elapsed:.1f}s (deadlock-timeout stall?)"


def test_async_event_wait_with_virtual_sleeper_autojumps() -> None:
    """An event waiter plus a virtual sleeper that eventually sets the event:
    the waiter parking on the event must not stop the autojump — the clock
    advances, the sleeper wakes and sets, and the run completes fast."""

    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()
            self.woke = False

    async def waiter(s: State) -> None:
        await s.event.wait()
        s.woke = True

    async def sleeper(s: State) -> None:
        await asyncio.sleep(1.0)
        s.event.set()

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[waiter, sleeper],
            invariant=lambda s: s.woke,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"event+sleeper took {wall_elapsed:.1f}s (autojump stall?)"


class _AsyncHoldAndSleep:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.a_sleep_virtual = 0.0
        self.b_acquired = False


def test_async_sleep_while_holding_lock() -> None:
    """A task sleeping while holding an asyncio.Lock must autojump, not die
    by deadlock timeout (regression: the run was scored as a false deadlock
    counterexample — the blocked contender has no scheduling points, so
    nothing ever asked the engine to reschedule)."""

    async def a(s: _AsyncHoldAndSleep) -> None:
        async with s.lock:
            start = time.monotonic()
            await asyncio.sleep(1.0)
            s.a_sleep_virtual = time.monotonic() - start

    async def b(s: _AsyncHoldAndSleep) -> None:
        await asyncio.sleep(0)
        async with s.lock:
            s.b_acquired = True

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_AsyncHoldAndSleep,
            workers=[a, b],
            invariant=lambda s: s.b_acquired and s.a_sleep_virtual >= 1.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"async lock+sleep took {wall_elapsed:.1f}s (deadlock-timeout stall?)"


def test_async_random_lock_sleep_quiescence_rescue() -> None:
    """Random strategy cannot see tasks parked on a raw asyncio.Lock; the
    quiescence heuristic must advance the clock instead of letting the run
    die by wall timeout."""

    async def a(s: _AsyncHoldAndSleep) -> None:
        async with s.lock:
            start = time.monotonic()
            await asyncio.sleep(1.0)
            s.a_sleep_virtual = time.monotonic() - start

    async def b(s: _AsyncHoldAndSleep) -> None:
        await asyncio.sleep(0)
        async with s.lock:
            s.b_acquired = True

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_AsyncHoldAndSleep,
            workers=[a, b],
            invariant=lambda s: s.b_acquired and s.a_sleep_virtual >= 1.0,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=7,
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 10.0, f"async random lock+sleep took {wall_elapsed:.1f}s"


def test_async_random_lock_sleep_quiescence_respects_small_deadlock_timeout() -> None:
    async def a(s: _AsyncHoldAndSleep) -> None:
        async with s.lock:
            start = time.monotonic()
            await asyncio.sleep(1.0)
            s.a_sleep_virtual = time.monotonic() - start

    async def b(s: _AsyncHoldAndSleep) -> None:
        await asyncio.sleep(0)
        async with s.lock:
            s.b_acquired = True

    result = asyncio.run(
        frontrun.explore(
            setup=_AsyncHoldAndSleep,
            workers=[a, b],
            invariant=lambda s: s.b_acquired and s.a_sleep_virtual >= 1.0,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=7,
            deadlock_timeout=0.05,
            timeout_per_run=1.0,
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation
