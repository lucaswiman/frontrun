"""Async virtual-clock tests (``clock="virtual"`` / ``clock="explored"``).

Async counterpart of ``test_virtual_clock.py``: ``asyncio.sleep`` becomes a
virtual deadline, ``time.monotonic()`` reads virtual time inside explored
tasks, and with ``clock="explored"`` the clock advance is a schedulable
choice for the async DPOR engine.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time

import pytest

import frontrun
from frontrun.common import InterleavingResult


def _assert_invariant_failure(result: InterleavingResult, expected: str | None = None) -> None:
    assert not result.property_holds
    assert result.explanation is not None
    assert "Deadlock" not in result.explanation
    assert "Task crash" not in result.explanation
    if expected is not None:
        assert expected in result.explanation


class _SleepObserver:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0


@pytest.mark.asyncio
async def test_async_random_finished_schedule_does_not_complete_infinite_sleep() -> None:
    from frontrun._virtual_clock import VirtualClock
    from frontrun.async_scheduler import SchedulerTimeoutError
    from frontrun.async_shuffler import AwaitScheduler

    scheduler = AwaitScheduler([], num_tasks=1, virtual_clock=VirtualClock(), clock_mode="virtual")
    scheduler._finished = True

    await scheduler.sleep_until(0, deadline=math.inf)

    assert isinstance(scheduler._error, SchedulerTimeoutError)
    assert scheduler.virtual_clock is not None
    assert math.isfinite(scheduler.virtual_clock.now())


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
    assert wall_elapsed < 4.0


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
    def invariant(s: _RetryRace) -> bool:
        assert s.x == 100, f"expected delayed writer to remain final, got x={s.x}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=_RetryRace,
            workers=[_rmw_worker, _delayed_writer],
            invariant=invariant,
            clock="explored",
        )
    )
    _assert_invariant_failure(result, "expected delayed writer")
    assert result.counterexample is not None


class _TimerCascadeRace:
    def __init__(self) -> None:
        self.x = 0


async def _early_timer_increments(s: _TimerCascadeRace) -> None:
    await asyncio.sleep(1.0)
    s.x += 1


async def _later_timer_writes(s: _TimerCascadeRace) -> None:
    await asyncio.sleep(2.0)
    s.x = 100


def test_async_explored_clock_can_fire_later_timer_before_earlier_sleeper_resumes() -> None:
    """The async clock actor must stay schedulable after waking the first
    deadline so DPOR can explore a later timer firing before that sleeper
    resumes."""

    def invariant(s: _TimerCascadeRace) -> bool:
        assert s.x == 100, f"expected later timer write to remain final, got x={s.x}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=_TimerCascadeRace,
            workers=[_early_timer_increments, _later_timer_writes],
            invariant=invariant,
            clock="explored",
        )
    )
    _assert_invariant_failure(result, "expected later timer write")
    assert result.counterexample is not None
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10


def test_async_random_virtual_sleep_zero_wall_time() -> None:
    async def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        await asyncio.sleep(300.0)
        s.end = time.monotonic()

    async def noop(s: _SleepObserver) -> None:
        pass

    invariant_checks = 0

    def invariant(s: _SleepObserver) -> bool:
        nonlocal invariant_checks
        invariant_checks += 1
        return s.end - s.start >= 300.0

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_SleepObserver,
            workers=[worker, noop],
            invariant=invariant,
            strategy="random",
            clock="virtual",
            max_attempts=5,
            timeout_per_run=1.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert invariant_checks > 0
    assert wall_elapsed < 4.0


@pytest.mark.parametrize("clock", ["virtual", "explored"])
def test_async_random_schedule_exhaustion_still_advances_virtual_sleep(clock: str) -> None:
    """Async random: a virtual sleep reached after the fixed-length schedule
    is exhausted must still advance the clock to its deadline (mirrors the
    sync max_ops regression) instead of returning with the clock frozen."""

    async def worker(s: _SleepObserver) -> None:
        for _ in range(300):
            await asyncio.sleep(0)  # burn the schedule (max_ops=10)
        s.start = time.monotonic()
        await asyncio.sleep(120.0)
        s.end = time.monotonic()

    result = asyncio.run(
        frontrun.explore(
            setup=_SleepObserver,
            workers=[worker],
            invariant=lambda s: s.end - s.start >= 120.0,
            strategy="random",
            clock=clock,  # type: ignore[arg-type]
            seed=0,
            max_attempts=1,
            max_ops=10,
            timeout_per_run=2.0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_random_explored_clock_can_fire_timer_early() -> None:
    def invariant(s: _RetryRace) -> bool:
        assert s.x == 100, f"expected delayed writer to remain final, got x={s.x}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=_RetryRace,
            workers=[_rmw_worker, _delayed_writer],
            invariant=invariant,
            strategy="random",
            clock="explored",
            max_attempts=200,
            seed=1234,
        )
    )
    _assert_invariant_failure(result, "expected delayed writer")


def test_async_wait_for_uses_virtual_deadline() -> None:
    """``asyncio.wait_for`` uses a virtual deadline inside explored tasks."""

    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        start = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.sleep(10.0), timeout=1.0)
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True
            s.elapsed = time.monotonic() - start

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker, lambda s: asyncio.sleep(0)],
            invariant=lambda s: s.timed_out and s.elapsed >= 1.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"virtual wait_for took {wall_elapsed:.1f}s wall time"


def test_async_wait_for_event_timeout_is_explored_before_delayed_setter() -> None:
    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()
            self.timed_out = False
            self.completed = False

    async def waiter(s: State) -> None:
        try:
            await asyncio.wait_for(s.event.wait(), timeout=1.0)
            s.completed = True
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True

    async def setter(s: State) -> None:
        await asyncio.sleep(1.0)
        s.event.set()

    def invariant(s: State) -> bool:
        assert not s.timed_out, "virtual wait_for timeout fired before the event setter"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[waiter, setter],
            invariant=invariant,
            clock="explored",
        )
    )
    _assert_invariant_failure(result, "virtual wait_for timeout")
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10


@pytest.mark.parametrize("timeout", [0, -1])
def test_async_wait_for_non_positive_timeout_does_not_start_inner_coroutine(timeout: float) -> None:
    class State:
        def __init__(self) -> None:
            self.inner_ran = False
            self.timed_out = False

    async def inner(s: State) -> None:
        s.inner_ran = True

    async def worker(s: State) -> None:
        try:
            await asyncio.wait_for(inner(s), timeout=timeout)
        except TimeoutError:
            s.timed_out = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and not s.inner_ran,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_wait_for_removes_timeout_deadline_after_success() -> None:
    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()
            self.completed = False
            self.elapsed = 0.0

    async def waiter(s: State) -> None:
        start = time.monotonic()
        await asyncio.wait_for(s.event.wait(), timeout=5.0)
        await asyncio.sleep(10.0)
        s.completed = True
        s.elapsed = time.monotonic() - start

    async def setter(s: State) -> None:
        await asyncio.sleep(1.0)
        s.event.set()

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[waiter, setter],
            invariant=lambda s: s.completed and s.elapsed >= 11.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_wait_for_bare_future_uses_virtual_deadline() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        start = time.monotonic()
        try:
            await asyncio.wait_for(fut, timeout=1.0)
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True
            s.elapsed = time.monotonic() - start

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.elapsed >= 1.0,
            clock="virtual",
            reproduce_on_failure=0,
            timeout_per_run=1.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 1.0, f"bare-future wait_for burned wall time ({wall_elapsed:.1f}s)"


def test_async_wait_for_cancels_inner_task_once() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.cancelled = 0

    async def never_finishes(s: State) -> None:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            s.cancelled += 1
            raise

    async def worker(s: State) -> None:
        try:
            await asyncio.wait_for(never_finishes(s), timeout=1.0)
        except (TimeoutError, asyncio.TimeoutError):
            s.timed_out = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.cancelled == 1,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_wait_for_returns_inner_value_when_cancel_is_suppressed() -> None:
    class State:
        def __init__(self) -> None:
            self.cancelled = False
            self.result = ""

    async def suppresses_cancel(s: State) -> str:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            s.cancelled = True
            return "cleaned"
        return "unexpected"

    async def worker(s: State) -> None:
        s.result = await asyncio.wait_for(suppresses_cancel(s), timeout=1.0)

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.cancelled and s.result == "cleaned",
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_zero_expires_and_clears_cancellation_state() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.cancelling_after_timeout = -1

    async def worker(s: State) -> None:
        try:
            async with asyncio.timeout(0):
                await asyncio.sleep(0)
        except TimeoutError:
            s.timed_out = True
            s.cancelling_after_timeout = asyncio.current_task().cancelling()  # type: ignore[union-attr]

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.cancelling_after_timeout == 0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_zero_without_await_does_not_cancel_later() -> None:
    class State:
        def __init__(self) -> None:
            self.completed = False
            self.cancelling_after_context = -1

    async def worker(s: State) -> None:
        async with asyncio.timeout(0):
            s.completed = True
        s.cancelling_after_context = asyncio.current_task().cancelling()  # type: ignore[union-attr]
        await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.completed and s.cancelling_after_context == 0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_positive_expiry_clears_cancellation_state() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.cancelling_after_timeout = -1

    async def worker(s: State) -> None:
        try:
            async with asyncio.timeout(1.0):
                await asyncio.sleep(10.0)
        except TimeoutError:
            s.timed_out = True
            s.cancelling_after_timeout = asyncio.current_task().cancelling()  # type: ignore[union-attr]

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.cancelling_after_timeout == 0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout_at"), reason="asyncio.timeout_at requires Python 3.11+")
def test_async_timeout_at_uses_loop_time_deadline() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        try:
            async with asyncio.timeout_at(loop.time() + 1.0):
                await asyncio.sleep(10.0)
        except TimeoutError:
            s.timed_out = True
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.elapsed == pytest.approx(1.0, abs=0.01),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_suppressed_cancel_clears_cancellation_state() -> None:
    class State:
        def __init__(self) -> None:
            self.cancelled = False
            self.completed = False
            self.cancelling_after_context = -1

    async def worker(s: State) -> None:
        async with asyncio.timeout(1.0):
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                s.cancelled = True
        s.completed = True
        s.cancelling_after_context = asyncio.current_task().cancelling()  # type: ignore[union-attr]

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.cancelled and s.completed and s.cancelling_after_context == 0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_context_uses_virtual_deadline_and_reports_expiry() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.expired = False
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        start = time.monotonic()
        timeout_cm: object | None = None
        try:
            async with asyncio.timeout(1.0) as active_timeout:
                timeout_cm = active_timeout
                await asyncio.sleep(10.0)
        except TimeoutError:
            s.timed_out = True
            s.elapsed = time.monotonic() - start
            s.expired = bool(timeout_cm is not None and getattr(timeout_cm, "expired")())

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.expired and s.elapsed == pytest.approx(1.0, abs=0.01),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_contexts_share_exact_virtual_deadline() -> None:
    """Two tasks entering ``asyncio.timeout(1.0)`` at virtual t=0 must get the
    *exact same* virtual deadline (one ``clock.now() + delay`` read, no
    ``loop.time()`` round-trip smearing real-time jitter into it).  With exact
    deadlines the DeadlineCoordinator's deterministic ``(deadline, actor_id,
    order)`` tie-break controls firing order, so identical runs observe the
    same order every time."""

    class State:
        def __init__(self) -> None:
            self.stamps: dict[str, float] = {}
            self.order: list[str] = []

    async def worker(name: str, s: State) -> None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(1.0):
                await asyncio.Event().wait()
        except TimeoutError:
            s.stamps[name] = time.monotonic() - start
            s.order.append(name)

    async def worker_a(s: State) -> None:
        await worker("a", s)

    async def worker_b(s: State) -> None:
        await worker("b", s)

    orders: list[tuple[str, ...]] = []

    def invariant(s: State) -> bool:
        orders.append(tuple(s.order))
        assert s.stamps.get("a") == 1.0, f"virtual deadline contaminated by real-time jitter: {s.stamps}"
        assert s.stamps.get("b") == 1.0, f"virtual deadline contaminated by real-time jitter: {s.stamps}"
        return True

    for _ in range(5):
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker_a, worker_b],
                invariant=invariant,
                clock="virtual",
                max_executions=1,
                reproduce_on_failure=0,
            )
        )
        assert result.property_holds, result.explanation

    assert len(set(orders)) == 1, f"timeout firing order flipped across identical runs: {orders}"


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_expiry_order_counterexample_replays_deterministically() -> None:
    """A counterexample that depends on the relative resume order of two
    identical ``asyncio.timeout(1.0)`` expiries must replay: exact virtual
    deadlines make the coordinator tie-break (and hence the recorded
    schedule's clock steps) deterministic across exploration and replay."""

    class State:
        def __init__(self) -> None:
            self.order: list[str] = []

    async def worker(name: str, s: State) -> None:
        try:
            async with asyncio.timeout(1.0):
                await asyncio.Event().wait()
        except TimeoutError:
            s.order.append(name)

    async def worker_a(s: State) -> None:
        await worker("a", s)

    async def worker_b(s: State) -> None:
        await worker("b", s)

    def invariant(s: State) -> bool:
        assert s.order != ["a", "b"], f"observed firing order {s.order}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker_a, worker_b],
            invariant=invariant,
            clock="virtual",
        )
    )
    _assert_invariant_failure(result, "observed firing order")
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_when_before_enter_matches_stdlib() -> None:
    """Stdlib ``asyncio.Timeout`` stores its loop-time deadline at
    construction, so ``when()`` reports it before ``__aenter__`` (``w`` for
    ``timeout_at(w)``, ``loop.time() + delay`` for ``timeout(delay)``, None
    for ``timeout(None)``); the virtual context must match."""

    class State:
        def __init__(self) -> None:
            self.at_when_ok = False
            self.rel_when_ok = False
            self.none_when_ok = False

    async def worker(s: State) -> None:
        loop = asyncio.get_running_loop()
        w = loop.time() + 5.0
        s.at_when_ok = asyncio.timeout_at(w).when() == w
        before = loop.time()
        cm = asyncio.timeout(5.0)
        after = loop.time()
        when = cm.when()
        s.rel_when_ok = when is not None and before + 5.0 <= when <= after + 5.0
        s.none_when_ok = asyncio.timeout(None).when() is None
        await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.at_when_ok and s.rel_when_ok and s.none_when_ok,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_deadline_starts_at_construction() -> None:
    class State:
        def __init__(self) -> None:
            self.elapsed_inside: float | None = None

    async def worker(s: State) -> None:
        timeout = asyncio.timeout(5.0)
        await asyncio.sleep(3.0)
        entered_at = time.monotonic()
        try:
            async with timeout:
                await asyncio.Event().wait()
        except TimeoutError:
            s.elapsed_inside = time.monotonic() - entered_at

    def invariant(s: State) -> bool:
        assert s.elapsed_inside == 2.0, f"timeout had {s.elapsed_inside}s left after entering 3s late"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=invariant,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_random_wait_for_bare_future_with_concurrent_worker() -> None:
    """Random strategy + virtual clock: a task parked in ``asyncio.wait_for``
    on a bare future must be registered as blocked so the schedule skips it
    and the clock advance fires its deadline.  Without that, the schedule
    stalls on the parked task's entries, the pause watchdog kills the
    *other* task mid-suspension after a real ``deadlock_timeout``, and the
    truncated run is scored as a completed interleaving (false
    counterexample / false pass)."""

    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.elapsed = 0.0
            self.b_steps = 0
            self.b_done = False

    async def parked(s: State) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        start = time.monotonic()
        try:
            await asyncio.wait_for(fut, timeout=1.0)
        except TimeoutError:
            s.timed_out = True
            s.elapsed = time.monotonic() - start

    async def runner(s: State) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
            s.b_steps += 1
        s.b_done = True

    def invariant(s: State) -> bool:
        assert s.timed_out and s.elapsed >= 1.0, f"parked task did not time out at its virtual deadline: {s.elapsed}"
        assert s.b_done and s.b_steps == 5, f"concurrent worker was cut short (steps={s.b_steps}, done={s.b_done})"
        return True

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[parked, runner],
            invariant=invariant,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=0,
            timeout_per_run=5.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 3.0, f"parked wait_for stalled the schedule for {wall_elapsed:.1f}s (watchdog rescue?)"


def test_async_random_wait_for_bare_future_uses_virtual_deadline() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        start = time.monotonic()
        try:
            await asyncio.wait_for(fut, timeout=1.0)
        except TimeoutError:
            s.timed_out = True
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.elapsed >= 1.0,
            strategy="random",
            clock="virtual",
            max_attempts=1,
            timeout_per_run=1.0,
            deadlock_timeout=0.05,
        )
    )
    assert result.property_holds, result.explanation


def test_async_explored_maybe_advance_clamps_to_earliest_pending_deadline() -> None:
    """Explored clock, random shuffler: a schedule entry landing on a sleeping
    task speculatively advances the clock ("maybe advance"), but each hop must
    stop at the *earliest* pending deadline — an earlier ``wait_for``-style
    timeout must fire at its own clock value, never at the later sleeper's
    target (virtual-clock hardening deferred item 2)."""
    from frontrun._virtual_clock import VirtualClock
    from frontrun.async_shuffler import AwaitScheduler

    clock = VirtualClock()
    epoch = clock.now()
    fired_at: list[float] = []

    class _TimeoutToken:
        def fire(self) -> None:
            fired_at.append(clock.now())

    async def drive() -> None:
        scheduler = AwaitScheduler([1, 0], num_tasks=2, virtual_clock=clock, clock_mode="explored")
        # Task 1 sleeps until t+5 (an asyncio.sleep); task 0 has an earlier
        # wait_for-style timeout deadline at t+1.
        scheduler._sleepers[1] = epoch + 5.0
        scheduler._deadlines.add_sleep(1, epoch + 5.0, wake_id=None)
        scheduler.add_timeout_deadline(0, epoch + 1.0, _TimeoutToken())
        async with scheduler._condition:
            # The schedule's head entry belongs to the sleeping task 1, so the
            # explored branch "maybe advances" the clock.
            scheduler.should_proceed(0)

    asyncio.run(drive())
    assert clock.now() == pytest.approx(epoch + 1.0)  # clamped: not the sleeper's t+5
    assert fired_at == [pytest.approx(epoch + 1.0)]  # the timeout fired at its own value


class _WaitForAndSleep:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.elapsed: float | None = None


async def _explored_wait_for_1(s: _WaitForAndSleep) -> None:
    start = time.monotonic()
    try:
        await asyncio.wait_for(s.event.wait(), timeout=1.0)  # nobody sets it
    except TimeoutError:
        pass
    s.elapsed = time.monotonic() - start


async def _explored_sleep_5(s: _WaitForAndSleep) -> None:
    await asyncio.sleep(5.0)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="pre-existing on 3.10/3.11, unrelated to the clamp: cancelling asyncio.Condition.wait "
    "(the scheduler's quiescence-slice timeout under this small deadlock_timeout) crashes with "
    "'Lock is not acquired'; CPython hardened Condition.wait cancellation in 3.12",
)
def test_async_random_explored_timed_wait_observes_own_deadline() -> None:
    """Random strategy, explored clock: the speculative "maybe advance" on the
    sleeper's entry must stop at ``wait_for``'s earlier t=1 deadline, so the
    wait times out (and is observed) at t=1 — never at the sleeper's t=5
    target (virtual-clock hardening deferred item 2).

    The seed is pinned so the maybe-advance hits while the t=1 deadline is
    pending; without the clamp the wait observes elapsed == 5.0.
    """
    observed: list[float] = []

    def invariant(s: _WaitForAndSleep) -> bool:
        if s.elapsed is not None:
            observed.append(s.elapsed)
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=_WaitForAndSleep,
            workers=[_explored_wait_for_1, _explored_sleep_5],
            invariant=invariant,
            strategy="random",
            clock="explored",
            seed=5,
            max_attempts=3,
            timeout_per_run=5.0,
            deadlock_timeout=0.2,
        )
    )
    assert result.property_holds, result.explanation
    if not observed:
        # An interpreter's interleaving may abort every attempt on the wall
        # watchdog before the timed wait resolves; the scheduler-level clamp
        # test above is the version-independent regression.
        pytest.skip("no attempt completed the timed wait under this interpreter's interleaving")
    for elapsed in observed:
        assert elapsed == pytest.approx(1.0), f"timed wait observed a later deadline's clock value: {elapsed}"


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_context_removes_deadline_after_success() -> None:
    class State:
        def __init__(self) -> None:
            self.completed = False
            self.expired_inside = True
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        start = time.monotonic()
        async with asyncio.timeout(5.0) as timeout_cm:
            await asyncio.sleep(1.0)
            s.expired_inside = timeout_cm.expired()
        await asyncio.sleep(10.0)
        s.completed = True
        s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.completed and not s.expired_inside and s.elapsed >= 11.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_context_reschedule_from_none_uses_virtual_deadline() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.expired = False
            self.deadline_delta = 0.0
            self.elapsed = 0.0

    async def worker(s: State) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        virtual_start = time.monotonic()
        timeout_cm: object | None = None
        try:
            async with asyncio.timeout(None) as active_timeout:
                timeout_cm = active_timeout
                active_timeout.reschedule(start + 1.0)
                deadline = active_timeout.when()
                s.deadline_delta = -1.0 if deadline is None else deadline - start
                await asyncio.sleep(10.0)
        except TimeoutError:
            s.timed_out = True
            s.elapsed = time.monotonic() - virtual_start
            s.expired = bool(timeout_cm is not None and getattr(timeout_cm, "expired")())

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: (
                s.timed_out
                and s.expired
                and s.deadline_delta == pytest.approx(1.0)
                and s.elapsed == pytest.approx(1.0, abs=0.01)
            ),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_context_cannot_reschedule_after_exit() -> None:
    class State:
        def __init__(self) -> None:
            self.reschedule_rejected = False

    async def worker(s: State) -> None:
        loop = asyncio.get_running_loop()
        async with asyncio.timeout(None) as timeout_cm:
            pass

        with pytest.raises(RuntimeError):
            timeout_cm.reschedule(loop.time())
        s.reschedule_rejected = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.reschedule_rejected,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="asyncio.timeout requires Python 3.11+")
def test_async_timeout_context_cannot_reschedule_after_expiry() -> None:
    class State:
        def __init__(self) -> None:
            self.timed_out = False
            self.reschedule_rejected = False

    async def worker(s: State) -> None:
        try:
            async with asyncio.timeout(0) as timeout_cm:
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    with pytest.raises(RuntimeError):
                        timeout_cm.reschedule(None)
                    s.reschedule_rejected = True
                    raise
        except TimeoutError:
            s.timed_out = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out and s.reschedule_rejected,
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


def test_timeout_cancelled_sleep_does_not_leak_deadline_into_deadlock_detection() -> None:
    """A wait_for timeout that cancels an in-flight sleep must scrub the sleep deadline.

    The cancellation is delivered while the task is parked in
    ``AsyncDporScheduler.sleep_until``; if the (task, _SLEEP_TOKEN) deadline and
    the ``_sleepers`` entry survive, a later genuine deadlock looks like it has
    a pending virtual deadline and is misreported as an inconclusive timeout
    instead of an exact deadlock counterexample.
    """

    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()

    async def timed_then_blocked(s: State) -> None:
        try:
            await asyncio.wait_for(asyncio.sleep(5), timeout=1)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        await s.event.wait()

    async def sleeper_then_blocked(s: State) -> None:
        await asyncio.sleep(10)
        await s.event.wait()

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[timed_then_blocked, sleeper_then_blocked],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
            timeout_per_run=2.0,
            deadlock_timeout=0.5,
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "Deadlock detected" in result.explanation, result.explanation
    assert "inconclusive" not in result.explanation


def test_async_task_crash_wakes_parked_event_waiter() -> None:
    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()

    async def waiter(s: State) -> None:
        await s.event.wait()

    async def crasher(s: State) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[waiter, crasher],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
            timeout_per_run=0.5,
            deadlock_timeout=0.05,
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "Task crash" in result.explanation
    assert "RuntimeError: boom" in result.explanation
    assert "inconclusive" not in result.explanation


def test_async_wait_for_lock_timeout_is_not_scored_as_deadlock() -> None:
    class State:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.holder_has_lock = asyncio.Event()
            self.release = asyncio.Event()
            self.timed_out = False
            self.holder_done = False

    async def holder(s: State) -> None:
        async with s.lock:
            s.holder_has_lock.set()
            await s.release.wait()
        s.holder_done = True

    async def waiter(s: State) -> None:
        await s.holder_has_lock.wait()
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


def test_async_dpor_plain_wall_timeout_is_not_reported_as_deadlock() -> None:
    from frontrun.async_dpor import _real_asyncio_sleep

    class State:
        def __init__(self) -> None:
            self.completed = False

    async def worker(s: State) -> None:
        await _real_asyncio_sleep(0.2)
        s.completed = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: not s.completed,
            clock="virtual",
            max_executions=1,
            reproduce_on_failure=0,
            timeout_per_run=0.05,
            deadlock_timeout=0.01,
        )
    )
    assert not result.property_holds
    assert result.counterexample is None
    assert result.explanation is not None
    assert "inconclusive" in result.explanation
    assert "Deadlock detected" not in result.explanation


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


def test_async_event_deadlock_rechecks_after_unrelated_user_timer() -> None:
    async def w1(s: _CrossEventState) -> None:
        await s.e1.wait()

    async def w2(s: _CrossEventState) -> None:
        await s.e2.wait()

    async def scenario() -> tuple[InterleavingResult, float]:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, lambda: None)
        wall_start = time.monotonic()
        result = await frontrun.explore(
            setup=_CrossEventState,
            workers=[w1, w2],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
            timeout_per_run=1.0,
            deadlock_timeout=0.2,
        )
        return result, time.monotonic() - wall_start

    result, wall_elapsed = asyncio.run(scenario())
    assert not result.property_holds
    assert "deadlock" in str(result.explanation).lower()
    assert wall_elapsed < 0.5, f"deadlock waited for timeout_per_run ({wall_elapsed:.1f}s)"


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


def test_async_event_set_clear_race_is_detected_without_waiters() -> None:
    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()

    async def clearer(s: State) -> None:
        await asyncio.sleep(0)
        s.event.clear()
        await asyncio.sleep(0)

    async def setter(s: State) -> None:
        await asyncio.sleep(0)
        s.event.set()
        await asyncio.sleep(0)

    def invariant(s: State) -> bool:
        assert s.event.is_set(), "expected event to remain set"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[clearer, setter],
            invariant=invariant,
            reproduce_on_failure=0,
        )
    )
    _assert_invariant_failure(result, "expected event to remain set")


def test_async_post_await_writes_are_attributed_to_resumed_step() -> None:
    class State:
        def __init__(self) -> None:
            self.value = 0

    async def write_one(s: State) -> None:
        await asyncio.sleep(0)
        s.value = 1
        await asyncio.sleep(0)

    async def write_two(s: State) -> None:
        await asyncio.sleep(0)
        s.value = 2
        await asyncio.sleep(0)

    def invariant(s: State) -> bool:
        assert s.value == 2, f"expected final value from write_two, got {s.value}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[write_one, write_two],
            invariant=invariant,
            reproduce_on_failure=0,
        )
    )
    _assert_invariant_failure(result, "expected final value from write_two")


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


def test_async_queue_get_deadlock_detected_exactly() -> None:
    class State:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def consumer(s: State) -> None:
        await s.queue.get()

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[consumer, consumer],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
            deadlock_timeout=0.2,
            timeout_per_run=1.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert result.explanation is not None
    assert "no virtual-clock deadline is pending" in result.explanation
    assert wall_elapsed < 0.8, f"queue deadlock took {wall_elapsed:.1f}s to report (wall fallback?)"


def test_async_queue_waiter_does_not_starve_virtual_sleep_autojump() -> None:
    class State:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[str] = asyncio.Queue()
            self.item = ""

    async def consumer(s: State) -> None:
        s.item = await s.queue.get()

    async def producer(s: State) -> None:
        await asyncio.sleep(1.0)
        await s.queue.put("ready")

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[consumer, producer],
            invariant=lambda s: s.item == "ready",
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"queue+sleeper took {wall_elapsed:.1f}s (autojump stall?)"


def test_async_bounded_queue_put_wake_is_schedulable() -> None:
    class State:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
            self.queue.put_nowait("initial")
            self.consumer_ran = asyncio.Event()
            self.producer_done = False
            self.observed_before_producer = False

    async def producer(s: State) -> None:
        await s.queue.put("replacement")
        s.producer_done = True

    async def consumer(s: State) -> None:
        await asyncio.sleep(1.0)
        await s.queue.get()
        s.consumer_ran.set()

    async def observer(s: State) -> None:
        await s.consumer_ran.wait()
        if not s.producer_done:
            s.observed_before_producer = True

    def invariant(s: State) -> bool:
        assert not s.observed_before_producer, "observer ran after queue space opened but before putter resumed"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[producer, consumer, observer],
            invariant=invariant,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    _assert_invariant_failure(result, "before putter resumed")


def test_async_queue_put_nowait_wakes_blocked_getter() -> None:
    class State:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[str] = asyncio.Queue()
            self.item = ""

    async def consumer(s: State) -> None:
        s.item = await s.queue.get()

    async def producer(s: State) -> None:
        await asyncio.sleep(1.0)
        s.queue.put_nowait("ready")

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[consumer, producer],
            invariant=lambda s: s.item == "ready",
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_queue_get_nowait_wakes_blocked_putter() -> None:
    class State:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
            self.queue.put_nowait("initial")
            self.put_done = False
            self.observed = ""

    async def producer(s: State) -> None:
        await s.queue.put("replacement")
        s.put_done = True

    async def consumer(s: State) -> None:
        await asyncio.sleep(1.0)
        s.observed = s.queue.get_nowait()
        await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[producer, consumer],
            invariant=lambda s: s.observed == "initial" and s.put_done,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_condition_patch_preserves_unmanaged_wait_notify() -> None:
    from frontrun.async_dpor import _patch_asyncio_queue_condition, _unpatch_asyncio_queue_condition

    async def scenario(*, notify_all: bool) -> bool:
        condition = asyncio.Condition()
        ready = asyncio.Event()
        woke = False

        async def waiter() -> None:
            nonlocal woke
            async with condition:
                ready.set()
                await condition.wait()
                woke = True

        task = asyncio.create_task(waiter())
        await ready.wait()
        async with condition:
            if notify_all:
                condition.notify_all()
            else:
                condition.notify()
        await asyncio.wait_for(task, timeout=1.0)
        return woke

    _patch_asyncio_queue_condition()
    try:
        assert asyncio.run(scenario(notify_all=False))
        assert asyncio.run(scenario(notify_all=True))
    finally:
        _unpatch_asyncio_queue_condition()


def test_async_condition_notify_all_wakes_mixed_waiters() -> None:
    from frontrun._async_autopause import _scheduler_var, _task_id_var
    from frontrun.async_dpor import _patch_asyncio_queue_condition, _unpatch_asyncio_queue_condition

    class FakeExecution:
        def block_thread(self, task_id: int) -> None:
            pass

        def unblock_thread(self, task_id: int) -> None:
            pass

    class FakeScheduler:
        def __init__(self) -> None:
            self.engine = object()
            self.execution = FakeExecution()
            self._event_blocked: set[int] = set()
            self._stable_ids = None
            self._error = None

        async def kick_stalled_schedule(self, task_id: int) -> None:
            pass

        async def wait_until_scheduled_after_block(self, task_id: int, reason: str) -> None:
            pass

        def report_task_sync(self, task_id: int, event_type: str, sync_id: int) -> None:
            pass

        def report_task_access(self, task_id: int, object_id: int, kind: str) -> None:
            pass

    async def scenario() -> tuple[bool, bool]:
        condition = asyncio.Condition()
        ready = asyncio.Event()
        ready_count = 0
        unmanaged_woke = False
        managed_woke = False
        scheduler = FakeScheduler()

        def mark_ready() -> None:
            nonlocal ready_count
            ready_count += 1
            if ready_count == 2:
                ready.set()

        async def unmanaged_waiter() -> None:
            nonlocal unmanaged_woke
            async with condition:
                mark_ready()
                await condition.wait()
                unmanaged_woke = True

        async def managed_waiter() -> None:
            nonlocal managed_woke
            scheduler_token = _scheduler_var.set(scheduler)
            task_token = _task_id_var.set(1)
            try:
                async with condition:
                    mark_ready()
                    await condition.wait()
                    managed_woke = True
            finally:
                _task_id_var.reset(task_token)
                _scheduler_var.reset(scheduler_token)

        unmanaged = asyncio.create_task(unmanaged_waiter())
        managed = asyncio.create_task(managed_waiter())
        await ready.wait()

        scheduler_token = _scheduler_var.set(scheduler)
        task_token = _task_id_var.set(2)
        try:
            async with condition:
                condition.notify_all()
        finally:
            _task_id_var.reset(task_token)
            _scheduler_var.reset(scheduler_token)

        await asyncio.wait_for(asyncio.gather(unmanaged, managed), timeout=1.0)
        return unmanaged_woke, managed_woke

    _patch_asyncio_queue_condition()
    try:
        assert asyncio.run(scenario()) == (True, True)
    finally:
        _unpatch_asyncio_queue_condition()


def test_async_condition_notify_one_wakes_exactly_one_waiter_first() -> None:
    class State:
        def __init__(self) -> None:
            self.condition = asyncio.Condition()
            self.ready_event = asyncio.Event()
            self.ready = 0
            self.woken: list[str] = []
            self.timed_out: list[str] = []
            self.remaining_after_first_notify = -1

    async def waiter(name: str, s: State) -> None:
        async with s.condition:
            s.ready += 1
            if s.ready == 2:
                s.ready_event.set()
            try:
                await asyncio.wait_for(s.condition.wait(), timeout=2.0)
                s.woken.append(name)
            except TimeoutError:
                s.timed_out.append(name)

    async def notifier(s: State) -> None:
        await s.ready_event.wait()
        async with s.condition:
            s.condition.notify(1)
            s.remaining_after_first_notify = len(getattr(s.condition, "_waiters"))

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            # The notify-one guarantee is "at most one waiter is woken" and
            # "notify(1) removes at most one waiter from the queue".  It is NOT
            # "exactly one wakes and one times out": under the virtual clock the
            # notifier may be starved past the 2.0s deadline (both time out), and
            # even a notified waiter can lose the wake if it cannot re-acquire the
            # lock before its own deadline — both are legitimate asyncio
            # interleavings (verified against stock asyncio).  What must never
            # happen is a double-wake.
            workers=[lambda s: waiter("a", s), lambda s: waiter("b", s), notifier],
            invariant=lambda s: (
                len(s.woken) <= 1 and s.remaining_after_first_notify <= 1 and len(s.woken) + len(s.timed_out) == 2
            ),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


class _AsyncHoldAndSleep:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.a_sleep_virtual = 0.0
        self.b_acquired = False


def test_async_sleep_while_holding_lock() -> None:
    """A task sleeping while holding an asyncio.Lock must autojump quickly."""

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

    invariant_checks = 0

    def invariant(s: _AsyncHoldAndSleep) -> bool:
        nonlocal invariant_checks
        invariant_checks += 1
        return s.b_acquired and s.a_sleep_virtual >= 1.0

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_AsyncHoldAndSleep,
            workers=[a, b],
            invariant=invariant,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=7,
            timeout_per_run=1.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert invariant_checks > 0
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

    invariant_checks = 0

    def invariant(s: _AsyncHoldAndSleep) -> bool:
        nonlocal invariant_checks
        invariant_checks += 1
        return s.b_acquired and s.a_sleep_virtual >= 1.0

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_AsyncHoldAndSleep,
            workers=[a, b],
            invariant=invariant,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=7,
            deadlock_timeout=0.05,
            timeout_per_run=1.0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert invariant_checks > 0
    assert wall_elapsed < 2.0, f"async random lock+sleep took {wall_elapsed:.1f}s"


def test_async_condition_wait_timeout_does_not_release_other_task_lock() -> None:
    """A ``cond.wait_for`` that times out while another task holds the lock must
    re-acquire the lock before propagating, not skip re-acquire and let the
    caller's ``async with cond:`` __aexit__ release the *other* task's lock."""

    class State:
        def __init__(self) -> None:
            self.cond = asyncio.Condition()
            self.a_timed_out = False
            self.b_in_critical = False
            self.b_done = False

    async def a(s: State) -> None:
        async with s.cond:
            try:
                await asyncio.wait_for(s.cond.wait(), timeout=0.5)
            except (TimeoutError, asyncio.TimeoutError):
                s.a_timed_out = True

    async def b(s: State) -> None:
        await asyncio.sleep(0.1)  # let A park in wait() first
        async with s.cond:  # hold the lock across A's timeout at t=0.5
            s.b_in_critical = True
            await asyncio.sleep(1.0)
            s.b_done = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[a, b],
            invariant=lambda s: s.a_timed_out and s.b_in_critical and s.b_done,
            clock="virtual",
            deadlock_timeout=2.0,
            timeout_per_run=3.0,
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation
    assert "Task crash" not in (result.explanation or ""), result.explanation


def test_async_condition_wait_for_predicate_timeout_uses_virtual_deadline() -> None:
    """The frontrun-extension ``Condition.wait_for(pred, timeout=...)`` must
    time out at its virtual deadline (zero wall time) inside explored tasks."""

    class State:
        def __init__(self) -> None:
            self.cond = asyncio.Condition()
            self.timed_out = False
            self.elapsed = 0.0

    async def waiter(s: State) -> None:
        start = time.monotonic()
        async with s.cond:
            try:
                await s.cond.wait_for(lambda: False, timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                s.timed_out = True
                s.elapsed = time.monotonic() - start

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[waiter],
            invariant=lambda s: s.timed_out and s.elapsed >= 1.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"condition wait_for timeout burned {wall_elapsed:.1f}s wall time"


def test_async_condition_wait_for_timeout_without_asyncio_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Condition.wait_for(pred, timeout=...)`` must not depend on
    ``asyncio.timeout``, which does not exist on Python 3.10 (the timeout
    guard must go through ``asyncio.wait_for`` instead)."""
    from frontrun._async_cooperative import _CooperativeAsyncCondition

    monkeypatch.delattr(asyncio, "timeout", raising=False)

    async def scenario() -> tuple[bool, object]:
        cond = _CooperativeAsyncCondition()
        timed_out = False
        async with cond:
            try:
                await cond.wait_for(lambda: False, timeout=0.05)
            except (TimeoutError, asyncio.TimeoutError):
                timed_out = True
        async with cond:
            satisfied = await cond.wait_for(lambda: True, timeout=1.0)
        return timed_out, satisfied

    timed_out, satisfied = asyncio.run(scenario())
    assert timed_out
    assert satisfied is True


def test_async_condition_wait_for_returns_last_predicate_result() -> None:
    """``Condition.wait_for`` must return the already-evaluated predicate
    result (stdlib behavior), not call the side-effecting predicate an extra
    time on the way out."""
    from frontrun._async_cooperative import _CooperativeAsyncCondition

    calls = 0

    def predicate() -> int:
        nonlocal calls
        calls += 1
        return calls

    async def scenario() -> int:
        cond = _CooperativeAsyncCondition()
        async with cond:
            return await cond.wait_for(predicate)

    assert asyncio.run(scenario()) == 1
    assert calls == 1


def test_async_condition_wait_for_returns_last_predicate_result_with_timeout() -> None:
    """Same as above for the ``timeout=`` path of the frontrun extension."""
    from frontrun._async_cooperative import _CooperativeAsyncCondition

    calls = 0

    def predicate() -> int:
        nonlocal calls
        calls += 1
        return calls

    async def scenario() -> int:
        cond = _CooperativeAsyncCondition()
        async with cond:
            return await cond.wait_for(predicate, timeout=1.0)

    assert asyncio.run(scenario()) == 1
    assert calls == 1


def test_async_default_condition_lock_is_engine_visible() -> None:
    """A default-constructed ``asyncio.Condition()`` under patching must use an
    engine-visible lock.  Otherwise a task contending on ``async with cond:``
    parks in a raw (engine-invisible) acquire while the engine still considers
    it runnable, and the contended interleavings die as inconclusive
    SchedulerTimeoutError instead of being explored.
    """

    class State:
        def __init__(self) -> None:
            self.cond = asyncio.Condition()
            self.value = 0

    async def worker(s: State) -> None:
        async with s.cond:
            tmp = s.value
            await asyncio.sleep(0)
            s.value = tmp + 1

    wall_start = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker, worker],
            invariant=lambda s: s.value == 2,
            clock="virtual",
            deadlock_timeout=1.0,
            timeout_per_run=2.0,
            reproduce_on_failure=0,
        )
    )
    wall_elapsed = time.monotonic() - wall_start
    # The condition lock serializes the read-modify-write, so no lost update.
    assert result.property_holds, result.explanation
    assert "timed out" not in (result.explanation or ""), result.explanation
    assert wall_elapsed < 6.0, f"default condition lock contention burned {wall_elapsed:.1f}s wall time"


def test_async_exact_deadlock_declines_with_live_external_thread() -> None:
    """A live external OS thread can still wake a parked task via
    ``loop.call_soon_threadsafe``, so exact-deadlock detection must decline
    to declare a deadlock while such a thread is alive (mirroring the sync
    scheduler's ``_has_live_external_threads`` guard).  Without the guard the
    single parked waiter is scored as a false DeadlockError.
    """
    import contextlib
    import threading

    threads: list[threading.Thread] = []

    class State:
        def __init__(self) -> None:
            self.event = asyncio.Event()
            self.woke = False

    async def waiter(s: State) -> None:
        loop = asyncio.get_running_loop()

        def _external() -> None:
            time.sleep(0.2)
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(s.event.set)

        thread = threading.Thread(target=_external, daemon=True)
        threads.append(thread)
        thread.start()
        await s.event.wait()
        s.woke = True

    try:
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[waiter],
                invariant=lambda s: s.woke,
                clock="virtual",
                reproduce_on_failure=0,
            )
        )
    finally:
        for thread in threads:
            thread.join(timeout=2.0)
    assert result.property_holds, result.explanation


def test_async_random_wait_for_timeout_cancels_longer_sleep() -> None:
    # Regression: the shuffler's autojump loop advanced the virtual clock
    # through EVERY pending deadline in one synchronous burst (each iteration
    # ended in ``continue`` with no await), so a fired wait_for timeout token
    # never got back to the event loop to cancel the inner sleep before the
    # sleep's own later deadline fired too. The inner coroutine ran code a
    # real timeout would have prevented, wait_for returned its result instead
    # of raising TimeoutError, and the unreachable final state was reported as
    # a counterexample — a false counterexample manufactured by frontrun's own
    # clock handling. (The DPOR strategy handles the same program correctly.)
    class State:
        def __init__(self) -> None:
            self.past_sleep = False
            self.result: object = None

    async def slow(s: State) -> str:
        await asyncio.sleep(5.0)
        s.past_sleep = True
        return "done"

    async def worker(s: State) -> None:
        try:
            s.result = ("returned", await asyncio.wait_for(slow(s), timeout=1.0))
        except (TimeoutError, asyncio.TimeoutError):
            s.result = "timeout"

    def invariant(s: State) -> bool:
        # Real asyncio semantics are deterministic for this single logical
        # task: at t=1 the timeout cancels slow() mid-sleep, so the sleep can
        # never complete and wait_for must raise.
        return s.result == "timeout" and not s.past_sleep

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=invariant,
            strategy="random",
            clock="virtual",
            max_attempts=3,
            seed=1,
        )
    )
    assert result.property_holds, f"false counterexample — the timeout was swallowed: {result.explanation}"
