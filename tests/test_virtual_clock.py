"""Tests for the virtual clock (``clock="virtual"`` / ``clock="explored"``).

Covers the design in ``ideas/virtual_clock.md``:

- v1 autojump: sleeps cost zero wall time, TTL expiry is reachable, wake
  order follows deadlines, timed lock acquires resolve virtually.
- v2 explored clock actor: "the timer fired between your read and your
  write" becomes a schedulable interleaving that DPOR can find.
"""

from __future__ import annotations

import threading
import time

import pytest

import frontrun

# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_explore_rejects_unknown_clock() -> None:
    with pytest.raises(ValueError, match="clock"):
        frontrun.explore(
            setup=lambda: None,
            workers=[lambda s: None],
            invariant=lambda s: True,
            clock="banana",  # type: ignore[arg-type]
        )


def test_virtual_clock_requires_patch_sleep() -> None:
    with pytest.raises(ValueError, match="patch_sleep"):
        frontrun.explore(
            setup=lambda: None,
            workers=[lambda s: None],
            invariant=lambda s: True,
            clock="virtual",
            patch_sleep=False,
        )


def test_process_execution_rejects_virtual_clock() -> None:
    with pytest.raises(ValueError, match="clock"):
        frontrun.explore(
            setup=lambda: None,
            workers=[lambda s: None],
            invariant=lambda s: True,
            execution="process",
            clock="virtual",
        )


def test_time_functions_restored_after_exploration() -> None:
    saved_monotonic = time.monotonic
    saved_time = time.time

    class State:
        pass

    result = frontrun.explore(
        setup=State,
        workers=[lambda s: time.sleep(10.0)],
        invariant=lambda s: True,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds
    assert time.monotonic is saved_monotonic
    assert time.time is saved_time


# ---------------------------------------------------------------------------
# v1 autojump — sync DPOR
# ---------------------------------------------------------------------------


class _SleepObserver:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0


def test_sleep_advances_virtual_clock_with_zero_wall_time() -> None:
    def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        time.sleep(500.0)
        s.end = time.monotonic()

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_SleepObserver,
        workers=[worker],
        invariant=lambda s: s.end - s.start >= 500.0,
        clock="virtual",
        reproduce_on_failure=0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 60.0  # the 500 s sleep must not be a real sleep


class _TTLCache:
    """Minimal TTL cache: values expire ``ttl`` seconds after being set."""

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


def test_ttl_cache_expiry_is_reachable() -> None:
    """Sleeping past a TTL must actually expire the entry (proposal phase 1)."""

    def worker(cache: _TTLCache) -> None:
        cache.set("k", "v", ttl=1.0)
        assert cache.get("k") == "v"
        time.sleep(2.0)
        cache.observed_miss_after_expiry = cache.get("k") is None

    result = frontrun.explore(
        setup=_TTLCache,
        workers=[worker],
        invariant=lambda c: c.observed_miss_after_expiry,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


class _WakeOrder:
    def __init__(self) -> None:
        self.order: list[str] = []


def test_sleepers_wake_in_deadline_order() -> None:
    def short(s: _WakeOrder) -> None:
        time.sleep(1.0)
        s.order.append("short")

    def long(s: _WakeOrder) -> None:
        time.sleep(5.0)
        s.order.append("long")

    result = frontrun.explore(
        setup=_WakeOrder,
        workers=[long, short],
        invariant=lambda s: s.order == ["short", "long"],
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# v2 explored clock actor — the timer fires *between* a read and a write
# ---------------------------------------------------------------------------


class _RetryRace:
    """A tenacity-style race: a delayed retry writes concurrently with RMW."""

    def __init__(self) -> None:
        self.x = 0


def _rmw_worker(s: _RetryRace) -> None:
    tmp = s.x
    s.x = tmp + 1


def _delayed_writer(s: _RetryRace) -> None:
    time.sleep(1.0)
    s.x = 100


def test_autojump_does_not_explore_early_timer_fire() -> None:
    """Autojump advances time as late as possible: the delayed write always
    lands after the RMW worker finished, so the invariant holds."""
    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=lambda s: s.x == 100,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_explored_clock_finds_timer_between_read_and_write() -> None:
    """With clock="explored" the clock advance is a schedulable DPOR step:
    the engine must find the interleaving where the delayed write fires
    between the RMW worker's read and write (final x == 1, not 100)."""
    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=lambda s: s.x == 100,
        clock="explored",
    )
    assert not result.property_holds
    assert result.counterexample is not None


# ---------------------------------------------------------------------------
# Timed lock acquires (proposal phase 4)
# ---------------------------------------------------------------------------


class _TimedAcquireState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.holder_has_lock = threading.Event()
        self.acquire_result: bool | None = None
        self.waited_virtual: float = 0.0


def test_timed_acquire_times_out_on_virtual_deadline() -> None:
    def holder(s: _TimedAcquireState) -> None:
        with s.lock:
            s.holder_has_lock.set()
            time.sleep(100.0)

    def contender(s: _TimedAcquireState) -> None:
        s.holder_has_lock.wait()
        start = time.monotonic()
        s.acquire_result = s.lock.acquire(timeout=1.0)
        s.waited_virtual = time.monotonic() - start
        if s.acquire_result:
            s.lock.release()

    result = frontrun.explore(
        setup=_TimedAcquireState,
        workers=[holder, contender],
        invariant=lambda s: s.acquire_result is False and s.waited_virtual >= 1.0,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


def test_random_strategy_virtual_sleep_zero_wall_time() -> None:
    def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        time.sleep(300.0)
        s.end = time.monotonic()

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_SleepObserver,
        workers=[worker, lambda s: None],
        invariant=lambda s: s.end - s.start >= 300.0,
        strategy="random",
        clock="virtual",
        max_attempts=5,
        reproduce_on_failure=0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 60.0


def test_random_strategy_explored_clock_can_fire_timer_early() -> None:
    """The random scheduler's "maybe advance" branch: with clock="explored",
    schedule entries landing on a sleeping thread advance the clock, so the
    delayed write can land between the RMW read and write."""
    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=lambda s: s.x == 100,
        strategy="random",
        clock="explored",
        max_attempts=200,
        seed=1234,
        reproduce_on_failure=0,
    )
    assert not result.property_holds
