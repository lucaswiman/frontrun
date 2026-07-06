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
from frontrun._virtual_clock import VirtualClock
from frontrun.bytecode import OpcodeScheduler, run_with_schedule

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


def test_run_with_schedule_virtual_clock_requires_patch_sleep() -> None:
    with pytest.raises(ValueError, match="patch_sleep"):
        run_with_schedule(
            [0],
            setup=lambda: None,
            threads=[lambda s: None],
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
    saved = (time.time, time.monotonic, time.perf_counter, time.time_ns, time.monotonic_ns, time.perf_counter_ns)

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
    assert (
        time.time,
        time.monotonic,
        time.perf_counter,
        time.time_ns,
        time.monotonic_ns,
        time.perf_counter_ns,
    ) == saved


def test_time_functions_restored_after_worker_exception() -> None:
    saved_monotonic = time.monotonic

    class State:
        pass

    def worker(s: State) -> None:
        time.sleep(1.0)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: True,
            clock="virtual",
            reproduce_on_failure=0,
        )
    assert time.monotonic is saved_monotonic


def test_invariant_sees_virtual_time() -> None:
    """The invariant is documented to see the same virtual time as workers.

    Regression: invariant evaluation happens after the workers' patch scope
    unwinds, so clock_scope must own the time.* patch itself.
    """
    observed: list[tuple[float, float]] = []

    class State:
        def __init__(self) -> None:
            self.worker_saw = 0.0

    def worker(s: State) -> None:
        time.sleep(5.0)
        s.worker_saw = time.monotonic()

    def invariant(s: State) -> bool:
        observed.append((s.worker_saw, time.monotonic()))
        return True

    result = frontrun.explore(
        setup=State,
        workers=[worker],
        invariant=invariant,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation
    worker_saw, invariant_saw = observed[0]
    assert worker_saw >= 1_000_000.0  # VIRTUAL_EPOCH
    assert invariant_saw >= worker_saw  # both virtual, monotonic


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
    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=lambda s: s.x == 100,
        clock="explored",
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert result.counterexample is not None
    # The recorded schedule contains clock-actor steps; replay must perform
    # the same advances without stalling on the deadlock timeout.
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10
    assert wall_elapsed < 30.0, f"exploration + 10 replays took {wall_elapsed:.1f}s (replay clock stall?)"


# ---------------------------------------------------------------------------
# Sleeping while holding a lock / signalling via events
# ---------------------------------------------------------------------------


class _HoldAndSleep:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.a_has_lock = threading.Event()
        self.a_sleep_virtual = 0.0
        self.b_acquired = False


def _hold_sleep_a(s: _HoldAndSleep) -> None:
    with s.lock:
        s.a_has_lock.set()
        start = time.monotonic()
        time.sleep(1.0)
        s.a_sleep_virtual = time.monotonic() - start


def _hold_sleep_b(s: _HoldAndSleep) -> None:
    s.a_has_lock.wait()
    with s.lock:
        s.b_acquired = True


def test_sleep_while_holding_lock_dpor() -> None:
    """A worker sleeping while holding a lock must not stall or deadlock:
    the contender engine-blocks (event wait + lock wait) and the clock
    autojumps.  Regression for the Event.wait spin that let DPOR schedule
    the waiter unboundedly (one burned deadlock timeout per branch)."""
    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_HoldAndSleep,
        workers=[_hold_sleep_a, _hold_sleep_b],
        invariant=lambda s: s.b_acquired and s.a_sleep_virtual >= 1.0,
        clock="virtual",
        reproduce_on_failure=0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"lock+sleep exploration took {wall_elapsed:.1f}s (deadlock-timeout stall?)"


def test_random_sleep_while_holding_lock() -> None:
    """Random strategy: an untimed lock spinner must not block the autojump
    (regression: sleep(1.0) silently returned with 0 virtual seconds)."""
    result = frontrun.explore(
        setup=_HoldAndSleep,
        workers=[_hold_sleep_a, _hold_sleep_b],
        invariant=lambda s: s.b_acquired and s.a_sleep_virtual >= 1.0,
        strategy="random",
        clock="virtual",
        max_attempts=3,
        seed=42,
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_event_deadlock_detected_exactly() -> None:
    """Two workers each waiting on the other's event is a genuine deadlock:
    with a virtual clock and no pending deadline it must be reported via
    exact detection, not by burning the wall-clock fallback timeout."""

    class State:
        def __init__(self) -> None:
            self.e1 = threading.Event()
            self.e2 = threading.Event()

    def w1(s: State) -> None:
        s.e1.wait()
        s.e2.set()

    def w2(s: State) -> None:
        s.e2.wait()
        s.e1.set()

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=State,
        workers=[w1, w2],
        invariant=lambda s: True,
        clock="virtual",
        reproduce_on_failure=0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert "deadlock" in str(result.explanation).lower()
    assert wall_elapsed < 4.0, f"deadlock took {wall_elapsed:.1f}s to report (wall-clock fallback?)"


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


def test_random_scheduler_autojumps_when_all_live_threads_are_timed_waits() -> None:
    clock = VirtualClock()
    scheduler = OpcodeScheduler([0], 2, virtual_clock=clock, clock_mode="virtual")
    first_deadline = clock.now() + 1.0
    scheduler.add_timed_wait(0, first_deadline)
    scheduler.add_timed_wait(1, clock.now() + 2.0)

    assert scheduler.wait_for_turn(0)
    assert clock.now() == first_deadline


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
