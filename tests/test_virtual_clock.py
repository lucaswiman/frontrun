"""Tests for the virtual clock (``clock="virtual"`` / ``clock="explored"``).

Covers the design in ``ideas/virtual_clock.md``:

- autojump: sleeps cost zero wall time, TTL expiry is reachable, wake order
  follows deadlines, and timed lock acquires resolve virtually.
- explored clock actor: "the timer fired between your read and your write"
  becomes a schedulable interleaving that DPOR can find.
"""

from __future__ import annotations

import datetime as dt
import queue
import sys
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import pytest

import frontrun
from frontrun._cooperative import _real_time_sleep as _real_sleep
from frontrun._virtual_clock import VIRTUAL_EPOCH, VirtualClock
from frontrun.bytecode import OpcodeScheduler, run_with_schedule
from frontrun.common import InterleavingResult

# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _assert_invariant_failure(result: InterleavingResult, expected: str | None = None) -> None:
    assert not result.property_holds
    explanation = result.explanation
    assert explanation is not None
    assert "Deadlock" not in explanation
    assert "Task crash" not in explanation
    if expected is not None:
        assert expected in explanation


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


def test_explore_rejects_serializable_invariant_with_virtual_clock() -> None:
    # The sequential baseline runs execute outside the scheduler, so their
    # sleeps and clock reads would use real wall-clock time; the combination
    # must be rejected up front, not silently produce a mixed-clock baseline.
    with pytest.raises(ValueError, match="serializable_invariant"):
        frontrun.explore(
            setup=lambda: None,
            workers=[lambda s: None],
            invariant=lambda s: True,
            clock="virtual",
            serializable_invariant=True,
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
    saved = (
        time.time,
        time.monotonic,
        time.perf_counter,
        time.time_ns,
        time.monotonic_ns,
        time.perf_counter_ns,
        dt.datetime,
        dt.date,
    )

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
        dt.datetime,
        dt.date,
    ) == saved


def test_time_functions_restored_after_worker_exception() -> None:
    saved = (time.monotonic, dt.datetime, dt.date)

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
    assert (time.monotonic, dt.datetime, dt.date) == saved


def test_invariant_sees_virtual_time() -> None:
    """The invariant is documented to see the same virtual time as workers."""
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
    assert worker_saw == pytest.approx(VIRTUAL_EPOCH + 5.0)
    assert invariant_saw == pytest.approx(worker_saw)


# ---------------------------------------------------------------------------
# Autojump — sync DPOR
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
    assert wall_elapsed < 4.0  # the 500 s sleep must not be a real sleep


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
    """Sleeping past a TTL must actually expire the entry."""

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


class _DatetimeObserver:
    def __init__(self) -> None:
        self.start = dt.datetime.min
        self.end = dt.datetime.min
        self.utc = dt.datetime.min
        self.aware = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        self.today = dt.date.min
        self.now_is_datetime = False
        self.utc_is_datetime = False
        self.today_is_date = False


def test_datetime_patch_preserves_isinstance_for_existing_objects() -> None:
    existing_datetime = dt.datetime(2026, 1, 2, 3, 4, 5)
    existing_date = dt.date(2026, 1, 2)

    class State:
        def __init__(self) -> None:
            self.datetime_ok = False
            self.date_ok = False

    def worker(s: State) -> None:
        s.datetime_ok = isinstance(existing_datetime, dt.datetime)
        s.date_ok = isinstance(existing_date, dt.date)

    result = frontrun.explore(
        setup=State,
        workers=[worker],
        invariant=lambda s: s.datetime_ok and s.date_ok,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_datetime_now_advances_with_virtual_sleep() -> None:
    def worker(s: _DatetimeObserver) -> None:
        s.start = dt.datetime.now()
        time.sleep(2.5)
        s.end = dt.datetime.now()
        s.now_is_datetime = isinstance(s.end, dt.datetime)

    result = frontrun.explore(
        setup=_DatetimeObserver,
        workers=[worker],
        invariant=lambda s: s.now_is_datetime and (s.end - s.start).total_seconds() >= 2.5,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_datetime_utcnow_reads_virtual_time() -> None:
    def worker(s: _DatetimeObserver) -> None:
        s.utc = dt.datetime.utcnow()
        s.utc_is_datetime = isinstance(s.utc, dt.datetime)

    result = frontrun.explore(
        setup=_DatetimeObserver,
        workers=[worker],
        invariant=lambda s: (
            s.utc_is_datetime and s.utc.replace(tzinfo=dt.timezone.utc).timestamp() == pytest.approx(VIRTUAL_EPOCH)
        ),
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_datetime_now_timezone_reads_virtual_time() -> None:
    def worker(s: _DatetimeObserver) -> None:
        s.aware = dt.datetime.now(dt.timezone.utc)

    result = frontrun.explore(
        setup=_DatetimeObserver,
        workers=[worker],
        invariant=lambda s: s.aware.tzinfo is dt.timezone.utc and s.aware.timestamp() == pytest.approx(VIRTUAL_EPOCH),
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_date_today_reads_virtual_time() -> None:
    expected = dt.date.fromtimestamp(VIRTUAL_EPOCH)

    def worker(s: _DatetimeObserver) -> None:
        s.today = dt.date.today()
        s.today_is_date = isinstance(s.today, dt.date)

    result = frontrun.explore(
        setup=_DatetimeObserver,
        workers=[worker],
        invariant=lambda s: s.today_is_date and s.today == expected,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


def test_clock_diagnostics_warn_for_captured_time_reference() -> None:
    captured_monotonic = time.monotonic

    class State:
        def __init__(self) -> None:
            self.value = 0.0

    def worker(s: State, monotonic: Callable[[], float] = captured_monotonic) -> None:
        s.value = monotonic()

    with pytest.warns(RuntimeWarning, match="captured.*time\\.monotonic"):
        result = frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.value > 0,
            clock="virtual",
            clock_diagnostics=True,
            reproduce_on_failure=0,
        )
    assert result.property_holds, result.explanation


def test_clock_diagnostics_ignore_module_qualified_time_calls() -> None:
    class State:
        pass

    def worker(s: State) -> None:
        time.monotonic()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: True,
            clock="virtual",
            clock_diagnostics=True,
            reproduce_on_failure=0,
        )
    assert result.property_holds, result.explanation
    assert not [warning for warning in caught if "captured" in str(warning.message)]


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
# Explored clock actor — the timer fires *between* a read and a write
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

    def invariant(s: _RetryRace) -> bool:
        assert s.x == 100, f"expected delayed writer to remain final, got x={s.x}"
        return True

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=invariant,
        clock="explored",
    )
    wall_elapsed = time.monotonic() - wall_start
    _assert_invariant_failure(result, "expected delayed writer")
    assert result.counterexample is not None
    # The recorded schedule contains clock-actor steps; replay must perform
    # the same advances without stalling on the deadlock timeout.
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10
    assert wall_elapsed < 10.0, f"exploration + 10 replays took {wall_elapsed:.1f}s (replay clock stall?)"


class _TimerCascadeRace:
    def __init__(self) -> None:
        self.x = 0


def _early_timer_increments(s: _TimerCascadeRace) -> None:
    time.sleep(1.0)
    s.x += 1


def _later_timer_writes(s: _TimerCascadeRace) -> None:
    time.sleep(2.0)
    s.x = 100


def test_explored_clock_can_fire_later_timer_before_earlier_sleeper_resumes() -> None:
    """After the first deadline wakes a sleeper, the clock actor must remain
    schedulable while later deadlines are still pending. Otherwise DPOR misses
    races where a later timer fires before an earlier sleeper gets CPU."""

    def invariant(s: _TimerCascadeRace) -> bool:
        assert s.x == 100, f"expected later timer write to remain final, got x={s.x}"
        return True

    result = frontrun.explore(
        setup=_TimerCascadeRace,
        workers=[_early_timer_increments, _later_timer_writes],
        invariant=invariant,
        clock="explored",
    )
    _assert_invariant_failure(result, "expected later timer write")
    assert result.counterexample is not None
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10


class _EqualDeadlineRace:
    def __init__(self) -> None:
        self.x = 0


def _equal_deadline_rmw(s: _EqualDeadlineRace) -> None:
    time.sleep(1.0)
    tmp = s.x
    s.x = tmp + 1


def test_equal_deadline_sleepers_remain_raceable_after_wake() -> None:
    """Waking equal-deadline sleepers in one clock step must not serialize
    their continuations by worker id; their post-wake shared-state accesses
    still need normal DPOR race exploration."""

    def invariant(s: _EqualDeadlineRace) -> bool:
        assert s.x == 2, f"expected both equal-deadline increments, got x={s.x}"
        return True

    result = frontrun.explore(
        setup=_EqualDeadlineRace,
        workers=[_equal_deadline_rmw, _equal_deadline_rmw],
        invariant=invariant,
        clock="explored",
    )
    _assert_invariant_failure(result, "expected both equal-deadline increments")
    assert result.counterexample is not None
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10


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
    needed for the holder's virtual sleep."""
    invariant_checks = 0

    def invariant(s: _HoldAndSleep) -> bool:
        nonlocal invariant_checks
        invariant_checks += 1
        return s.b_acquired and s.a_sleep_virtual >= 1.0

    result = frontrun.explore(
        setup=_HoldAndSleep,
        workers=[_hold_sleep_a, _hold_sleep_b],
        invariant=invariant,
        strategy="random",
        clock="virtual",
        max_attempts=3,
        seed=42,
        reproduce_on_failure=0,
        timeout_per_run=2.0,
    )
    assert result.property_holds, result.explanation
    assert invariant_checks > 0


class _SleepAndTimedWait:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.t1_elapsed: float | None = None


def _sleep_20(s: _SleepAndTimedWait) -> None:
    time.sleep(20.0)


def _wait_10(s: _SleepAndTimedWait) -> None:
    start = time.monotonic()
    s.event.wait(timeout=10.0)  # nobody sets the event; must time out at t=10
    s.t1_elapsed = time.monotonic() - start


def test_random_timed_wait_not_double_advanced_past_deadline() -> None:
    """Random strategy, autojump clock: a virtual timed wait
    (event.wait(timeout=10)) must observe exactly its own deadline elapsing,
    even when another thread sleeps to a *later* deadline.

    Only ``clock="virtual"`` (autojump, time advances as late as possible) is
    asserted here: under ``clock="explored"`` the clock advance is itself a
    schedulable step, so after the wait correctly times out at t=10 the
    explorer may *legitimately* let the 20s sleeper's timer fire before the
    waiter reads ``monotonic()`` again — observing 20s there is a valid
    interleaving, not the double-advance defect.
    """

    def invariant(s: _SleepAndTimedWait) -> bool:
        # If T1 finished its timed wait, it must have observed exactly its own
        # 10s deadline, never the 20s sleeper's deadline.
        return s.t1_elapsed is None or abs(s.t1_elapsed - 10.0) < 1e-9

    result = frontrun.explore(
        setup=_SleepAndTimedWait,
        workers=[_sleep_20, _wait_10],
        invariant=invariant,
        strategy="random",
        clock="virtual",
        max_attempts=8,
        seed=2,
        reproduce_on_failure=0,
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation


def test_explored_maybe_advance_clamps_to_earliest_pending_deadline() -> None:
    """Explored clock, random scheduler: a schedule entry landing on a
    sleeping thread speculatively advances the clock ("maybe advance"), but
    each hop must stop at the *earliest* pending deadline — an earlier timed
    wait must fire at its own clock value, never at the later sleeper's
    target (virtual-clock hardening deferred item 2)."""
    clock = VirtualClock()
    epoch = clock.now()
    scheduler = OpcodeScheduler(
        [1, 0],
        num_threads=2,
        deadlock_timeout=0.2,
        virtual_clock=clock,
        clock_mode="explored",
    )
    # Thread 1 sleeps until t+5 (a time.sleep); thread 0 spins on a timed
    # wait with the earlier deadline t+1 (an event.wait(timeout=1)).
    scheduler._deadlines.add_sleep(1, epoch + 5.0, wake_id=None)
    scheduler.add_timed_wait(0, epoch + 1.0)

    # Thread 0 asks for a turn; the schedule's head entry belongs to the
    # sleeping thread 1, so the explored branch "maybe advances" the clock.
    granted = scheduler.wait_for_turn(0)

    assert clock.now() == pytest.approx(epoch + 1.0)  # clamped: not the sleeper's t+5
    assert not scheduler._deadlines.in_timed_wait(0)  # the timed wait fired, at its own value
    assert scheduler._deadlines.is_sleeping(1)  # the sleeper's own deadline is still pending
    # The clamped hop consumed the sleeper's entries, so the woken thread 0
    # got its turn (at t+1) instead of deadlocking behind the sleeper's slot.
    assert granted


class _ExploredTimedWaitAndSleep:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.elapsed: float | None = None


def _explored_wait_1(s: _ExploredTimedWaitAndSleep) -> None:
    start = time.monotonic()
    s.event.wait(timeout=1.0)  # nobody sets the event; must time out at t=1
    s.elapsed = time.monotonic() - start


def _explored_sleep_5(s: _ExploredTimedWaitAndSleep) -> None:
    time.sleep(5.0)


def _explored_busy(s: _ExploredTimedWaitAndSleep) -> None:
    # A runnable third thread keeps the all-blocked autojump (which already
    # fires deadlines in order) out of the way, so the speculative explored
    # "maybe advance" on the sleeper's schedule entries is the clock mover.
    x = 0
    for _ in range(1500):
        x += 1


@pytest.mark.skipif(
    sys.version_info[:2] > (3, 13),
    reason="seed-pinned interleaving verified on 3.10-3.13; the opcode stream differs on newer "
    "interpreters, where the post-timeout read may be legitimately preempted by a later advance "
    "(the scheduler-level clamp test above is the version-independent regression)",
)
@pytest.mark.parametrize("seed", [21, 88])
def test_random_explored_timed_wait_observes_own_deadline(seed: int) -> None:
    """Random strategy, explored clock: the speculative "maybe advance" on the
    sleeper's entry must stop at the timed wait's earlier t=1 deadline, so the
    wait times out (and is observed) at t=1 — never at the sleeper's t=5
    target (virtual-clock hardening deferred item 2).

    Seeds are pinned so the maybe-advance hits while the t=1 deadline is
    pending and the waiter's post-timeout read runs before any further
    advance; without the clamp both seeds observe elapsed == 5.0.
    """
    observed: list[float] = []

    def invariant(s: _ExploredTimedWaitAndSleep) -> bool:
        if s.elapsed is not None:
            observed.append(s.elapsed)
        return True

    result = frontrun.explore(
        setup=_ExploredTimedWaitAndSleep,
        workers=[_explored_wait_1, _explored_sleep_5, _explored_busy],
        invariant=invariant,
        strategy="random",
        clock="explored",
        seed=seed,
        max_attempts=1,
        detect_io=False,
        reproduce_on_failure=0,
        timeout_per_run=10.0,
    )
    assert result.property_holds, result.explanation
    assert observed, "the waiter never finished its timed wait"
    for elapsed in observed:
        assert elapsed == pytest.approx(1.0), f"timed wait observed a later deadline's clock value: {elapsed}"


class _ExhaustedSleeper:
    def __init__(self) -> None:
        self.expires_at = time.monotonic() + 60.0
        self.saw_expired: bool | None = None


def _burn_budget_then_sleep(s: _ExhaustedSleeper) -> None:
    # Burn well past the scheduler's max_ops cap (len(schedule) * 10 + 10000)
    # so the schedule budget is exhausted before the sleep is reached.
    x = 0
    for _ in range(20000):
        x += 1
    time.sleep(120.0)  # must advance the virtual clock past expires_at
    s.saw_expired = time.monotonic() >= s.expires_at


@pytest.mark.parametrize("clock", ["virtual", "explored"])
def test_random_max_ops_exhaustion_still_advances_virtual_sleep(clock: str) -> None:
    """Random strategy: a virtual sleep reached after the schedule/op budget
    is exhausted must still advance the clock to its deadline.  Returning
    instantly with the clock frozen silently truncates the sleep and reports
    a phantom TTL counterexample."""
    result = frontrun.explore(
        setup=_ExhaustedSleeper,
        workers=[_burn_budget_then_sleep],
        invariant=lambda s: s.saw_expired is True,
        strategy="random",
        clock=clock,  # type: ignore[arg-type]
        seed=0,
        max_attempts=1,
        max_ops=10,
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


class _ExhaustedTimedWait:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.elapsed: float | None = None


def _burn_budget_then_timed_wait(s: _ExhaustedTimedWait) -> None:
    x = 0
    for _ in range(20000):
        x += 1
    start = time.monotonic()
    s.event.wait(timeout=10.0)  # nobody sets the event; must time out at t=10
    s.elapsed = time.monotonic() - start


def test_random_max_ops_exhaustion_event_wait_times_out_virtually() -> None:
    """Random strategy: a timed Event.wait reached after budget exhaustion
    must resolve on the virtual clock (observe its own 10s deadline), not
    degrade to a real 1-second wait with the clock frozen."""
    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_ExhaustedTimedWait,
        workers=[_burn_budget_then_timed_wait],
        invariant=lambda s: s.elapsed is not None and s.elapsed >= 10.0,
        strategy="random",
        clock="virtual",
        seed=0,
        max_attempts=1,
        max_ops=10,
        detect_io=False,
        reproduce_on_failure=0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 4.0, f"exhausted timed wait burned wall time ({wall_elapsed:.1f}s)"


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


def test_queue_deadlock_detected_exactly() -> None:
    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()

    def worker(s: State) -> None:
        s.q.get()

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=State,
        workers=[worker, worker],
        invariant=lambda s: True,
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=2.0,
        timeout_per_run=3.0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert result.explanation is not None
    assert "no virtual-clock deadline is pending" in result.explanation
    assert wall_elapsed < 1.0, f"queue deadlock took {wall_elapsed:.1f}s to report (wall-clock fallback?)"


def test_event_deadlock_reproduces_under_virtual_clock() -> None:
    """An exact virtual-clock deadlock must reproduce N/N under replay.

    The replay schedulers historically had no exact-deadlock detection:
    ``ReplayExecution.block_thread`` is a no-op, so Event waiters spun
    through the positional schedule, exhausted the op budget, and died with
    a plain ``TimeoutError`` after ``deadlock_timeout`` — every reproduction
    attempt scored 0 and burned ~``deadlock_timeout`` of wall time.
    """

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
        deadlock_timeout=2.0,
        timeout_per_run=3.0,
        reproduce_on_failure=3,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert "deadlock" in str(result.explanation).lower()
    assert result.reproduction_attempts == 3
    assert result.reproduction_successes == 3, (
        f"Event-cycle deadlock reproduced {result.reproduction_successes}/{result.reproduction_attempts}; "
        "replay must detect the exact deadlock and raise DeadlockError"
    )
    assert wall_elapsed < 5.0, f"deadlock reproduction took {wall_elapsed:.1f}s (deadlock_timeout burned per attempt?)"


def test_queue_deadlock_reproduces_under_virtual_clock() -> None:
    """queue.get() cycle variant of the reproduction test above."""

    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()

    def worker(s: State) -> None:
        s.q.get()

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=State,
        workers=[worker, worker],
        invariant=lambda s: True,
        clock="virtual",
        deadlock_timeout=2.0,
        timeout_per_run=3.0,
        reproduce_on_failure=3,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert not result.property_holds
    assert "deadlock" in str(result.explanation).lower()
    assert result.reproduction_attempts == 3
    assert result.reproduction_successes == 3, (
        f"queue.get() deadlock reproduced {result.reproduction_successes}/{result.reproduction_attempts}; "
        "replay must detect the exact deadlock and raise DeadlockError"
    )
    assert wall_elapsed < 5.0, f"deadlock reproduction took {wall_elapsed:.1f}s (deadlock_timeout burned per attempt?)"


def test_spin_flag_registry_purge_helper() -> None:
    """_purge_spin_schedulers drops exactly the entries recorded against one scheduler."""
    from frontrun import _cooperative as coop

    sched_a = object()
    sched_b = object()
    coop._record_spin_scheduler(101, 0, sched_a)
    coop._record_spin_scheduler(101, 1, sched_b)
    coop._record_spin_scheduler(202, 0, sched_a)
    try:
        coop._purge_spin_schedulers(sched_a)
        assert coop._spin_schedulers_for(101) == [sched_b]
        assert coop._spin_schedulers_for(202) == []
        assert 202 not in coop._spin_flag_schedulers  # empty per-resource dicts are dropped
        coop._purge_spin_schedulers(sched_b)
        assert coop._spin_schedulers_for(101) == []
    finally:
        # Never leave test fixtures behind in the module-global registry.
        coop._purge_spin_schedulers(sched_a)
        coop._purge_spin_schedulers(sched_b)


def test_spin_flag_registry_does_not_retain_dpor_schedulers() -> None:
    """The module-global spin-flag registry must not keep DPOR schedulers alive.

    Entries are keyed by ``id(primitive)``, so a stale entry can alias a new
    primitive that reuses the id — and each entry strongly references its
    scheduler.  Scheduler teardown must purge its entries.
    """
    from frontrun import _cooperative as coop
    from frontrun._dpor_runtime.scheduler import DporScheduler

    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()
            self.timed_out = False

    def worker(s: State) -> None:
        # Untimed spin flags come from the queue poll; the timeout exercises
        # the timed-wait flag path as well.
        try:
            s.q.get(timeout=0.5)
        except queue.Empty:
            s.timed_out = True

    result = frontrun.explore(
        setup=State,
        workers=[worker],
        invariant=lambda s: s.timed_out,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation
    leftover = [sched for per_resource in coop._spin_flag_schedulers.values() for sched in per_resource.values()]
    assert not [s for s in leftover if isinstance(s, DporScheduler)], (
        "DPOR schedulers leaked in _spin_flag_schedulers after explore() finished"
    )


def test_event_set_clear_race_is_detected_without_waiters() -> None:
    class State:
        def __init__(self) -> None:
            self.event = threading.Event()

    def clearer(s: State) -> None:
        s.event.clear()

    def setter(s: State) -> None:
        s.event.set()

    def invariant(s: State) -> bool:
        assert s.event.is_set(), "expected event to remain set"
        return True

    result = frontrun.explore(
        setup=State,
        workers=[clearer, setter],
        invariant=invariant,
        detect_io=False,
        reproduce_on_failure=0,
    )
    _assert_invariant_failure(result, "expected event to remain set")


def test_event_wait_can_be_woken_by_unmanaged_thread() -> None:
    from frontrun._cooperative import _real_time_sleep

    setter_threads: list[threading.Thread] = []

    class State:
        def __init__(self) -> None:
            self.event = threading.Event()
            self.woke = False

    def worker(s: State) -> None:
        def setter() -> None:
            setter_started.set()
            _real_time_sleep(0.05)
            s.event.set()

        setter_started = threading.Event()
        t = threading.Thread(target=setter)
        setter_threads.append(t)
        t.start()
        assert setter_started.wait(timeout=1.0)
        s.woke = s.event.wait()
        t.join(timeout=1.0)

    try:
        result = frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.woke,
            clock="virtual",
            reproduce_on_failure=0,
            deadlock_timeout=1.0,
            timeout_per_run=2.0,
        )
    finally:
        for thread in setter_threads:
            thread.join(timeout=1.0)
    assert result.property_holds, result.explanation


def test_unmanaged_release_clears_recorded_spin_flag_random() -> None:
    """An unmanaged releaser must clear a waiter's blocking-spin flag.

    A managed waiter that blocks in a cooperative wait under a virtual clock
    flags itself in the scheduler's ``spin_waiters`` so the autojump can tell it
    apart from a runnable thread.  If the resource is released/set by an
    *unmanaged* helper OS thread (no scheduler in TLS), the releaser must reach
    the scheduler the waiter *recorded* at flag time.

    A deterministic end-to-end timing test is infeasible (virtual scheduling
    always races the real-time setter — the autojump-to-own-deadline fires
    before a delayed set() lands); this asserts the recording/resolution
    contract directly, as ``test_give_up_timed_wait_is_atomic_and_unblocks_first``
    does for its OS-descheduling race."""
    from frontrun._cooperative import (
        _note_spin_release,
        _spin_note_hook,
        clear_context,
        get_context,
        set_context,
    )

    scheduler = OpcodeScheduler([], num_threads=1, virtual_clock=VirtualClock(), clock_mode="virtual")
    resource_id = 0xC0FFEE
    tid = 0

    # A managed waiter flags its blocking spin (this must record the scheduler).
    set_context(scheduler, tid)
    try:
        note_spin = _spin_note_hook(scheduler)
        assert note_spin is not None
        note_spin(tid, resource_id, True)
    finally:
        clear_context()
    assert tid in scheduler._spin_waiters

    # Unmanaged releaser: no scheduler in TLS.
    assert get_context() is None
    _note_spin_release(resource_id)

    assert tid not in scheduler._spin_waiters, (
        "an unmanaged release must clear the spin flag via the scheduler the waiter recorded"
    )


def test_unmanaged_release_unblocks_engine_spin_waiter_dpor() -> None:
    """Regression (DPOR): an unmanaged set()/release() must engine-unblock a
    timed cooperative waiter.

    A DPOR timed ``Event.wait(timeout=...)`` waiter blocks itself in the engine
    via ``note_blocking_spin``; only the *untimed* Event path had an
    ``_engine_blocked`` fallback for unmanaged setters, so an unmanaged ``set()``
    on the timed path left the waiter engine-blocked until the deadlock timeout.
    Resolving the scheduler from the waiter's recording fixes it."""
    from frontrun._cooperative import _note_spin_release, _spin_note_hook, clear_context, set_context
    from frontrun._dpor_runtime.scheduler import DporScheduler

    execution = _FakeExecution([0])
    scheduler = DporScheduler(
        _FakeEngine(),
        execution,
        num_threads=1,
        virtual_clock=VirtualClock(),
        clock_mode="virtual",
        clock_actor_id=99,
    )
    resource_id = 0xBEEF
    tid = 0

    set_context(scheduler, tid)
    try:
        note_spin = _spin_note_hook(scheduler)
        assert note_spin is not None
        note_spin(tid, resource_id, True)  # engine-blocks tid + records scheduler
    finally:
        clear_context()
    assert tid in execution.blocked, "note_blocking_spin must engine-block the waiter"

    _note_spin_release(resource_id)  # unmanaged releaser: no scheduler in TLS

    assert tid not in execution.blocked, "an unmanaged release must engine-unblock the recorded spin waiter"


def test_zero_timeout_waits_are_pure_probes() -> None:
    """Event / Condition / Queue with ``timeout == 0`` must be pure probes that
    match threading/queue stdlib semantics exactly: succeed if satisfiable now,
    else immediate ``False`` / ``Empty`` / ``Full``.  (Characterisation test:
    the *result* is unchanged by the refactor that stops these paths from
    registering a zero-length virtual deadline; it passes before and after.)"""

    class State:
        def __init__(self) -> None:
            self.event_unset = threading.Event()
            self.event_set = threading.Event()
            self.event_set.set()
            self.q_empty: queue.Queue[str] = queue.Queue()
            self.q_full: queue.Queue[str] = queue.Queue(maxsize=1)
            self.q_full.put("x")
            self.cond = threading.Condition()
            self.results: dict[str, object] = {}

    def worker(s: State) -> None:
        s.results["event_unset"] = s.event_unset.wait(timeout=0)
        s.results["event_set"] = s.event_set.wait(timeout=0)
        try:
            s.q_empty.get(timeout=0)
            s.results["get_empty"] = "item"
        except queue.Empty:
            s.results["get_empty"] = "empty"
        try:
            s.q_full.put("y", timeout=0)
            s.results["put_full"] = "ok"
        except queue.Full:
            s.results["put_full"] = "full"
        with s.cond:
            s.results["cond"] = s.cond.wait(timeout=0)

    def invariant(s: State) -> bool:
        r = s.results
        assert r.get("event_unset") is False, r
        assert r.get("event_set") is True, r
        assert r.get("get_empty") == "empty", r
        assert r.get("put_full") == "full", r
        assert r.get("cond") is False, r
        return True

    result = frontrun.explore(
        setup=State,
        workers=[worker],
        invariant=invariant,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Timed lock acquires
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


@pytest.mark.parametrize("lock_factory", [threading.Lock, threading.RLock])
def test_timed_acquire_succeeds_before_virtual_deadline(lock_factory: Callable[[], Any]) -> None:
    class State:
        def __init__(self) -> None:
            self.lock = lock_factory()
            self.holder_has_lock = threading.Event()
            self.acquire_result: bool | None = None
            self.waited_virtual = 0.0

    def holder(s: State) -> None:
        with s.lock:
            s.holder_has_lock.set()
            time.sleep(1.0)

    def contender(s: State) -> None:
        s.holder_has_lock.wait()
        start = time.monotonic()
        s.acquire_result = s.lock.acquire(timeout=5.0)
        s.waited_virtual = time.monotonic() - start
        if s.acquire_result:
            s.lock.release()

    result = frontrun.explore(
        setup=State,
        workers=[holder, contender],
        invariant=lambda s: s.acquire_result is True and 1.0 <= s.waited_virtual < 5.0,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


class _TimedAcquireReplayRace:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.holding = threading.Event()
        self.x = 0
        self.contender_ran = False


def _timed_acquire_holder(s: _TimedAcquireReplayRace) -> None:
    s.lock.acquire()
    s.holding.set()
    time.sleep(1.0)
    s.lock.release()
    # Unprotected increment after release — races with the contender's
    # under-lock increment (which only happens if the timed acquire succeeds).
    tmp = s.x
    s.x = tmp + 1


def _timed_acquire_contender(s: _TimedAcquireReplayRace) -> None:
    # Wait until the holder actually holds the lock so the timed acquire is
    # guaranteed contended: it registers a virtual timed wait, the clock
    # advances to the holder's t=1.0 release, and the acquire SUCCEEDS well
    # before its own 5.0s deadline.
    s.holding.wait()
    got = s.lock.acquire(timeout=5.0)
    if got:
        s.contender_ran = True
        tmp = s.x
        s.x = tmp + 1
        s.lock.release()


def test_replay_preserves_successful_timed_acquire() -> None:
    """End-to-end guard: a counterexample whose schedule contains a SUCCESSFUL
    contended timed acquire must reproduce.  The lost-update failure requires
    ``contender_ran`` (i.e. the acquire returned True); if replay ever
    force-expired the timed wait the acquire would take the timeout branch and
    the failure could not reproduce.  See
    ``test_wake_scheduled_sleeper_ignores_timed_waits`` for the deterministic
    unit-level check of the underlying ``_wake_scheduled_sleeper`` behaviour."""
    result = frontrun.explore(
        setup=_TimedAcquireReplayRace,
        workers=[_timed_acquire_holder, _timed_acquire_contender],
        invariant=lambda s: not (s.contender_ran and s.x < 2),
        clock="virtual",
    )
    _assert_invariant_failure(result)
    assert result.counterexample is not None
    assert result.reproduction_attempts == 10
    assert result.reproduction_successes == 10, (
        f"successful timed acquire was force-expired on replay: "
        f"{result.reproduction_successes}/{result.reproduction_attempts}"
    )


class _FakeExecution:
    def __init__(self, runnable: list[int]) -> None:
        self._runnable = list(runnable)
        self.blocked: set[int] = set()
        self.finished: set[int] = set()

    def runnable_threads(self) -> list[int]:
        return [t for t in self._runnable if t not in self.blocked and t not in self.finished]

    def block_thread(self, thread_id: int) -> None:
        self.blocked.add(thread_id)

    def unblock_thread(self, thread_id: int) -> None:
        self.blocked.discard(thread_id)

    def finish_thread(self, thread_id: int) -> None:
        self.finished.add(thread_id)


class _FakeEngine:
    def schedule(self, execution: _FakeExecution) -> int | None:
        runnable = execution.runnable_threads()
        return runnable[0] if runnable else None


def _make_virtual_clock_scheduler() -> Any:
    from frontrun._dpor_runtime.scheduler import DporScheduler

    clock = VirtualClock()
    return DporScheduler(
        _FakeEngine(),
        _FakeExecution([]),
        num_threads=1,
        virtual_clock=clock,
        clock_mode="virtual",
        clock_actor_id=99,
    )


def test_wake_scheduled_sleeper_ignores_timed_waits() -> None:
    """Deterministic regression for the replay force-expiry bug.

    ``_wake_scheduled_sleeper`` is the replay safety net that advances the
    virtual clock when the recorded schedule points at a *sleeping* thread.  A
    thread in a contended ``acquire(timeout=...)`` is registered in
    ``_timed_waits`` for the whole spin, but — unlike a sleeper — it is not
    genuinely stuck (``ReplayExecution.block_thread`` is a no-op) and it may
    have acquired the lock before its deadline in the recorded run.  Advancing
    the clock to that deadline force-expires a wait that never timed out,
    flipping the acquire to the timeout branch and dragging every earlier
    deadline due.  So the safety net must only fire for sleepers."""
    from frontrun._virtual_clock import _TIMED_WAIT_TOKEN

    scheduler = _make_virtual_clock_scheduler()
    clock = scheduler.virtual_clock
    tid = 0
    deadline = clock.now() + 5.0
    scheduler._deadlines.add_timeout(tid, deadline, _TIMED_WAIT_TOKEN)
    scheduler._current_thread = tid

    before = clock.now()
    with scheduler._condition:
        advanced = scheduler._wake_scheduled_sleeper()

    assert advanced is False, "timed waits must not drive the replay clock advance"
    assert clock.now() == before, "the virtual clock must not jump to a timed-wait deadline"
    assert scheduler._deadlines.timed_wait_deadline(tid) == deadline, "the timed wait must be left intact"


def test_before_io_reschedules_when_idle_under_virtual_clock() -> None:
    """``before_io`` gets the same idle-reschedule rescue as its siblings.

    Under a virtual clock the scheduler can be idle (``_current_thread is
    None``) when a worker reaches a scheduling point — e.g. right after a
    timed-wait give-up.  ``_report_and_wait`` and ``before_sync_retry`` rescue
    that state by asking the engine for the next thread instead of stalling;
    a worker issuing a Redis command through ``before_io`` must not instead
    wait out ``deadlock_timeout`` and abort the run with a TimeoutError."""
    from frontrun._dpor_runtime.scheduler import DporScheduler

    scheduler = DporScheduler(
        _FakeEngine(),
        _FakeExecution([0]),
        num_threads=1,
        deadlock_timeout=0.2,
        virtual_clock=VirtualClock(),
        clock_mode="virtual",
        clock_actor_id=99,
    )
    scheduler._current_thread = None  # idle: nobody holds the turn

    scheduler.before_io(0, "redis:key")

    assert scheduler._error is None, f"idle before_io must reschedule, not time out: {scheduler._error!r}"
    assert scheduler._active_io_thread == 0, "the caller must hold the IO turn after the rescue"


def test_give_up_timed_wait_is_atomic_and_unblocks_first() -> None:
    """``give_up_timed_wait`` unblocks before dropping the deadline.

    The waiter must never be engine-blocked with no pending deadline.  It would
    then be indistinguishable from an exact deadlock if every other thread was
    also blocked.

    ``give_up_timed_wait`` keeps the invariant by unblocking and dropping
    the deadline under a single lock, unblock-first, so no scheduler advance can
    ever see the blocked-with-no-deadline state."""
    from frontrun._dpor_runtime.scheduler import DporScheduler
    from frontrun._virtual_clock import _TIMED_WAIT_TOKEN

    unblock_observations: list[bool] = []

    class _RecordingExecution(_FakeExecution):
        def unblock_thread(self, thread_id: int) -> None:
            # Record whether the deadline is still registered at unblock time.
            if thread_id == 0:
                unblock_observations.append(scheduler._deadlines.in_timed_wait(thread_id))
            super().unblock_thread(thread_id)

    clock = VirtualClock()
    execution = _RecordingExecution([0])
    scheduler = DporScheduler(
        _FakeEngine(),
        execution,
        num_threads=1,
        virtual_clock=clock,
        clock_mode="virtual",
        clock_actor_id=99,
    )
    tid = 0
    deadline = clock.now() + 5.0
    scheduler._deadlines.add_timeout(tid, deadline, _TIMED_WAIT_TOKEN)
    execution.block_thread(tid)

    scheduler.give_up_timed_wait(tid)

    assert tid not in execution.blocked, "give_up_timed_wait must unblock the waiter"
    assert not scheduler._deadlines.in_timed_wait(tid), "give_up_timed_wait must drop the deadline"
    assert not scheduler._deadlines.has_pending(), "the deadline must be cancelled"
    assert unblock_observations == [True], (
        "the waiter must be unblocked while its deadline is still registered "
        "(unblock-first closes the exact-deadlock false-positive window)"
    )


def test_baseline_threads_tracked_weakly_so_dead_ids_drop() -> None:
    """Baseline threads must be tracked by weak reference, not by raw id.

    ``_has_live_external_threads`` subtracts the baseline threads captured at
    construction to decide whether some non-worker thread could still unblock a
    waiter (which suppresses exact-deadlock detection).  Keying by ``id(Thread)``
    is unsound: if a baseline thread exits and is GC'd, a new external thread can
    reuse its id and be wrongly subtracted, re-enabling exact-deadlock while that
    thread could still make progress.  Storing weak references makes dead
    baseline threads drop out so their ids cannot mask a reused id."""
    import gc
    import weakref

    started = threading.Event()
    release = threading.Event()

    def _run() -> None:
        started.set()
        release.wait()

    t = threading.Thread(target=_run)
    t.start()
    try:
        assert started.wait(timeout=5.0)
        scheduler = _make_virtual_clock_scheduler()  # captures baseline incl. t
        assert any(bt is t for bt in scheduler._baseline_thread_keys), (
            "baseline must store live Thread objects (weakly), not raw ids"
        )
        tid = id(t)
        wr = weakref.ref(t)
    finally:
        release.set()
        t.join(timeout=5.0)

    del t
    gc.collect()

    assert wr() is None, "a finished baseline thread must not be strongly pinned"
    baseline_ids = {id(bt) for bt in scheduler._baseline_thread_keys}
    assert tid not in baseline_ids, (
        "a dead+GC'd baseline thread's id must drop out of the tracking set so a "
        "reused id on a new external thread cannot be misclassified as baseline"
    )


def test_timed_semaphore_acquire_times_out_without_false_deadlock() -> None:
    class State:
        def __init__(self) -> None:
            self.sem = threading.Semaphore(0)
            self.acquire_result: bool | None = None
            self.waited_virtual = 0.0

    def worker(s: State) -> None:
        start = time.monotonic()
        s.acquire_result = s.sem.acquire(timeout=1.0)
        s.waited_virtual = time.monotonic() - start

    result = frontrun.explore(
        setup=State,
        workers=[worker],
        invariant=lambda s: s.acquire_result is False and s.waited_virtual >= 1.0,
        clock="virtual",
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation


class _TimedWaitState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.cond = threading.Condition()
        self.q: queue.Queue[str] = queue.Queue()
        self.full_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.full_q.put_nowait("old")
        self.timed_out = False
        self.waited_virtual = 0.0


def _assert_timed_wait_expires(
    waiter: Callable[[_TimedWaitState], None],
    releaser: Callable[[_TimedWaitState], None],
) -> None:
    result = frontrun.explore(
        setup=_TimedWaitState,
        workers=[waiter, releaser],
        invariant=lambda s: s.timed_out and s.waited_virtual >= 0.05,
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=0.2,
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation


def test_timed_event_wait_expires_on_virtual_deadline() -> None:
    def waiter(s: _TimedWaitState) -> None:
        start = time.monotonic()
        s.timed_out = not s.event.wait(timeout=0.05)
        s.waited_virtual = time.monotonic() - start

    def setter(s: _TimedWaitState) -> None:
        time.sleep(1.0)
        s.event.set()

    _assert_timed_wait_expires(waiter, setter)


def test_timed_condition_wait_expires_on_virtual_deadline() -> None:
    def waiter(s: _TimedWaitState) -> None:
        with s.cond:
            start = time.monotonic()
            s.timed_out = not s.cond.wait(timeout=0.05)
            s.waited_virtual = time.monotonic() - start

    def notifier(s: _TimedWaitState) -> None:
        time.sleep(1.0)
        with s.cond:
            s.cond.notify()

    _assert_timed_wait_expires(waiter, notifier)


def test_condition_wait_for_timeout_uses_virtual_deadline() -> None:
    class State:
        def __init__(self) -> None:
            self.cond = threading.Condition()
            self.wait_result = True
            self.waited_virtual = 0.0

    def waiter(s: State) -> None:
        with s.cond:
            start = time.monotonic()
            s.wait_result = s.cond.wait_for(lambda: False, timeout=5.0)
            s.waited_virtual = time.monotonic() - start

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=State,
        workers=[waiter],
        invariant=lambda s: not s.wait_result and s.waited_virtual >= 5.0,
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=0.05,
        timeout_per_run=1.0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert wall_elapsed < 1.0, f"wait_for burned wall time instead of virtual time ({wall_elapsed:.3f}s)"


def test_timed_queue_get_expires_on_virtual_deadline() -> None:
    def consumer(s: _TimedWaitState) -> None:
        start = time.monotonic()
        try:
            s.q.get(timeout=0.05)
        except queue.Empty:
            s.timed_out = True
        s.waited_virtual = time.monotonic() - start

    def producer(s: _TimedWaitState) -> None:
        time.sleep(1.0)
        s.q.put("x")

    _assert_timed_wait_expires(consumer, producer)


def test_timed_queue_put_expires_on_virtual_deadline() -> None:
    def producer(s: _TimedWaitState) -> None:
        start = time.monotonic()
        try:
            s.full_q.put("new", timeout=0.05)
        except queue.Full:
            s.timed_out = True
        s.waited_virtual = time.monotonic() - start

    def consumer(s: _TimedWaitState) -> None:
        time.sleep(1.0)
        s.full_q.get()

    _assert_timed_wait_expires(producer, consumer)


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


def test_random_strategy_virtual_sleep_zero_wall_time() -> None:
    def worker(s: _SleepObserver) -> None:
        s.start = time.monotonic()
        time.sleep(300.0)
        s.end = time.monotonic()

    invariant_checks = 0

    def invariant(s: _SleepObserver) -> bool:
        nonlocal invariant_checks
        invariant_checks += 1
        return s.end - s.start >= 300.0

    wall_start = time.monotonic()
    result = frontrun.explore(
        setup=_SleepObserver,
        workers=[worker, lambda s: None],
        invariant=invariant,
        strategy="random",
        clock="virtual",
        max_attempts=5,
        reproduce_on_failure=0,
        timeout_per_run=1.0,
    )
    wall_elapsed = time.monotonic() - wall_start
    assert result.property_holds, result.explanation
    assert invariant_checks > 0
    assert wall_elapsed < 60.0


def test_random_queue_waiter_does_not_starve_virtual_sleep_autojump() -> None:
    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()
            self.got: str | None = None
            self.slept_virtual = 0.0

    def producer(s: State) -> None:
        start = time.monotonic()
        time.sleep(1.0)
        s.slept_virtual = time.monotonic() - start
        s.q.put("x")

    def consumer(s: State) -> None:
        s.got = s.q.get()

    # timeout is only the hang backstop (a starved autojump never completes at
    # any budget); keep it generous — 1.0s flaked on loaded free-threaded CI.
    state = run_with_schedule(
        [0, 1] * 20,
        setup=State,
        threads=[consumer, producer],
        clock="virtual",
        timeout=10.0,
        deadlock_timeout=0.05,
    )
    assert state.got == "x"
    assert state.slept_virtual >= 1.0


def test_dpor_queue_waiter_does_not_starve_virtual_sleep_autojump() -> None:
    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()
            self.got: str | None = None
            self.slept_virtual = 0.0

    def producer(s: State) -> None:
        start = time.monotonic()
        time.sleep(1.0)
        s.slept_virtual = time.monotonic() - start
        s.q.put("x")

    def consumer(s: State) -> None:
        s.got = s.q.get()

    result = frontrun.explore(
        setup=State,
        workers=[consumer, producer],
        invariant=lambda s: s.got == "x" and s.slept_virtual >= 1.0,
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=0.2,
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation


def test_dpor_queue_put_nowait_wakes_blocked_getter() -> None:
    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue()
            self.got: str | None = None

    def consumer(s: State) -> None:
        s.got = s.q.get()

    def producer(s: State) -> None:
        s.q.put_nowait("x")

    result = frontrun.explore(
        setup=State,
        workers=[consumer, producer],
        invariant=lambda s: s.got == "x",
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=0.2,
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation


def test_dpor_queue_get_nowait_wakes_blocked_putter() -> None:
    class State:
        def __init__(self) -> None:
            self.q: queue.Queue[str] = queue.Queue(maxsize=1)
            self.q.put_nowait("old")
            self.got: str | None = None
            self.put_completed = False

    def producer(s: State) -> None:
        s.q.put("new")
        s.put_completed = True

    def consumer(s: State) -> None:
        s.got = s.q.get_nowait()

    result = frontrun.explore(
        setup=State,
        workers=[producer, consumer],
        invariant=lambda s: s.got == "old" and s.put_completed,
        clock="virtual",
        reproduce_on_failure=0,
        deadlock_timeout=0.2,
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation


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

    def invariant(s: _RetryRace) -> bool:
        assert s.x == 100, f"expected delayed writer to remain final, got x={s.x}"
        return True

    result = frontrun.explore(
        setup=_RetryRace,
        workers=[_rmw_worker, _delayed_writer],
        invariant=invariant,
        strategy="random",
        clock="explored",
        max_attempts=200,
        seed=1234,
        reproduce_on_failure=0,
    )
    _assert_invariant_failure(result, "expected delayed writer")


def test_stale_timed_wait_spin_flag_is_refused() -> None:
    """A timed-wait spin flag must not land after its deadline already fired.

    TOCTOU race (reliable on 3.14t, where threads run truly concurrently;
    possible on GIL builds): a timed Event/Condition/Queue waiter passes its
    expiry check, then blocks on the scheduler condition inside
    ``note_blocking_spin`` while another thread's autojump advances the clock
    past its deadline (popping deadline and flag).  The queued flag then lands
    with no deadline behind it, the waiter counts as clock-blocked, and the
    next autojump advances past the waiter's own deadline before it re-probes
    — ``event.wait(timeout=10)`` observes 20 elapsed virtual seconds.

    The port must refuse a ``timed_wait`` flag when the actor no longer has a
    pending timeout deadline (flag and deadline share the scheduler condition,
    so the check is race-free).
    """
    clock = VirtualClock()
    scheduler = OpcodeScheduler([], num_threads=2, virtual_clock=clock, clock_mode="virtual")
    resource_id = 0xC0FFEE

    # Actor 1 registers a timed wait, then its deadline fires (another
    # thread's autojump pops deadline + flag)...
    scheduler.add_timed_wait(1, clock.now() + 10.0)
    scheduler._advance_clock_to(clock.now() + 10.0)
    # ...and only then does the queued flag land: it must be refused.
    scheduler.note_blocking_spin(1, resource_id, True, timed_wait=True)
    assert 1 not in scheduler._spin_waiters

    # Control: with a pending deadline the flag lands normally.
    scheduler.add_timed_wait(1, clock.now() + 5.0)
    scheduler.note_blocking_spin(1, resource_id, True, timed_wait=True)
    assert 1 in scheduler._spin_waiters


def test_invariant_sleep_is_virtual_not_wall_clock() -> None:
    # Regression: the sync driver held clock_scope (time.* READS -> virtual)
    # across setup/run/invariant, but runner.patch_scope (time.sleep patch)
    # ended before invariant evaluation. A TTL-style invariant that sleeps to
    # age past an expiry then re-checks therefore blocked for REAL wall time
    # while its time reads stayed frozen at virtual time — self-inconsistent
    # (elapsed == 0.0 after sleep(5)) and costing real seconds per explored
    # interleaving in a test the user declared "virtual". setup() already ran
    # inside the patch scope; the invariant must see the identical clock.
    class State:
        elapsed: float | None = None

    def worker(s: State) -> None:
        time.sleep(1.0)

    def invariant(s: State) -> bool:
        t0 = time.monotonic()
        time.sleep(5.0)
        s.elapsed = time.monotonic() - t0
        return s.elapsed >= 5.0

    start = time.monotonic()
    result = frontrun.explore(
        setup=State,
        workers=[worker, worker],
        invariant=invariant,
        clock="virtual",
        reproduce_on_failure=0,
    )
    wall = time.monotonic() - start
    assert result.property_holds, f"invariant saw frozen/real-time-inconsistent clock: {result.explanation}"
    # The invariant's sleep must be virtual: multiple interleavings each
    # sleeping a real 5s would take tens of seconds.
    assert wall < 4.0, f"invariant sleep ran on the wall clock ({wall:.1f}s elapsed)"


@pytest.mark.xfail(
    strict=True,
    reason="known gap (round-2 review): a timeout-kind deadline firing carries no engine-visible "
    "event (scheduler.py _on_clock_wake 'timeout' branch), so it commutes with every worker step "
    "and DPOR never seeds the 'timeout beats the zero-virtual-time holder's release' branch — "
    "unlike sleep wakes, which report release/acquire edges. Tracked in "
    "ideas/possible-future-roadmap/virtual-clock-hardening-deferred.md #10.",
)
def test_explored_clock_finds_timed_acquire_timeout_against_runnable_holder() -> None:
    # Desired: with clock="explored", "the timeout fired before the holder
    # released" is a legitimate interleaving (distinct final state — the timed
    # acquire returns False), so DPOR must explore it and report the invariant
    # violation. Today only the acquired=True branch is ever explored.
    class State:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.holder_has_lock = threading.Event()
            self.acquire_result: bool | None = None

    def holder(s: State) -> None:
        s.lock.acquire()
        s.holder_has_lock.set()
        s.lock.release()  # runnable in zero virtual time: never sleeps

    def contender(s: State) -> None:
        s.holder_has_lock.wait()
        s.acquire_result = s.lock.acquire(timeout=1.0)
        if s.acquire_result:
            s.lock.release()

    result = frontrun.explore(
        setup=State,
        workers=[holder, contender],
        invariant=lambda s: s.acquire_result is True,
        clock="explored",
        reproduce_on_failure=0,
    )
    assert not result.property_holds, (
        f"the timeout branch (acquire_result=False) was never explored: {result.num_explored} interleavings"
    )


def test_helper_thread_spawned_in_earlier_execution_does_not_arm_false_deadlock() -> None:
    """A worker-spawned service thread must keep waking waiters in later executions.

    Two coupled defects broke executions after the first:

    1. The baseline-thread snapshot used by external-liveness reasoning was
       taken per-execution (at scheduler construction), so a helper thread
       spawned during execution 1 was classified as inert 'baseline' for
       execution 2 even though it services explored waiters.  The snapshot
       must cover the whole exploration.
    2. When the helper's unmanaged ``event.set()`` landed while the turn was
       vacant (every managed thread blocked, ``_schedule_next`` had returned
       None), ``clear_engine_block`` only notified — nobody re-asked the
       engine until a wait-timeout arm fired, costing a full deadlock_timeout
       per wake and failing the run when the join budget is shorter.
    """
    from frontrun._cooperative import _real_time_sleep

    requests: list[threading.Event] = []  # GIL-atomic append/pop
    helper_state = {"started": False, "stop": False}
    helper_box: list[threading.Thread] = []

    def _serve() -> None:
        while not helper_state["stop"]:
            evt = requests.pop() if requests else None
            if evt is None:
                _real_time_sleep(0.005)
                continue
            _real_time_sleep(0.3)  # longer than the exact-deadlock confirm window
            evt.set()

    class _State:
        def __init__(self) -> None:
            self.counter = 0
            self.ok: list[bool] = [False, False]

    def _worker(index: int):
        def w(s: _State) -> None:
            # Shared-counter race so DPOR explores more than one execution.
            value = s.counter
            s.counter = value + 1
            if not helper_state["started"]:
                helper_state["started"] = True
                helper = threading.Thread(target=_serve, daemon=True)
                helper_box.append(helper)
                helper.start()
            evt = threading.Event()
            requests.append(evt)
            s.ok[index] = evt.wait()

        return w

    try:
        result = frontrun.explore(
            setup=_State,
            workers=[_worker(0), _worker(1)],
            invariant=lambda s: all(s.ok),
            clock="virtual",
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation
    finally:
        helper_state["stop"] = True
        for helper in helper_box:
            helper.join(timeout=5.0)


def test_replay_sleep_until_phase2_bypasses_positional_gate_when_anchor_waiters_pend() -> None:
    """A woken replay sleeper must not stall while access-gate waiters pend.

    ``_wait_for_turn`` suspends positional gating for every thread while any
    thread gate-waits on an access anchor (the anchor owner must be able to
    reach its recorded access).  ``sleep_until`` phase 1 has the matching
    escape (``_replay_sleep_self_wake``), but phase 2 previously insisted on
    ``_current_thread == thread_id`` — which nothing can satisfy while gates
    hold the walk — burning a full deadlock_timeout per reproduction attempt
    (or failing the attempt when the gate's own timeout loses the race).
    """
    from frontrun._dpor_runtime.scheduler import _ReplayDporScheduler
    from frontrun._virtual_clock import real_monotonic

    sched = _ReplayDporScheduler(
        schedule=[1] * 8,
        num_threads=2,
        deadlock_timeout=3.0,
        access_schedule=[(0, "Thing.attr", "write")],
        virtual_clock=VirtualClock(),
        clock_mode="virtual",
        clock_actor_id=2,
    )
    gate_done = threading.Event()

    def _gate_waiter() -> None:
        # Thread 1 waits for an anchor owned by thread 0 (arms _gate_waiters).
        sched._gate_access((1, "Thing.attr", "read"))
        gate_done.set()

    gate_thread = threading.Thread(target=_gate_waiter, daemon=True)
    gate_thread.start()
    deadline = real_monotonic() + 3.0
    while sched._gate_waiters == 0 and real_monotonic() < deadline:
        time.sleep(0.005)
    assert sched._gate_waiters == 1

    try:
        # The positional walk points elsewhere; only the gate bypass can move us.
        with sched._condition:
            sched._current_thread = 1
        start = real_monotonic()
        sched.sleep_until(0, VIRTUAL_EPOCH + 5.0)
        elapsed = real_monotonic() - start
        assert elapsed < 1.5, f"sleep_until stalled {elapsed:.2f}s behind the positional gate"
        assert sched._error is None, sched._error
    finally:
        # Unstick the gate waiter and reap its thread.
        with sched._condition:
            sched._finished = True
            sched._condition.notify_all()
        gate_done.wait(timeout=5.0)
        gate_thread.join(timeout=5.0)


def test_timed_wait_deadline_computed_under_scheduler_lock_survives_concurrent_advance() -> None:
    """A timed wait must never register an already-expired deadline.

    ``_timed_acquire_state`` used to compute ``clock.now() + timeout`` in the
    caller and pass the absolute deadline to ``add_timed_wait``.  Under
    ``clock="explored"`` another thread's clock-actor step can land between
    that read and the registration; the wait then observed (up to) the whole
    advance as elapsed time and expired on its first probe — a timed acquire
    seeing far more virtual time than its timeout, nondeterministically.  The
    deadline is now computed inside the scheduler's serialising lock, which
    every clock advance also holds.
    """
    from frontrun import _cooperative

    clock = VirtualClock()
    scheduler = OpcodeScheduler(
        [0],
        num_threads=1,
        deadlock_timeout=0.2,
        virtual_clock=clock,
        clock_mode="explored",
    )
    original_add = scheduler.add_timed_wait

    def racing_add(thread_id: int, deadline: float | None = None, *, timeout: float | None = None) -> float:
        # Simulate the concurrent explored-mode advance landing just before
        # the registration is serialised.
        clock.advance_to(clock.now() + 100.0)
        if deadline is not None:
            return original_add(thread_id, deadline)
        return original_add(thread_id, timeout=timeout)

    scheduler.add_timed_wait = racing_add  # type: ignore[method-assign]
    deadline, graph, got_clock = _cooperative._timed_acquire_state(5.0, scheduler, 0)
    assert got_clock is clock
    assert graph is None
    assert deadline == pytest.approx(clock.now() + 5.0)
    assert not _cooperative._timed_acquire_expired(deadline, clock)
