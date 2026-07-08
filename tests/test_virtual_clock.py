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
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import pytest

import frontrun
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
    (regression: sleep(1.0) silently returned with 0 virtual seconds)."""
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
        timeout_per_run=1.0,
    )
    assert result.property_holds, result.explanation
    assert invariant_checks > 0


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
    from frontrun._dpor_runtime.scheduler import _TIMED_WAIT_TOKEN

    scheduler = _make_virtual_clock_scheduler()
    clock = scheduler.virtual_clock
    tid = 0
    deadline = clock.now() + 5.0
    scheduler._timed_waits[tid] = deadline
    scheduler._deadlines.add_timeout(tid, deadline, _TIMED_WAIT_TOKEN)
    scheduler._current_thread = tid

    before = clock.now()
    with scheduler._condition:
        advanced = scheduler._wake_scheduled_sleeper()

    assert advanced is False, "timed waits must not drive the replay clock advance"
    assert clock.now() == before, "the virtual clock must not jump to a timed-wait deadline"
    assert scheduler._timed_waits.get(tid) == deadline, "the timed wait must be left intact"


def test_give_up_timed_wait_is_atomic_and_unblocks_first() -> None:
    """Regression guard for the give-up exact-deadlock false-positive window.

    The old give-up path removed the timed-wait deadline BEFORE clearing the
    engine block, in two separate lock acquisitions.  Between them the waiter
    was engine-blocked with no pending deadline; if every other thread was also
    blocked, ``_schedule_next`` could observe "no runnable thread and no
    deadline" and (after the confirm window) raise a spurious DeadlockError.

    ``give_up_timed_wait`` closes the window: it unblocks the waiter and drops
    the deadline under a single lock, unblock-first, so no scheduler advance can
    ever see the blocked-with-no-deadline state.  A deterministic test of the
    OS-descheduling race itself is infeasible; this asserts the ordering and
    atomicity contract instead."""
    from frontrun._dpor_runtime.scheduler import _TIMED_WAIT_TOKEN, DporScheduler

    unblock_observations: list[bool] = []

    class _RecordingExecution(_FakeExecution):
        def unblock_thread(self, thread_id: int) -> None:
            # Record whether the deadline is still registered at unblock time.
            if thread_id == 0:
                unblock_observations.append(thread_id in scheduler._timed_waits)
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
    scheduler._timed_waits[tid] = deadline
    scheduler._deadlines.add_timeout(tid, deadline, _TIMED_WAIT_TOKEN)
    execution.block_thread(tid)

    scheduler.give_up_timed_wait(tid)

    assert tid not in execution.blocked, "give_up_timed_wait must unblock the waiter"
    assert tid not in scheduler._timed_waits, "give_up_timed_wait must drop the deadline"
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

    state = run_with_schedule(
        [0, 1] * 20,
        setup=State,
        threads=[consumer, producer],
        clock="virtual",
        timeout=1.0,
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
