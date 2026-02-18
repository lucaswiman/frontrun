"""
Exhaustive concurrency bug exploration for pybreaker CircuitBreaker.

This file tests MANY distinct concurrency bugs in pybreaker beyond the single
increment_counter() lost-update test in test_pybreaker_real.py.

Each test targets a different race condition or TOCTOU bug identified through
manual source-code review of pybreaker's CircuitMemoryStorage and
CircuitBreaker state machine.

Key insight: pybreaker's CircuitBreaker.call() acquires self._lock (an RLock)
which serializes threads using a single breaker instance. However, bugs
manifest when:
  (a) Two CircuitBreaker instances share the same storage (common with Redis,
      also possible with memory storage -- pybreaker documents this pattern).
  (b) State/storage methods are called directly (e.g. on_success, on_failure,
      _handle_error, _handle_success) outside the lock.
  (c) Storage operations (increment_counter, reset_counter, etc.) are invoked
      concurrently from any path.

Tests below use approach (a) or (b) depending on which best exposes the bug.

Repository: https://github.com/danielfm/pybreaker
"""

import os
import sys
from datetime import datetime

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_test_dir, "..", "external_repos", "pybreaker", "src")
# Insert local repo path FIRST so interlace can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_root))

from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from pybreaker import (  # noqa: E402
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
)

from interlace.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: create a CircuitBreaker attached to an existing storage, forcing
# it into a specific state. This models the documented pattern of multiple
# breaker instances sharing one storage backend.
# ---------------------------------------------------------------------------

def _make_breaker(storage, fail_max=5, reset_timeout=60, success_threshold=1, state=None):
    """Create a CircuitBreaker sharing the given storage."""
    breaker = CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        success_threshold=success_threshold,
        state_storage=storage,
    )
    if state is not None:
        breaker._state = breaker._create_new_state(state)
    return breaker


# ===========================================================================
# 1. Lost update on _success_counter (mirror of fail_counter bug)
# ===========================================================================
# CircuitMemoryStorage.increment_success_counter() does:
#     self._success_counter += 1
# This is the same non-atomic read-modify-write as increment_counter().
# Two threads calling increment_success_counter() concurrently can both read
# the same value, both add one, and one write overwrites the other.


class SuccessCounterLostUpdateState:
    """Two threads each call increment_success_counter() once."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)

    def thread1(self):
        self.storage.increment_success_counter()

    def thread2(self):
        self.storage.increment_success_counter()


def _success_counter_invariant(s: SuccessCounterLostUpdateState) -> bool:
    return s.storage.success_counter == 2


def test_success_counter_lost_update():
    """Find the increment_success_counter() lost update in CircuitMemoryStorage."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SuccessCounterLostUpdateState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_success_counter_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 2. Double closed-to-open transition (two breakers, shared storage)
# ===========================================================================
# CircuitClosedState.on_failure() checks:
#     if self._breaker._state_storage.counter >= self._breaker.fail_max:
#         self._breaker.open()
# Two breakers sharing storage: both see counter >= fail_max, BOTH call open().
# This results in open() being called twice, with doubled side effects.


class DoubleOpenTransitionState:
    """Two breakers share storage; each triggers a failure via call()."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        self.breaker1 = _make_breaker(self.storage, fail_max=1, state=STATE_CLOSED)
        self.breaker2 = _make_breaker(self.storage, fail_max=1, state=STATE_CLOSED)
        self.open_count = 0

        # Patch open() on both breakers to count invocations
        self._orig_open1 = self.breaker1.open
        self._orig_open2 = self.breaker2.open
        self.breaker1.open = self._counting_open1
        self.breaker2.open = self._counting_open2

    def _counting_open1(self):
        self.open_count += 1
        return self._orig_open1()

    def _counting_open2(self):
        self.open_count += 1
        return self._orig_open2()

    def thread1(self):
        try:
            self.breaker1.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    def thread2(self):
        try:
            self.breaker2.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    @staticmethod
    def _fail():
        raise RuntimeError("simulated failure")


def _double_open_invariant(s: DoubleOpenTransitionState) -> bool:
    # open() should be called at most once for a single closed->open transition.
    return s.open_count <= 1


def test_double_closed_to_open_transition():
    """Detect race where two breakers both trigger closed->open on shared storage."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoubleOpenTransitionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_double_open_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 3. Half-open success counter race => double close (direct on_success)
# ===========================================================================
# CircuitHalfOpenState.on_success() does:
#     self._breaker._state_storage.increment_success_counter()
#     if self._breaker._state_storage.success_counter >= self._breaker.success_threshold:
#         self._breaker.close()
#
# With success_threshold=2, two concurrent on_success() calls can each
# increment the success counter (lost update makes it 1 instead of 2),
# or both see >= 2 and both call close() (TOCTOU).


class HalfOpenDoubleCloseState:
    """Two threads call on_success() concurrently in half-open with threshold=2."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_HALF_OPEN)
        self.breaker = _make_breaker(
            self.storage, fail_max=1, reset_timeout=0,
            success_threshold=2, state=STATE_HALF_OPEN,
        )
        self.close_count = 0
        self._original_close = self.breaker.close
        self.breaker.close = self._counting_close

    def _counting_close(self):
        self.close_count += 1
        return self._original_close()

    def thread1(self):
        self.breaker._state.on_success()

    def thread2(self):
        self.breaker._state.on_success()


def _half_open_double_close_invariant(s: HalfOpenDoubleCloseState) -> bool:
    # After two on_success() calls with threshold=2, close() should be called
    # exactly once. If the success counter lost update causes it to stay at 1,
    # close() is called 0 times. If TOCTOU, it may be called 2 times.
    return s.close_count == 1


def test_half_open_double_close():
    """Detect race where concurrent on_success() double-closes or misses close."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: HalfOpenDoubleCloseState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_half_open_double_close_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 4. Counter reset vs increment race
# ===========================================================================
# _handle_success() calls storage.reset_counter() (sets _fail_counter = 0).
# _handle_error() calls _inc_counter() -> storage.increment_counter()
# (does _fail_counter += 1).
#
# Starting from _fail_counter = 5:
#   Thread B reads _fail_counter = 5
#   Thread A resets _fail_counter = 0
#   Thread B writes _fail_counter = 5 + 1 = 6
# Result 6 is illegal: should be 0 or 1.


class ResetVsIncrementState:
    """One thread resets the fail counter, one thread increments it.
    We start with counter at 5 to make the stale-read race visible.
    """

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        # Prime the counter to 5
        for _ in range(5):
            self.storage.increment_counter()

    def thread1(self):
        """Reset counter to 0."""
        self.storage.reset_counter()

    def thread2(self):
        """Increment counter by 1."""
        self.storage.increment_counter()


def _reset_vs_increment_invariant(s: ResetVsIncrementState) -> bool:
    # Result should be 0 (reset after increment) or 1 (increment after reset),
    # never 6 (increment read stale 5, reset ran, increment writes 6).
    return s.storage.counter in (0, 1)


def test_reset_vs_increment_race():
    """Detect race between reset_counter() and increment_counter()."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ResetVsIncrementState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_reset_vs_increment_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 5. Closed state: two breakers fail, lost update on fail counter
# ===========================================================================
# Two CircuitBreaker instances share storage. Each calls breaker.call(fail).
# Each breaker holds its own lock, so the storage operations interleave.
# With fail_max=2, both failures should push counter to 2, but lost update
# can leave it at 1.


class ClosedStateDoubleFailState:
    """Two breakers share storage; both fail concurrently with fail_max=2."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        self.breaker1 = _make_breaker(self.storage, fail_max=2, state=STATE_CLOSED)
        self.breaker2 = _make_breaker(self.storage, fail_max=2, state=STATE_CLOSED)

    def thread1(self):
        try:
            self.breaker1.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    def thread2(self):
        try:
            self.breaker2.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    @staticmethod
    def _fail():
        raise RuntimeError("simulated failure")


def _closed_double_fail_invariant(s: ClosedStateDoubleFailState) -> bool:
    # After two failures with fail_max=2, counter should be 2.
    return s.storage.counter == 2


def test_closed_state_double_fail_lost_update():
    """Detect lost fail_counter update when two breakers fail on shared storage."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ClosedStateDoubleFailState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_closed_double_fail_invariant,
            max_attempts=500,
            max_ops=400,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 6. Concurrent success and failure in half-open state (direct calls)
# ===========================================================================
# In half-open, on_success() increments success counter and may close.
# on_failure() calls open(). If both execute concurrently, the circuit
# may end up in an inconsistent state.
# We call _handle_success() and _handle_error() directly on the state.


class HalfOpenSuccessFailRaceState:
    """One thread triggers success, one triggers failure, in half-open."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_HALF_OPEN)
        self.breaker = _make_breaker(
            self.storage, fail_max=1, reset_timeout=0,
            success_threshold=1, state=STATE_HALF_OPEN,
        )

    def thread1(self):
        """Success path: calls _handle_success on the half-open state."""
        self.breaker._state._handle_success()

    def thread2(self):
        """Failure path: calls _handle_error on the half-open state."""
        try:
            exc = RuntimeError("simulated failure")
            self.breaker._state._handle_error(exc)
        except (CircuitBreakerError, RuntimeError):
            pass


def _half_open_success_fail_invariant(s: HalfOpenSuccessFailRaceState) -> bool:
    # After one success and one failure in half-open, the circuit should be
    # in a definite state (open or closed) with consistent counters:
    # - If closed (success won): fail_counter should be 0 (reset by _handle_success).
    #   Bug: _handle_error increments counter, then _handle_success resets it,
    #   but the increment can race and write AFTER the reset, leaving counter > 0.
    # - If open (failure won): fail_counter should be >= 1 (incremented by _handle_error).
    # - Should not remain half-open.
    state = s.storage.state
    if state == STATE_CLOSED:
        return s.storage.counter == 0
    elif state == STATE_OPEN:
        return s.storage.counter >= 1
    else:
        return False


def test_half_open_success_fail_race():
    """Detect race between concurrent success and failure in half-open state."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: HalfOpenSuccessFailRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_half_open_success_fail_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 7. TOCTOU in CircuitBreaker.state property (getter)
# ===========================================================================
# The state property getter does:
#     if self.current_state != self._state.name:
#         self.state = self.current_state   # setter, acquires lock
#     return self._state
#
# This check-then-act is NOT locked. Between the comparison and the
# assignment, another thread can change the storage state, causing
# the cached _state to be stale after the property returns.


class StateTOCTOUState:
    """One thread mutates storage state, another reads .state property."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        self.breaker = CircuitBreaker(
            fail_max=5,
            reset_timeout=0,
            state_storage=self.storage,
        )
        self.observed_state_name = None
        self.observed_storage_state = None

    def thread1(self):
        """Mutate the underlying storage state to open."""
        self.storage._state = STATE_OPEN

    def thread2(self):
        """Read the .state property and capture what we see."""
        state_obj = self.breaker.state
        self.observed_state_name = state_obj.name
        self.observed_storage_state = self.storage.state


def _state_toctou_invariant(s: StateTOCTOUState) -> bool:
    # The cached state object returned by .state should match the current
    # storage state. If TOCTOU, they diverge.
    if s.observed_state_name is None:
        return True
    return s.observed_state_name == s.observed_storage_state


def test_state_property_toctou():
    """Detect TOCTOU race in CircuitBreaker.state property getter."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: StateTOCTOUState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_state_toctou_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 8. Concurrent _handle_error: both increment and both check threshold
# ===========================================================================
# Two threads call _handle_error on a closed-state breaker (bypassing the
# lock). Each does: increment_counter() then checks counter >= fail_max.
# With fail_max=2 and starting counter=0:
#   Thread A: reads counter=0, writes counter=1
#   Thread B: reads counter=0, writes counter=1  (LOST UPDATE)
#   Thread A: checks counter (1) >= 2? No.
#   Thread B: checks counter (1) >= 2? No.
# Neither thread opens the circuit even though two failures occurred.
# Alternatively: counter correctly reaches 2, but both threads see 2 and
# both call open() (double transition).
# We check that after two _handle_error calls from closed with fail_max=2,
# the counter should be 2.


class ClosedHandleErrorRaceState:
    """Two threads call _handle_error directly on closed state, bypass lock."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        self.breaker = _make_breaker(
            self.storage, fail_max=2, reset_timeout=60, state=STATE_CLOSED,
        )

    def thread1(self):
        try:
            exc = RuntimeError("simulated failure 1")
            self.breaker._state._handle_error(exc)
        except (CircuitBreakerError, RuntimeError):
            pass

    def thread2(self):
        try:
            exc = RuntimeError("simulated failure 2")
            self.breaker._state._handle_error(exc)
        except (CircuitBreakerError, RuntimeError):
            pass


def _closed_handle_error_invariant(s: ClosedHandleErrorRaceState) -> bool:
    # After two _handle_error() calls with fail_max=2, counter should be 2.
    return s.storage.counter == 2


def test_closed_handle_error_race():
    """Detect lost update when two threads call _handle_error on closed state."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ClosedHandleErrorRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_closed_handle_error_invariant,
            max_attempts=500,
            max_ops=400,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 9. Three threads: fail counter lost update with three increments
# ===========================================================================
# Three concurrent increment_counter() calls. Counter can lose 1 or 2
# increments due to the non-atomic read-modify-write.


class ThreeThreadFailCounterState:
    """Three threads each call increment_counter() once."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)

    def thread1(self):
        self.storage.increment_counter()

    def thread2(self):
        self.storage.increment_counter()

    def thread3(self):
        self.storage.increment_counter()


def _three_thread_fail_counter_invariant(s: ThreeThreadFailCounterState) -> bool:
    return s.storage.counter == 3


def test_three_thread_fail_counter_lost_update():
    """Find lost update with three concurrent increment_counter() calls."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadFailCounterState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_three_thread_fail_counter_invariant,
            max_attempts=500,
            max_ops=300,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 10. Circuit stays closed when it should open (shared storage, fail_max=2)
# ===========================================================================
# Two breakers share storage with fail_max=2. Each triggers one failure.
# Due to the lost update, the counter may only reach 1 instead of 2,
# so neither breaker opens the circuit. The circuit stays closed when it
# should be open.


class CircuitShouldOpenState:
    """Two breakers share storage; each fails once with fail_max=2."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        self.breaker1 = _make_breaker(
            self.storage, fail_max=2, reset_timeout=60, state=STATE_CLOSED,
        )
        self.breaker2 = _make_breaker(
            self.storage, fail_max=2, reset_timeout=60, state=STATE_CLOSED,
        )

    def thread1(self):
        try:
            self.breaker1.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    def thread2(self):
        try:
            self.breaker2.call(self._fail)
        except (CircuitBreakerError, RuntimeError):
            pass

    @staticmethod
    def _fail():
        raise RuntimeError("simulated failure")


def _circuit_should_open_invariant(s: CircuitShouldOpenState) -> bool:
    # After 2 failures with fail_max=2, the circuit MUST be open.
    return s.storage.state == STATE_OPEN


def test_circuit_stays_closed_when_should_open():
    """Detect bug where circuit stays closed because lost update misses threshold."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: CircuitShouldOpenState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_circuit_should_open_invariant,
            max_attempts=500,
            max_ops=300,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 11. Half-open success counter lost update prevents close
# ===========================================================================
# With success_threshold=2, two concurrent on_success() in half-open should
# increment success_counter to 2 and close. Lost update leaves it at 1.


class HalfOpenSuccessCounterState:
    """Two threads call on_success() concurrently in half-open with threshold=2."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_HALF_OPEN)
        self.breaker = _make_breaker(
            self.storage, fail_max=1, reset_timeout=0,
            success_threshold=2, state=STATE_HALF_OPEN,
        )

    def thread1(self):
        self.breaker._state.on_success()

    def thread2(self):
        self.breaker._state.on_success()


def _half_open_success_counter_invariant(s: HalfOpenSuccessCounterState) -> bool:
    # After two on_success() with threshold=2, success_counter should be 2.
    return s.storage.success_counter == 2


def test_half_open_success_counter_lost_update():
    """Detect lost update on success_counter preventing half-open->closed transition."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: HalfOpenSuccessCounterState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_half_open_success_counter_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 12. TOCTOU in half-open on_success: both see threshold, both close
# ===========================================================================
# CircuitHalfOpenState.on_success():
#     self._breaker._state_storage.increment_success_counter()
#     if self._breaker._state_storage.success_counter >= threshold:
#         self._breaker.close()
#
# With threshold=1: thread A increments to 1, thread B increments to 2 (or
# lost-update keeps it at 1). Both read success_counter >= 1, both close().


class HalfOpenOnSuccessTOCTOUState:
    """Two threads hit on_success in half-open with threshold=1."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_HALF_OPEN)
        self.breaker = _make_breaker(
            self.storage, fail_max=1, reset_timeout=0,
            success_threshold=1, state=STATE_HALF_OPEN,
        )
        self.close_count = 0
        self._original_close = self.breaker.close
        self.breaker.close = self._counting_close

    def _counting_close(self):
        self.close_count += 1
        return self._original_close()

    def thread1(self):
        self.breaker._state.on_success()

    def thread2(self):
        self.breaker._state.on_success()


def _half_open_on_success_toctou_invariant(s: HalfOpenOnSuccessTOCTOUState) -> bool:
    # close() should be called at most once.
    return s.close_count <= 1


def test_half_open_on_success_toctou():
    """Detect TOCTOU where two threads both see threshold reached and both close()."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: HalfOpenOnSuccessTOCTOUState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_half_open_on_success_toctou_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# 13. Success counter reset vs increment race
# ===========================================================================
# reset_success_counter() racing with increment_success_counter().
# Starting from success_counter=5, result should be 0 or 1, not 6.


class SuccessResetVsIncrementState:
    """One thread resets success counter, one increments it. Starting at 5."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)
        for _ in range(5):
            self.storage.increment_success_counter()

    def thread1(self):
        self.storage.reset_success_counter()

    def thread2(self):
        self.storage.increment_success_counter()


def _success_reset_vs_increment_invariant(s: SuccessResetVsIncrementState) -> bool:
    return s.storage.success_counter in (0, 1)


def test_success_reset_vs_increment_race():
    """Detect race between reset_success_counter() and increment_success_counter()."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SuccessResetVsIncrementState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_success_reset_vs_increment_invariant,
            max_attempts=500,
            max_ops=200,
            seed=0,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Main: run all tests
# ===========================================================================

if __name__ == "__main__":
    tests = [
        ("1. Success counter lost update", test_success_counter_lost_update),
        ("2. Double closed-to-open transition", test_double_closed_to_open_transition),
        ("3. Half-open double close (on_success)", test_half_open_double_close),
        ("4. Reset counter vs increment race", test_reset_vs_increment_race),
        ("5. Closed state double fail (shared storage)", test_closed_state_double_fail_lost_update),
        ("6. Half-open success/fail race", test_half_open_success_fail_race),
        ("7. State property TOCTOU", test_state_property_toctou),
        ("8. Closed _handle_error race (direct)", test_closed_handle_error_race),
        ("9. Three-thread fail counter lost update", test_three_thread_fail_counter_lost_update),
        ("10. Circuit stays closed when should open", test_circuit_stays_closed_when_should_open),
        ("11. Half-open success counter lost update", test_half_open_success_counter_lost_update),
        ("12. Half-open on_success TOCTOU (double close)", test_half_open_on_success_toctou),
        ("13. Success reset vs increment race", test_success_reset_vs_increment_race),
    ]

    bugs_found = 0
    bugs_not_found = 0

    for name, test_fn in tests:
        print(f"\n{'=' * 70}")
        print(f"=== {name} ===")
        print(f"{'=' * 70}")
        try:
            result = test_fn()
            if hasattr(result, "property_holds"):
                if not result.property_holds:
                    bugs_found += 1
                    print(f"Result: BUG FOUND")
                else:
                    bugs_not_found += 1
                    print(f"Result: No bug found (in {result.num_explored} interleavings)")
        except Exception as e:
            bugs_not_found += 1
            print(f"Test raised exception: {e}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {bugs_found} bugs found, {bugs_not_found} tests did not find bugs")
    print(f"{'=' * 70}")
