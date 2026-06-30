"""Tests for serializable_invariant and error_on_any_race options.

These tests verify that:
1. serializable_invariant detects when interleaved execution produces
   a state not achievable by any sequential ordering.
2. error_on_any_race flags unsynchronized races even when the user
   invariant passes.
"""

import asyncio
import threading

import pytest

import frontrun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Counter:
    """Unsynchronized counter — classic lost-update bug."""

    def __init__(self):
        self.value = 0

    def __repr__(self):
        return f"Counter(value={self.value})"


def increment(state: Counter) -> None:
    temp = state.value
    state.value = temp + 1


class TwoCounters:
    """Two counters that are always equal when run sequentially."""

    def __init__(self):
        self.a = 0
        self.b = 0

    def __repr__(self):
        return f"TwoCounters(a={self.a}, b={self.b})"


def increment_both(state: TwoCounters) -> None:
    """Increment both counters — non-atomic, so interleaving can split them."""
    temp_a = state.a
    temp_b = state.b
    state.a = temp_a + 1
    state.b = temp_b + 1


class LockedCounter:
    """Properly synchronized counter — no races."""

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def __repr__(self):
        return f"LockedCounter(value={self.value})"


def locked_increment(state: LockedCounter) -> None:
    with state.lock:
        temp = state.value
        state.value = temp + 1


# ---------------------------------------------------------------------------
# Tests: serializable_invariant
# ---------------------------------------------------------------------------


class TestSerializableInvariantDpor:
    """Test serializable_invariant with explore(strategy="dpor")."""

    def test_lost_update_detected(self):
        """Two threads incrementing without locks — counter=1 is not serializable.

        Sequential orderings: [T0, T1] -> 2, [T1, T0] -> 2.
        Only valid final state: Counter(value=2).
        Interleaved can produce Counter(value=1) — not serializable.
        """
        result = frontrun.explore(
            setup=Counter,
            workers=[increment, increment],
            invariant=lambda s: True,  # trivially passing — bug should come from serializability
            serializable_invariant=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, (
            "serializable_invariant should detect lost update (counter=1 is not serializable)"
        )

    def test_locked_counter_is_serializable(self):
        """Properly locked counter is always serializable."""
        result = frontrun.explore(
            setup=LockedCounter,
            workers=[locked_increment, locked_increment],
            invariant=lambda s: True,
            serializable_invariant=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert result.property_holds, "locked counter should always be serializable"

    def test_custom_hash_function(self):
        """serializable_invariant accepts a callable for custom state hashing."""
        result = frontrun.explore(
            setup=Counter,
            workers=[increment, increment],
            invariant=lambda s: True,
            serializable_invariant=lambda s: s.value,  # custom hash
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, "serializable_invariant with custom hash should detect lost update"

    def test_two_counters_atomicity(self):
        """Two counters incremented non-atomically — interleaving can desync them."""
        result = frontrun.explore(
            setup=TwoCounters,
            workers=[increment_both, increment_both],
            invariant=lambda s: True,
            serializable_invariant=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        # Sequential: both counters always end at 2.
        # Interleaved: can end with a=2, b=1 (not serializable).
        assert not result.property_holds, "serializable_invariant should detect non-atomic two-counter update"


class TestSerializableInvariantBytecode:
    """Test serializable_invariant with bytecode explore_random."""

    def test_lost_update_detected(self):
        result = frontrun.explore_random(
            setup=Counter,
            threads=[increment, increment],
            invariant=lambda s: True,
            serializable_invariant=True,
            max_attempts=200,
            seed=42,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "serializable_invariant should detect lost update in bytecode shuffler"


class TestSerializableInvariantAsyncDpor:
    """Test serializable_invariant with async explore(strategy="dpor")."""

    def test_lost_update_detected(self):
        class AsyncCounter:
            def __init__(self):
                self.value = 0

            def __repr__(self):
                return f"AsyncCounter(value={self.value})"

        async def async_increment(state: AsyncCounter) -> None:
            temp = state.value
            await asyncio.sleep(0)
            state.value = temp + 1

        async def run():
            return await frontrun.explore(
                setup=AsyncCounter,
                workers=[async_increment, async_increment],
                invariant=lambda s: True,
                serializable_invariant=True,
                preemption_bound=2,
                reproduce_on_failure=0,
                strategy="dpor",
            )

        result = asyncio.run(run())
        assert not result.property_holds, "serializable_invariant should detect lost update in async DPOR"


class TestSerializableInvariantAsyncShuffler:
    """Test serializable_invariant with async shuffler explore_async_random."""

    def test_lost_update_detected(self):
        class AsyncCounter:
            def __init__(self):
                self.value = 0

            def __repr__(self):
                return f"AsyncCounter(value={self.value})"

        async def async_increment(state: AsyncCounter) -> None:
            temp = state.value
            await asyncio.sleep(0)
            state.value = temp + 1

        async def run():
            return await frontrun.explore_async_random(
                setup=AsyncCounter,
                tasks=[async_increment, async_increment],
                invariant=lambda s: True,
                serializable_invariant=True,
                max_attempts=200,
                seed=42,
            )

        result = asyncio.run(run())
        assert not result.property_holds, "serializable_invariant should detect lost update in async shuffler"


# ---------------------------------------------------------------------------
# Tests: error_on_any_race
# ---------------------------------------------------------------------------


class TestErrorOnAnyRaceDpor:
    """Test error_on_any_race with explore(strategy="dpor")."""

    def test_unsynchronized_write_detected(self):
        """Two threads writing same variable without locks — race even if invariant passes.

        The invariant `True` passes trivially, but the unsynchronized write
        should be flagged as a race.
        """
        result = frontrun.explore(
            setup=Counter,
            workers=[increment, increment],
            invariant=lambda s: True,
            error_on_any_race=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, "error_on_any_race should flag unsynchronized write-write race"

    def test_locked_counter_no_race(self):
        """Properly locked counter has no races."""
        result = frontrun.explore(
            setup=LockedCounter,
            workers=[locked_increment, locked_increment],
            invariant=lambda s: True,
            error_on_any_race=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert result.property_holds, "locked counter should have no races"

    def test_read_write_race_detected(self):
        """One thread reads, another writes — race detected."""

        class SharedState:
            def __init__(self):
                self.value = 0

        def writer(state: SharedState) -> None:
            state.value = 42

        def reader(state: SharedState) -> None:
            _ = state.value

        result = frontrun.explore(
            setup=SharedState,
            workers=[writer, reader],
            invariant=lambda s: True,
            error_on_any_race=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, "error_on_any_race should flag unsynchronized read-write race"


class TestErrorOnAnyRaceAsyncDpor:
    """Test error_on_any_race with async explore(strategy="dpor")."""

    def test_unsynchronized_write_detected(self):
        class AsyncCounter:
            def __init__(self):
                self.value = 0

        async def async_increment(state: AsyncCounter) -> None:
            temp = state.value
            await asyncio.sleep(0)
            state.value = temp + 1

        async def run():
            return await frontrun.explore(
                setup=AsyncCounter,
                workers=[async_increment, async_increment],
                invariant=lambda s: True,
                error_on_any_race=True,
                preemption_bound=2,
                reproduce_on_failure=0,
                strategy="dpor",
            )

        result = asyncio.run(run())
        assert not result.property_holds, "error_on_any_race should flag unsynchronized write in async DPOR"


class TestErrorOnAnyRaceBytecode:
    """Test error_on_any_race with bytecode shuffler.

    The bytecode shuffler does not have DPOR race detection, so
    error_on_any_race should raise ValueError.
    """

    def test_raises_valueerror(self):
        with pytest.raises(ValueError, match="error_on_any_race.*DPOR"):
            frontrun.explore_random(
                setup=Counter,
                threads=[increment, increment],
                invariant=lambda s: True,
                error_on_any_race=True,
                max_attempts=1,
                reproduce_on_failure=0,
            )


class TestErrorOnAnyRaceAsyncShuffler:
    """Test error_on_any_race with async shuffler.

    The async shuffler does not have DPOR race detection, so
    error_on_any_race should raise ValueError.
    """

    def test_raises_valueerror(self):
        async def async_noop(state: Counter) -> None:
            pass

        async def run():
            return await frontrun.explore_async_random(
                setup=Counter,
                tasks=[async_noop, async_noop],
                invariant=lambda s: True,
                error_on_any_race=True,
                max_attempts=1,
            )

        with pytest.raises(ValueError, match="error_on_any_race.*DPOR"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Tests: combined
# ---------------------------------------------------------------------------


class TestCombinedOptions:
    """Test serializable_invariant and error_on_any_race together."""

    def test_both_enabled_dpor(self):
        """Both options enabled — both should report failure."""
        result = frontrun.explore(
            setup=Counter,
            workers=[increment, increment],
            invariant=lambda s: True,
            serializable_invariant=True,
            error_on_any_race=True,
            preemption_bound=2,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds


# ---------------------------------------------------------------------------
# Tests: deadlock interaction (stop_on_first=False)
# ---------------------------------------------------------------------------


class _DeadlockState:
    """Two locks with shared counter — lock-order inversion causes deadlock."""

    def __init__(self):
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()
        self.value = 0

    def __repr__(self):
        return f"_DeadlockState(value={self.value})"


def _deadlock_thread0(state: _DeadlockState) -> None:
    with state.lock_a:
        state.value += 1
        with state.lock_b:
            pass


def _deadlock_thread1(state: _DeadlockState) -> None:
    with state.lock_b:
        state.value += 1
        with state.lock_a:
            pass


class TestSerializableInvariantDeadlockInteraction:
    """serializable_invariant must not fire on deadlocked executions.

    When a deadlock occurs, threads don't run to completion, so the
    state is partial/undefined.  Checking it against sequential baselines
    produces false positives.  The async_dpor version correctly guards
    with ``not is_deadlock``; the sync dpor.py must do the same.
    """

    def test_no_spurious_serializability_failure_on_deadlock(self):
        """Deadlock + serializable_invariant should not add duplicate failures.

        Each execution that deadlocks should appear exactly once in
        result.failures (from the deadlock detection), not a second time
        from the serializable_invariant check on partial state.
        """
        result = frontrun.explore(
            setup=_DeadlockState,
            workers=[_deadlock_thread0, _deadlock_thread1],
            invariant=lambda s: True,
            serializable_invariant=True,
            stop_on_first=False,
            max_executions=50,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, "Deadlock should be detected"
        assert "deadlock" in (result.explanation or "").lower(), (
            f"Explanation should mention deadlock, got: {result.explanation}"
        )
        # Each execution number should appear at most once in failures
        exec_nums = [num for num, _ in result.failures]
        assert len(exec_nums) == len(set(exec_nums)), f"Duplicate execution numbers in failures: {exec_nums}"


class TestErrorOnAnyRaceDeadlockInteraction:
    """error_on_any_race must not fire on deadlocked executions.

    A deadlock is not a "race" — it's a distinct failure mode.  Checking
    races on incomplete executions is misleading.  The async_dpor version
    correctly guards with ``not is_deadlock``; sync dpor.py must match.
    """

    def test_no_spurious_race_detection_on_deadlock(self):
        """Deadlock + error_on_any_race should not add duplicate failures."""
        result = frontrun.explore(
            setup=_DeadlockState,
            workers=[_deadlock_thread0, _deadlock_thread1],
            invariant=lambda s: True,
            error_on_any_race=True,
            stop_on_first=False,
            max_executions=50,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            reproduce_on_failure=0,
            strategy="dpor",
        )
        assert not result.property_holds, "Deadlock should be detected"
        # Each execution number should appear at most once in failures
        exec_nums = [num for num, _ in result.failures]
        assert len(exec_nums) == len(set(exec_nums)), f"Duplicate execution numbers in failures: {exec_nums}"
