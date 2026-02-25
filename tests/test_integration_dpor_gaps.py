"""Adversarial integration tests: race conditions that expose DPOR detection gaps.

All bug-exposing tests assert the *correct* (expected) behavior, so they
**fail** until the corresponding gap is fixed.  When a gap is fixed the
test will start passing and can be moved to the "CONTROLS" section.

Gaps tested (false negatives — real races DPOR misses):

1. ``LOAD_GLOBAL`` / ``STORE_GLOBAL`` — pushed/popped on the shadow
   stack but never reported as reads or writes.
2. C-level container method mutations — ``list.append()``,
   ``set.add()`` etc. execute in C; the shadow stack sees
   ``LOAD_ATTR`` (a read) but not the mutation.
"""

from __future__ import annotations

import threading

from frontrun.dpor import explore_dpor

# ============================================================================
# FALSE NEGATIVES — real races DPOR misses
# ============================================================================

# ---------------------------------------------------------------------------
# 1. Global variable lost-update (LOAD_GLOBAL / STORE_GLOBAL untracked)
# ---------------------------------------------------------------------------
#
# Two threads increment a module-level global.  The read (LOAD_GLOBAL)
# and write (STORE_GLOBAL) are not reported to the DPOR engine, so it
# sees zero conflicting accesses and explores only one interleaving.

_global_counter = 0


class _GlobalCounterState:
    def __init__(self) -> None:
        global _global_counter
        _global_counter = 0


def _global_increment(_state: _GlobalCounterState) -> None:
    global _global_counter
    tmp = _global_counter
    _global_counter = tmp + 1


def _global_invariant(_state: _GlobalCounterState) -> bool:
    return _global_counter == 2


class TestGlobalVariableRace:
    """DPOR misses lost-update on module-level globals (STORE_GLOBAL gap)."""

    def test_dpor_misses_global_race(self) -> None:
        """DPOR should find this race but doesn't — STORE_GLOBAL is untracked."""
        result = explore_dpor(
            setup=_GlobalCounterState,
            threads=[_global_increment, _global_increment],
            invariant=_global_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        # BUG: DPOR reports property_holds=True because LOAD_GLOBAL/STORE_GLOBAL
        # accesses are not reported to the engine.  This assertion documents the
        # expected-correct behavior — it will pass once the gap is fixed.
        assert not result.property_holds, "DPOR should detect the global-variable lost-update race"

    def test_barrier_proves_global_race_is_real(self) -> None:
        """Barrier-forced interleaving proves the lost update is real."""
        global _global_counter
        barrier = threading.Barrier(2)

        def handler() -> None:
            global _global_counter
            tmp = _global_counter
            barrier.wait()
            _global_counter = tmp + 1

        for _ in range(10):
            _global_counter = 0
            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if _global_counter != 2:
                return
        raise AssertionError("Barrier-forced global race never triggered in 10 attempts")


# ---------------------------------------------------------------------------
# 2. Global augmented assignment (LOAD_GLOBAL + BINARY_OP + STORE_GLOBAL)
# ---------------------------------------------------------------------------
#
# The simplest possible race: ``global_var += 1``.  Compiles to
# LOAD_GLOBAL → BINARY_OP → STORE_GLOBAL — none report accesses.

_simple_global = 0


class _SimpleGlobalState:
    def __init__(self) -> None:
        global _simple_global
        _simple_global = 0


def _simple_global_inc(_state: _SimpleGlobalState) -> None:
    global _simple_global
    _simple_global += 1


def _simple_global_check(_state: _SimpleGlobalState) -> bool:
    return _simple_global == 2


class TestSimpleGlobalIncrement:
    """``global_var += 1`` — completely invisible to DPOR."""

    def test_dpor_misses_augmented_global_assignment(self) -> None:
        result = explore_dpor(
            setup=_SimpleGlobalState,
            threads=[_simple_global_inc, _simple_global_inc],
            invariant=_simple_global_check,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the global += lost-update race"

    def test_barrier_proves_augmented_assign_race_is_real(self) -> None:
        global _simple_global
        barrier = threading.Barrier(2)

        def handler() -> None:
            global _simple_global
            tmp = _simple_global
            barrier.wait()
            _simple_global = tmp + 1

        for _ in range(10):
            _simple_global = 0
            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if _simple_global != 2:
                return
        raise AssertionError("Barrier-forced global += race never triggered")


# ---------------------------------------------------------------------------
# 3. Container method mutation — list.append check-then-act
# ---------------------------------------------------------------------------
#
# Two threads check ``len(items) < max_size`` then ``items.append()``.
# The append() call is a C-level method — the shadow stack sees
# LOAD_ATTR "items" and LOAD_ATTR "append" (reads) but not the list
# mutation (a write).  Without a write, DPOR sees no read-write
# conflict and doesn't backtrack.


class _ListAppendState:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.max_size = 1


def _list_append_thread(state: _ListAppendState) -> None:
    if len(state.items) < state.max_size:
        state.items.append("item")


def _list_append_invariant(state: _ListAppendState) -> bool:
    return len(state.items) <= state.max_size


class TestListAppendRace:
    """DPOR misses mutations through list.append() (C-level method gap)."""

    def test_dpor_misses_list_append_race(self) -> None:
        """list.append() mutation is invisible to the shadow stack.

        DPOR sees LOAD_ATTR reads on ``state.items`` from both threads
        but no write to the list, so it doesn't detect a conflict.
        """
        result = explore_dpor(
            setup=_ListAppendState,
            threads=[_list_append_thread, _list_append_thread],
            invariant=_list_append_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the list.append check-then-act race"

    def test_barrier_proves_list_append_race_is_real(self) -> None:
        barrier = threading.Barrier(2)
        items: list[str] = []

        def handler() -> None:
            size = len(items)
            barrier.wait()
            if size < 1:
                items.append("item")

        for _ in range(10):
            items.clear()
            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if len(items) > 1:
                return
        raise AssertionError("Barrier-forced list.append race never triggered")


# ---------------------------------------------------------------------------
# 4. Container method mutation — set.add check-then-act
# ---------------------------------------------------------------------------
#
# Two threads check ``item not in seen`` then ``seen.add(item)``.
# Both ``__contains__`` and ``add`` execute in C.  In the first
# (serialized) execution, thread 0 adds the item, so thread 1 sees
# it already present and skips the body.  DPOR needs a WRITE event
# from set.add() to backtrack and try the other ordering — but it
# never sees one.


class _SetAddState:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.first_adders = 0


def _set_check_and_add(state: _SetAddState) -> None:
    if "shared-item" not in state.seen:
        state.first_adders += 1
        state.seen.add("shared-item")


def _set_add_invariant(state: _SetAddState) -> bool:
    return state.first_adders == 1


class TestSetAddRace:
    """DPOR misses check-then-act on set (C-level __contains__ + add)."""

    def test_dpor_misses_set_add_race(self) -> None:
        """set.__contains__ and set.add both execute in C — invisible.

        In the first execution, thread 0 adds the item and thread 1
        sees it already present.  DPOR would need to see set.add as a
        WRITE to backtrack and explore the ordering where both threads
        check before either adds.
        """
        result = explore_dpor(
            setup=_SetAddState,
            threads=[_set_check_and_add, _set_check_and_add],
            invariant=_set_add_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the set.add check-then-act race"

    def test_barrier_proves_set_add_race_is_real(self) -> None:
        barrier = threading.Barrier(2)
        seen: set[str] = set()
        first_count = [0]

        def handler() -> None:
            present = "shared-item" in seen
            barrier.wait()
            if not present:
                first_count[0] += 1
                seen.add("shared-item")

        for _ in range(10):
            seen.clear()
            first_count[0] = 0
            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if first_count[0] != 1:
                return
        raise AssertionError("Barrier-forced set.add race never triggered")


# ============================================================================
# CONTROLS — DPOR works correctly on these
# ============================================================================

# ---------------------------------------------------------------------------
# 5. Semaphore-protected counter (spin-yield enforces mutual exclusion)
# ---------------------------------------------------------------------------
#
# CooperativeSemaphore doesn't report lock_acquire/lock_release sync events,
# but its spin-yield behaviour still enforces mutual exclusion at the
# cooperative-scheduling level.  DPOR explores extra interleavings
# (explored=3 vs explored=1 for Lock) because it doesn't know about the
# Semaphore dependency — but it still finds property_holds=True.


class _SemaphoreCounterState:
    def __init__(self) -> None:
        self.counter = 0
        self.sem = threading.Semaphore(1)


def _semaphore_increment(state: _SemaphoreCounterState) -> None:
    state.sem.acquire()
    tmp = state.counter
    state.counter = tmp + 1
    state.sem.release()


def _semaphore_invariant(state: _SemaphoreCounterState) -> bool:
    return state.counter == 2


class TestSemaphoreControlCorrect:
    """Control: Semaphore spin-yield enforces mutual exclusion."""

    def test_dpor_correctly_handles_semaphore(self) -> None:
        result = explore_dpor(
            setup=_SemaphoreCounterState,
            threads=[_semaphore_increment, _semaphore_increment],
            invariant=_semaphore_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, (
            "DPOR incorrectly reports a race on Semaphore-protected code! "
            "CooperativeSemaphore spin-yield may be broken."
        )


# ---------------------------------------------------------------------------
# 6. Lock-protected counter (sync events reported correctly)
# ---------------------------------------------------------------------------


class _LockCounterState:
    def __init__(self) -> None:
        self.counter = 0
        self.lock = threading.Lock()


def _lock_increment(state: _LockCounterState) -> None:
    state.lock.acquire()
    tmp = state.counter
    state.counter = tmp + 1
    state.lock.release()


def _lock_invariant(state: _LockCounterState) -> bool:
    return state.counter == 2


class TestLockControlCorrect:
    """Control: Lock sync IS tracked — DPOR correctly finds no race."""

    def test_dpor_correctly_handles_lock(self) -> None:
        result = explore_dpor(
            setup=_LockCounterState,
            threads=[_lock_increment, _lock_increment],
            invariant=_lock_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, (
            "DPOR incorrectly reports a race on Lock-protected code! Lock sync reporting may be broken."
        )


# ---------------------------------------------------------------------------
# 7. Global dict with subscript access (STORE_SUBSCR IS tracked)
# ---------------------------------------------------------------------------
#
# Even though the dict is loaded via LOAD_GLOBAL (untracked), the
# subsequent subscript operations (BINARY_SUBSCR, STORE_SUBSCR) ARE
# tracked because the shadow stack correctly resolves the dict object.

_tracked_dict: dict[str, int] = {}


class _TrackedDictState:
    def __init__(self) -> None:
        _tracked_dict.clear()


def _tracked_dict_inc(_state: _TrackedDictState) -> None:
    current = _tracked_dict.get("count", 0)
    _tracked_dict["count"] = current + 1


def _tracked_dict_inv(_state: _TrackedDictState) -> bool:
    return _tracked_dict.get("count") == 2


class TestGlobalDictControlDetected:
    """Control: global dict subscript operations ARE tracked by DPOR."""

    def test_dpor_detects_global_dict_race(self) -> None:
        """DPOR correctly detects the race because STORE_SUBSCR reports
        a write on the dict object even though it came from LOAD_GLOBAL."""
        result = explore_dpor(
            setup=_TrackedDictState,
            threads=[_tracked_dict_inc, _tracked_dict_inc],
            invariant=_tracked_dict_inv,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect this via STORE_SUBSCR tracking"
