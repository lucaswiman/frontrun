"""
Tests for the bytecode-tracing DPOR implementation.

Covers:
- Rust DPOR engine via PyO3 (low-level API)
- Python orchestration with bytecode tracing and shadow stack
- Classic concurrency bugs: lost update, bank account transfer
- Lock-protected code (no bugs)
- Independent threads (minimal exploration)
- Edge cases
"""

from __future__ import annotations

import threading

from frontrun._dpor import PyDporEngine

from frontrun.common import InterleavingResult
from frontrun.dpor import explore_dpor

# ---------------------------------------------------------------------------
# Low-level Rust engine tests (via PyO3)
# ---------------------------------------------------------------------------


class TestPyDporEngine:
    def test_creation(self) -> None:
        engine = PyDporEngine(2)
        assert engine.executions_completed == 0
        assert engine.num_threads == 2

    def test_single_thread(self) -> None:
        engine = PyDporEngine(1)
        execution = engine.begin_execution()
        chosen = engine.schedule(execution)
        assert chosen == 0
        execution.finish_thread(0)
        assert not engine.next_execution()

    def test_two_threads_no_conflict(self) -> None:
        engine = PyDporEngine(2)
        execution = engine.begin_execution()

        t0 = engine.schedule(execution)
        assert t0 == 0
        engine.report_access(execution, 0, 1, "write")
        execution.finish_thread(0)

        t1 = engine.schedule(execution)
        assert t1 == 1
        engine.report_access(execution, 1, 2, "write")
        execution.finish_thread(1)

        assert not engine.next_execution()
        assert engine.executions_completed == 1

    def test_two_threads_write_write_conflict(self) -> None:
        engine = PyDporEngine(2)
        exec_count = 0

        while True:
            execution = engine.begin_execution()
            first = engine.schedule(execution)
            engine.report_access(execution, first, 1, "write")
            execution.finish_thread(first)

            second = engine.schedule(execution)
            engine.report_access(execution, second, 1, "write")
            execution.finish_thread(second)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 2

    def test_read_read_no_conflict(self) -> None:
        engine = PyDporEngine(2)
        execution = engine.begin_execution()

        first = engine.schedule(execution)
        engine.report_access(execution, first, 1, "read")
        execution.finish_thread(first)

        second = engine.schedule(execution)
        engine.report_access(execution, second, 1, "read")
        execution.finish_thread(second)

        assert not engine.next_execution()

    def test_read_write_conflict(self) -> None:
        engine = PyDporEngine(2)
        exec_count = 0

        while True:
            execution = engine.begin_execution()
            first = engine.schedule(execution)
            engine.report_access(execution, first, 1, "write")
            execution.finish_thread(first)

            second = engine.schedule(execution)
            engine.report_access(execution, second, 1, "read")
            execution.finish_thread(second)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 2

    def test_max_executions_limit(self) -> None:
        engine = PyDporEngine(2, max_executions=1)
        execution = engine.begin_execution()

        engine.schedule(execution)
        engine.report_access(execution, 0, 1, "write")
        execution.finish_thread(0)

        engine.schedule(execution)
        engine.report_access(execution, 1, 1, "write")
        execution.finish_thread(1)

        assert not engine.next_execution()
        assert engine.executions_completed == 1

    def test_sync_lock_acquire_release(self) -> None:
        """Lock sync events should not crash."""
        engine = PyDporEngine(2)
        execution = engine.begin_execution()

        engine.schedule(execution)
        engine.report_sync(execution, 0, "lock_acquire", 99)
        engine.report_access(execution, 0, 1, "write")
        engine.report_sync(execution, 0, "lock_release", 99)
        execution.finish_thread(0)

        engine.schedule(execution)
        engine.report_sync(execution, 1, "lock_acquire", 99)
        engine.report_access(execution, 1, 1, "write")
        engine.report_sync(execution, 1, "lock_release", 99)
        execution.finish_thread(1)

        # Should complete without crash
        engine.next_execution()

    def test_schedule_trace(self) -> None:
        engine = PyDporEngine(2)
        execution = engine.begin_execution()

        engine.schedule(execution)
        execution.finish_thread(0)
        engine.schedule(execution)
        execution.finish_thread(1)

        assert execution.schedule_trace == [0, 1]

    def test_runnable_threads(self) -> None:
        engine = PyDporEngine(3)
        execution = engine.begin_execution()
        assert execution.runnable_threads() == [0, 1, 2]
        execution.finish_thread(1)
        assert execution.runnable_threads() == [0, 2]

    def test_block_unblock(self) -> None:
        engine = PyDporEngine(2)
        execution = engine.begin_execution()
        execution.block_thread(0)
        assert execution.runnable_threads() == [1]
        execution.unblock_thread(0)
        assert execution.runnable_threads() == [0, 1]

    def test_counter_lost_update_model(self) -> None:
        """Model-level test: two threads doing read-modify-write."""
        engine = PyDporEngine(2, max_executions=500)
        found_bug = False

        while True:
            execution = engine.begin_execution()
            counter = [0]
            local = [0, 0]
            pcs = [0, 0]

            while True:
                for i in range(2):
                    if pcs[i] >= 2:
                        execution.finish_thread(i)

                if not execution.runnable_threads():
                    break

                chosen = engine.schedule(execution)
                if chosen is None:
                    break

                pc = pcs[chosen]
                if pc >= 2:
                    break

                if pc == 0:
                    engine.report_access(execution, chosen, 0, "read")
                    local[chosen] = counter[0]
                else:
                    engine.report_access(execution, chosen, 0, "write")
                    counter[0] = local[chosen] + 1

                pcs[chosen] += 1

            if counter[0] != 2:
                found_bug = True

            if not engine.next_execution():
                break

        assert found_bug


# ---------------------------------------------------------------------------
# High-level explore_dpor tests (full bytecode tracing)
# ---------------------------------------------------------------------------


class TestExploreDpor:
    def test_lost_update_bug(self) -> None:
        """Two threads doing read-modify-write on a shared counter.
        DPOR should find the lost-update interleaving."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert len(result.failures) > 0
        assert result.counterexample is not None

    def test_lost_update_via_getter_setter(self) -> None:
        """Lost update through getter/setter methods.

        Regression test: on Python 3.14, LOAD_FAST_BORROW_LOAD_FAST_BORROW
        was not handled by the shadow stack, causing STORE_ATTR to pop None
        instead of the actual object.  This made DPOR miss the write access
        entirely, so it never explored the interleaving that reveals the race.
        """

        class AccountBalance:
            def __init__(self) -> None:
                self._balance = 0

            def get_balance(self) -> int:
                return self._balance

            def set_balance(self, value: int) -> None:
                self._balance = value

            def deposit(self, amount: int) -> None:
                current = self.get_balance()
                self.set_balance(current + amount)

        result = explore_dpor(
            setup=AccountBalance,
            threads=[lambda bal: bal.deposit(100), lambda bal: bal.deposit(100)],
            invariant=lambda bal: bal.get_balance() == 200,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert len(result.failures) > 0
        assert result.counterexample is not None

    def test_atomic_increment_no_bug(self) -> None:
        """Each thread does a single atomic write. No race possible."""

        class AtomicCounter:
            def __init__(self) -> None:
                self.value = 0

            def atomic_set(self, val: int) -> None:
                self.value = val

        result = explore_dpor(
            setup=AtomicCounter,
            threads=[
                lambda c: setattr(c, "value", c.value + 1),
                lambda c: setattr(c, "value", c.value + 1),
            ],
            invariant=lambda c: c.value >= 1,  # at least one increment
            max_executions=100,
            preemption_bound=2,
        )

        assert result.property_holds

    def test_bank_account_transfer(self) -> None:
        """Classic bank account race: two transfers from the same account."""

        class BankAccount:
            def __init__(self, balance: int) -> None:
                self.balance = balance

        class Bank:
            def __init__(self) -> None:
                self.a = BankAccount(100)
                self.b = BankAccount(100)

            def transfer(self, amount: int) -> None:
                temp_a = self.a.balance
                temp_b = self.b.balance
                self.a.balance = temp_a - amount
                self.b.balance = temp_b + amount

        result = explore_dpor(
            setup=Bank,
            threads=[lambda b: b.transfer(50), lambda b: b.transfer(50)],
            invariant=lambda b: b.a.balance + b.b.balance == 200,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds

    def test_lock_protected_counter(self) -> None:
        """Counter protected by a lock should always be correct."""

        class LockedCounter:
            def __init__(self) -> None:
                self.value = 0
                self.lock = threading.Lock()

            def increment(self) -> None:
                with self.lock:
                    temp = self.value
                    self.value = temp + 1

        result = explore_dpor(
            setup=LockedCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=50,
            preemption_bound=2,
        )

        assert result.property_holds

    def test_independent_objects(self) -> None:
        """Threads accessing independent objects need minimal exploration."""

        class State:
            def __init__(self) -> None:
                self.a = 0
                self.b = 0

        result = explore_dpor(
            setup=State,
            threads=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 1),
            ],
            invariant=lambda s: s.a == 1 and s.b == 1,
            max_executions=100,
            preemption_bound=2,
        )

        assert result.property_holds

    def test_result_structure(self) -> None:
        """explore_dpor result should have expected fields."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=50,
            preemption_bound=2,
        )

        assert isinstance(result, InterleavingResult)
        assert isinstance(result.property_holds, bool)
        assert isinstance(result.num_explored, int)
        assert result.num_explored >= 1
        if not result.property_holds:
            assert result.counterexample is not None
            assert isinstance(result.failures, list)

    def test_augmented_assignment_bug(self) -> None:
        """The += pattern should be detected by the shadow stack."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                self.value += 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds

    def test_preemption_bound_none_unbounded(self) -> None:
        """preemption_bound=None should do full (unbounded) DPOR."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            preemption_bound=None,
            max_executions=500,
        )

        assert not result.property_holds

    def test_three_threads_counter(self) -> None:
        """Three threads incrementing a shared counter."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[
                lambda c: c.increment(),
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
            invariant=lambda c: c.value == 3,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert result.num_explored >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_thread(self) -> None:
        """Single thread should always pass."""

        class State:
            def __init__(self) -> None:
                self.value = 0

            def run(self) -> None:
                self.value = 42

        result = explore_dpor(
            setup=State,
            threads=[lambda s: s.run()],
            invariant=lambda s: s.value == 42,
            max_executions=10,
        )

        assert result.property_holds
        assert result.num_explored == 1

    def test_max_executions_respected(self) -> None:
        """max_executions should limit exploration."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: True,  # Always passes
            max_executions=3,
            preemption_bound=None,
        )

        assert result.num_explored <= 3


# ---------------------------------------------------------------------------
# Global variable races (LOAD_GLOBAL / STORE_GLOBAL tracking)
# ---------------------------------------------------------------------------

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


class TestGlobalVariableRace:
    """DPOR detects lost-update on module-level globals (LOAD_GLOBAL / STORE_GLOBAL)."""

    def test_dpor_detects_global_race(self) -> None:
        result = explore_dpor(
            setup=_GlobalCounterState,
            threads=[_global_increment, _global_increment],
            invariant=_global_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the global-variable lost-update race"

    def test_dpor_detects_augmented_global_assignment(self) -> None:
        """``global_var += 1`` compiles to LOAD_GLOBAL + BINARY_OP + STORE_GLOBAL."""
        result = explore_dpor(
            setup=_SimpleGlobalState,
            threads=[_simple_global_inc, _simple_global_inc],
            invariant=_simple_global_check,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the global += lost-update race"

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
# C-level container method races (CALL handler tracks builtin methods)
# ---------------------------------------------------------------------------


class _ListAppendState:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.max_size = 1


def _list_append_thread(state: _ListAppendState) -> None:
    if len(state.items) < state.max_size:
        state.items.append("item")


def _list_append_invariant(state: _ListAppendState) -> bool:
    return len(state.items) <= state.max_size


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


class TestContainerMethodRace:
    """DPOR detects C-level container mutations (list.append, set.add, etc.)."""

    def test_dpor_detects_list_append_race(self) -> None:
        """list.append() executes in C but the CALL handler reports the write."""
        result = explore_dpor(
            setup=_ListAppendState,
            threads=[_list_append_thread, _list_append_thread],
            invariant=_list_append_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the list.append check-then-act race"

    def test_dpor_detects_set_add_race(self) -> None:
        """set.add() executes in C but the CALL handler reports the write."""
        result = explore_dpor(
            setup=_SetAddState,
            threads=[_set_check_and_add, _set_check_and_add],
            invariant=_set_add_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the set.add check-then-act race"

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


# ---------------------------------------------------------------------------
# Sync primitive correctness (Lock and Semaphore protect critical sections)
# ---------------------------------------------------------------------------


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


_tracked_dict: dict[str, int] = {}


class _TrackedDictState:
    def __init__(self) -> None:
        _tracked_dict.clear()


def _tracked_dict_inc(_state: _TrackedDictState) -> None:
    current = _tracked_dict.get("count", 0)
    _tracked_dict["count"] = current + 1


def _tracked_dict_inv(_state: _TrackedDictState) -> bool:
    return _tracked_dict.get("count") == 2


class TestSyncPrimitiveCorrectness:
    """DPOR correctly handles lock/semaphore-protected critical sections."""

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
            "CooperativeSemaphore sync reporting may be broken."
        )

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

    def test_dpor_detects_global_dict_race(self) -> None:
        """DPOR detects the race via STORE_SUBSCR even though the dict is loaded via LOAD_GLOBAL."""
        result = explore_dpor(
            setup=_TrackedDictState,
            threads=[_tracked_dict_inc, _tracked_dict_inc],
            invariant=_tracked_dict_inv,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect this via STORE_SUBSCR tracking"
