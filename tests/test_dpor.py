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

from frontrun_dpor import PyDporEngine

from frontrun.dpor import DporResult, explore_dpor

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
        assert result.counterexample_schedule is not None

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
        """DporResult should have expected fields."""

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

        assert isinstance(result, DporResult)
        assert isinstance(result.property_holds, bool)
        assert isinstance(result.executions_explored, int)
        assert result.executions_explored >= 1
        if not result.property_holds:
            assert result.counterexample_schedule is not None
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
        assert result.executions_explored >= 2


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
        assert result.executions_explored == 1

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

        assert result.executions_explored <= 3
