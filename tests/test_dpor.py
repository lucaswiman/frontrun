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
from typing import Any

import pytest

import frontrun
from frontrun._deadlock import SchedulerAbort
from frontrun._dpor import PyDporEngine
from frontrun._dpor_runtime.runner import DporBytecodeRunner
from frontrun.common import InterleavingResult
from frontrun.dpor import DporScheduler


class _CountingCondition:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.wait_calls = 0
        self.notify_calls = 0

    def __enter__(self) -> _CountingCondition:
        self._lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._lock.release()

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_calls += 1
        return False

    def notify_all(self) -> None:
        self.notify_calls += 1


def test_dpor_report_and_wait_reschedules_done_current_without_timeout() -> None:
    """A stale current thread that already finished should not cost deadlock_timeout."""

    class Host:
        def __init__(self) -> None:
            self._condition = _CountingCondition()
            self._finished = False
            self._error = None
            self._current_thread = 1
            self._threads_done = {1}
            self.num_threads = 2
            self.virtual_clock = None
            self.deadlock_timeout = 0.01
            self._preload_bridge = None
            self.trace_recorder = None
            self._pending_io_by_thread: dict[int, list[tuple[int, str, bool]]] = {}
            self._lock_depth_by_thread: dict[int, int] = {}
            self._switch_point_collector = None
            self._step_event_collector = None
            self._last_scheduled_path_id = None
            self.scheduled = 0

        _report_and_wait = DporScheduler._report_and_wait
        report_and_wait = DporScheduler.report_and_wait
        _condition_wait_timeout = DporScheduler._condition_wait_timeout
        _reschedule_done_current_unlocked = DporScheduler._reschedule_done_current_unlocked
        _flush_other_pending_io_for_current_io_unlocked = DporScheduler._flush_other_pending_io_for_current_io_unlocked
        _flush_pending_io_for_unlocked = DporScheduler._flush_pending_io_for_unlocked

        def _schedule_next(self) -> int | None:
            self.scheduled += 1
            return 0 if self.scheduled == 1 else None

    host = Host()

    assert host.report_and_wait(None, 0)
    assert host._condition.wait_calls == 0

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


def test_dpor_runner_cleans_up_runtime_state_after_error(monkeypatch) -> None:
    class _StubScheduler:
        def __init__(self) -> None:
            self.error: Exception | None = None
            self.done: list[int] = []
            self.lifecycle: list[str] = []

        def report_error(self, error: Exception) -> None:
            self.error = error

        def mark_done(self, thread_id: int) -> None:
            self.done.append(thread_id)

        def register_worker_thread(self) -> None:
            self.lifecycle.append("register")

        def unregister_worker_thread(self) -> None:
            self.lifecycle.append("unregister")

    scheduler = _StubScheduler()
    runner = DporBytecodeRunner(scheduler)  # type: ignore[arg-type]
    events: list[str] = []

    def setup(thread_id: int) -> None:
        scheduler.register_worker_thread()
        events.append(f"setup:{thread_id}")

    monkeypatch.setattr(runner, "_setup_dpor_tls", setup)
    monkeypatch.setattr(runner, "_teardown_dpor_tls", lambda: events.append("teardown"))
    monkeypatch.setattr(
        "frontrun._dpor_runtime.runner.install_thread_opcode_trace",
        lambda _handle: events.append("trace:on"),
    )
    monkeypatch.setattr(
        "frontrun._dpor_runtime.runner.uninstall_thread_opcode_trace",
        lambda _handle: events.append("trace:off"),
    )

    def boom() -> None:
        raise RuntimeError("boom")

    runner._opcode_handle = object()  # type: ignore[assignment]  # sentinel triggers install/uninstall
    runner._run_thread(0, boom, ())

    assert isinstance(runner.errors[0], RuntimeError)
    assert isinstance(scheduler.error, RuntimeError)
    assert scheduler.done == [0]
    assert scheduler.lifecycle == ["register", "unregister"]
    assert events == ["setup:0", "trace:on", "trace:off", "teardown"]


def test_dpor_runner_swallows_scheduler_abort(monkeypatch) -> None:
    class _StubScheduler:
        def __init__(self) -> None:
            self.error: Exception | None = None
            self.done: list[int] = []
            self.lifecycle: list[str] = []

        def report_error(self, error: Exception) -> None:
            self.error = error

        def mark_done(self, thread_id: int) -> None:
            self.done.append(thread_id)

        def register_worker_thread(self) -> None:
            self.lifecycle.append("register")

        def unregister_worker_thread(self) -> None:
            self.lifecycle.append("unregister")

    scheduler = _StubScheduler()
    runner = DporBytecodeRunner(scheduler)  # type: ignore[arg-type]
    events: list[str] = []

    def setup(thread_id: int) -> None:
        scheduler.register_worker_thread()
        events.append(f"setup:{thread_id}")

    monkeypatch.setattr(runner, "_setup_dpor_tls", setup)
    monkeypatch.setattr(runner, "_teardown_dpor_tls", lambda: events.append("teardown"))

    def abort() -> None:
        raise SchedulerAbort()

    runner._run_thread(0, abort, ())

    assert runner.errors == {}
    assert scheduler.error is None
    assert scheduler.done == [0]
    assert scheduler.lifecycle == ["register", "unregister"]
    assert events == ["setup:0", "teardown"]


# ---------------------------------------------------------------------------
# High-level explore() (DPOR) tests (full bytecode tracing)
# ---------------------------------------------------------------------------


class TestWakeupTreeEngine:
    """Tests verifying the wakeup tree-based backtracking in the DPOR engine."""

    def test_four_threads_write_conflict_exhaustive(self) -> None:
        """Four threads all writing to the same object should explore 4! = 24 orderings."""
        engine = PyDporEngine(4)
        exec_count = 0

        while True:
            execution = engine.begin_execution()

            while True:
                runnable = execution.runnable_threads()
                if not runnable:
                    break
                chosen = engine.schedule(execution)
                if chosen is None:
                    break
                engine.report_access(execution, chosen, 1, "write")
                execution.finish_thread(chosen)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 24, f"Expected 24 orderings (4!), got {exec_count}"

    def test_independent_pairs_optimal(self) -> None:
        """Two independent pairs (T0/T1 on X, T2/T3 on Y).

        The Mazurkiewicz-optimal count is 2*2=4 orderings.  Since T0/T1
        conflict on object 1 and T2/T3 conflict on object 2, but there
        are no cross-pair conflicts, DPOR explores exactly 4 interleavings.
        """
        engine = PyDporEngine(4)
        thread_objects = [1, 1, 2, 2]
        exec_count = 0

        while True:
            execution = engine.begin_execution()

            while True:
                runnable = execution.runnable_threads()
                if not runnable:
                    break
                chosen = engine.schedule(execution)
                if chosen is None:
                    break
                engine.report_access(execution, chosen, thread_objects[chosen], "write")
                execution.finish_thread(chosen)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 4, f"Expected 4 orderings (2*2 for independent pairs), got {exec_count}"

    def test_read_read_no_backtrack(self) -> None:
        """Two threads reading the same object should produce exactly 1 execution."""
        engine = PyDporEngine(2)
        exec_count = 0

        while True:
            execution = engine.begin_execution()
            for _ in range(2):
                chosen = engine.schedule(execution)
                if chosen is None:
                    break
                engine.report_access(execution, chosen, 1, "read")
                execution.finish_thread(chosen)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 1, f"Read-read should be independent, got {exec_count}"


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

        result = frontrun.explore(
            setup=Counter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert len(result.failures) > 0
        assert result.counterexample is not None

    def test_assert_style_invariant_reproduces_like_boolean(self) -> None:
        """An assert-style invariant must reproduce a DPOR counterexample as
        reliably as the equivalent boolean invariant.

        The sync reproduction loop evaluated the invariant raw instead of via
        ``check_invariant``, so an assert-style invariant raised
        ``AssertionError`` which the broad ``except Exception: continue``
        swallowed, scoring every replay as a non-reproduction (0/N) even though
        the counterexample is a real, deterministic failure.
        """
        from frontrun._dpor_runtime.replay import _reproduce_dpor_counterexample

        class State:
            def __init__(self) -> None:
                self.counter = 0

        def setup() -> State:
            return State()

        def worker(s: State) -> None:
            x = s.counter
            s.counter = x

        threads = [worker, worker]
        schedule = [0, 1, 0, 1, 0, 1, 0, 1]

        def inv_bool(s: State) -> bool:
            return s.counter == 2  # always False for this schedule -> "fails"

        def inv_assert(s: State) -> None:
            assert s.counter == 2  # always raises AssertionError

        def reproduce(invariant) -> tuple[int, int]:
            return _reproduce_dpor_counterexample(
                schedule_list=schedule,
                setup=setup,
                threads=threads,
                timeout_per_run=5.0,
                deadlock_timeout=2.0,
                reproduce_on_failure=5,
                lock_timeout=None,
                invariant=invariant,
                detect_io=False,
            )

        bool_attempts, bool_successes = reproduce(inv_bool)
        assert_attempts, assert_successes = reproduce(inv_assert)

        assert bool_successes > 0
        # Same deterministic failing state: reproduction must not depend on
        # whether the invariant returns False or raises AssertionError.
        assert assert_successes == bool_successes
        assert assert_attempts == bool_attempts

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

        result = frontrun.explore(
            setup=AccountBalance,
            workers=[lambda bal: bal.deposit(100), lambda bal: bal.deposit(100)],
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

        result = frontrun.explore(
            setup=AtomicCounter,
            workers=[
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

        result = frontrun.explore(
            setup=Bank,
            workers=[lambda b: b.transfer(50), lambda b: b.transfer(50)],
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

        result = frontrun.explore(
            setup=LockedCounter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
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

        result = frontrun.explore(
            setup=State,
            workers=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 1),
            ],
            invariant=lambda s: s.a == 1 and s.b == 1,
            max_executions=100,
            preemption_bound=2,
        )

        assert result.property_holds

    def test_result_structure(self) -> None:
        """explore() result should have expected fields."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = frontrun.explore(
            setup=Counter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
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

        result = frontrun.explore(
            setup=Counter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
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

        result = frontrun.explore(
            setup=Counter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
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

        result = frontrun.explore(
            setup=Counter,
            workers=[
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

        result = frontrun.explore(
            setup=State,
            workers=[lambda s: s.run()],
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

        result = frontrun.explore(
            setup=Counter,
            workers=[lambda c: c.increment(), lambda c: c.increment()],
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
        result = frontrun.explore(
            setup=_GlobalCounterState,
            workers=[_global_increment, _global_increment],
            invariant=_global_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the global-variable lost-update race"

    def test_dpor_detects_augmented_global_assignment(self) -> None:
        """``global_var += 1`` compiles to LOAD_GLOBAL + BINARY_OP + STORE_GLOBAL."""
        result = frontrun.explore(
            setup=_SimpleGlobalState,
            workers=[_simple_global_inc, _simple_global_inc],
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
        result = frontrun.explore(
            setup=_ListAppendState,
            workers=[_list_append_thread, _list_append_thread],
            invariant=_list_append_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the list.append check-then-act race"

    def test_dpor_detects_set_add_race(self) -> None:
        """set.add() executes in C but the CALL handler reports the write."""
        result = frontrun.explore(
            setup=_SetAddState,
            workers=[_set_check_and_add, _set_check_and_add],
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


class _BoundedSemaphoreCounterState:
    def __init__(self) -> None:
        self.counter = 0
        self.sem = threading.BoundedSemaphore(1)


def _bounded_semaphore_increment(state: _BoundedSemaphoreCounterState) -> None:
    state.sem.acquire()
    tmp = state.counter
    state.counter = tmp + 1
    state.sem.release()


def _bounded_semaphore_invariant(state: _BoundedSemaphoreCounterState) -> bool:
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
        result = frontrun.explore(
            setup=_SemaphoreCounterState,
            workers=[_semaphore_increment, _semaphore_increment],
            invariant=_semaphore_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, (
            "DPOR incorrectly reports a race on Semaphore-protected code! "
            "CooperativeSemaphore sync reporting may be broken."
        )

    def test_dpor_correctly_handles_bounded_semaphore(self) -> None:
        result = frontrun.explore(
            setup=_BoundedSemaphoreCounterState,
            workers=[_bounded_semaphore_increment, _bounded_semaphore_increment],
            invariant=_bounded_semaphore_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, (
            "DPOR incorrectly reports a race on BoundedSemaphore-protected code! "
            "CooperativeBoundedSemaphore sync reporting may be broken."
        )

    def test_dpor_correctly_handles_lock(self) -> None:
        result = frontrun.explore(
            setup=_LockCounterState,
            workers=[_lock_increment, _lock_increment],
            invariant=_lock_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, (
            "DPOR incorrectly reports a race on Lock-protected code! Lock sync reporting may be broken."
        )

    def test_dpor_detects_global_dict_race(self) -> None:
        """DPOR detects the race via STORE_SUBSCR even though the dict is loaded via LOAD_GLOBAL."""
        result = frontrun.explore(
            setup=_TrackedDictState,
            workers=[_tracked_dict_inc, _tracked_dict_inc],
            invariant=_tracked_dict_inv,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect this via STORE_SUBSCR tracking"


# ---------------------------------------------------------------------------
# Deadlock detection as invariant violations
# ---------------------------------------------------------------------------


class TestDeadlockAsInvariantViolation:
    """Deadlocks should surface as property_holds=False, not silently time out."""

    def test_cooperative_lock_deadlock_is_reported(self) -> None:
        """Two threads with lock-order inversion: deadlock → property_holds=False.

        Each thread writes shared state INSIDE its first lock, creating a
        conflict that DPOR tracks.  The conflict forces exploration of the
        ordering where T0 holds lock_a and T1 holds lock_b simultaneously,
        which exposes the deadlock via WaitForGraph cycle detection.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.x = 0

        def thread0(s: State) -> None:
            with s.lock_a:
                s.x += 1  # write inside lock_a — DPOR uses this as the conflict point
                with s.lock_b:
                    pass

        def thread1(s: State) -> None:
            with s.lock_b:
                s.x += 1  # write inside lock_b — conflict with T0's write
                with s.lock_a:
                    pass

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1],
            invariant=lambda s: True,  # no data invariant — deadlock itself is the problem
            max_executions=500,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()
        assert result.counterexample is not None
        # Deadlock reproduction should be deterministic: 100% reproduction rate
        assert result.reproduction_attempts == 10, (
            f"Expected 10 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_no_deadlock_is_not_reported(self) -> None:
        """Consistent lock ordering does not cause a false deadlock report."""

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.x = 0

        def thread0(s: State) -> None:
            with s.lock_a:
                s.x += 1
                with s.lock_b:
                    pass

        def thread1(s: State) -> None:
            with s.lock_a:
                s.x += 1
                with s.lock_b:
                    pass

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1],
            invariant=lambda s: True,
            max_executions=100,
            detect_io=False,
            deadlock_timeout=2.0,
        )

        assert result.property_holds, "Consistent lock order should not be reported as deadlock"

    def test_three_thread_directed_cycle_deadlock(self) -> None:
        """Three-thread deadlock: T0→lock_a→lock_b, T1→lock_b→lock_c, T2→lock_c→lock_a.

        Forms the directed cycle T0→T1→T2→T0 when each holds its first
        lock and waits for its second.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.lock_c = threading.Lock()
                self.x = 0

        def thread0(s: State) -> None:
            with s.lock_a:
                s.x += 1
                with s.lock_b:
                    pass

        def thread1(s: State) -> None:
            with s.lock_b:
                s.x += 1
                with s.lock_c:
                    pass

        def thread2(s: State) -> None:
            with s.lock_c:
                s.x += 1
                with s.lock_a:
                    pass

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1, thread2],
            invariant=lambda s: True,
            max_executions=1000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "3-way deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()
        # Deadlock reproduction should be deterministic: 100% reproduction rate
        assert result.reproduction_attempts == 10, (
            f"Expected 10 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_partial_deadlock_third_thread_completes(self) -> None:
        """Three threads: two deadlock (lock-order inversion), third does independent work.

        The partial deadlock should still be detected even though T2 finishes.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.x = 0
                self.t2_done = False

        def thread0(s: State) -> None:
            with s.lock_a:
                s.x += 1
                with s.lock_b:
                    pass

        def thread1(s: State) -> None:
            with s.lock_b:
                s.x += 1
                with s.lock_a:
                    pass

        def thread2(s: State) -> None:
            s.t2_done = True

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1, thread2],
            invariant=lambda s: True,
            max_executions=1000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Partial deadlock should still be detected"
        # Deadlock reproduction should be deterministic: 100% reproduction rate
        assert result.reproduction_attempts == 10, (
            f"Expected 10 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_data_dependent_lock_order_deadlock(self) -> None:
        """Lock acquisition order depends on runtime state.

        T0 always acquires lock_a then lock_b.  T1 reads a shared flag
        (set by T0) that determines lock order.  Only the interleaving
        where T0 sets the flag before T1 reads it triggers the deadlock.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.reverse_order = False
                self.x = 0

        def thread0(s: State) -> None:
            s.reverse_order = True
            s.x += 1  # write conflict
            with s.lock_a:
                s.x += 1
                with s.lock_b:
                    pass

        def thread1(s: State) -> None:
            s.x += 1  # write conflict
            if s.reverse_order:
                with s.lock_b:
                    s.x += 1
                    with s.lock_a:
                        pass
            else:
                with s.lock_a:
                    s.x += 1
                    with s.lock_b:
                        pass

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1],
            invariant=lambda s: True,
            max_executions=500,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Data-dependent deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()
        # Deadlock reproduction should be deterministic: 100% reproduction rate
        assert result.reproduction_attempts == 10, (
            f"Expected 10 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_race_only_triggers_when_second_thread_first(self) -> None:
        """A race condition that ONLY manifests if thread 1 executes before thread 0.

        Thread 0 writes x=1, thread 1 writes x=2.  The invariant ``x != 1``
        only fails when thread 0 writes *last*, which requires thread 1 to
        execute first.  DPOR must explore both orderings to find this.
        """

        class State:
            def __init__(self) -> None:
                self.x = 0

        def thread0(s: State) -> None:
            s.x = 1

        def thread1(s: State) -> None:
            s.x = 2

        result = frontrun.explore(
            setup=State,
            workers=[thread0, thread1],
            invariant=lambda s: s.x != 1,
            max_executions=100,
            preemption_bound=2,
            detect_io=False,
            stop_on_first=False,
        )

        assert not result.property_holds, (
            "DPOR should find the invariant violation (x==1) that occurs when "
            "thread 1 executes first. This requires exploring an interleaving "
            f"where thread 1 goes before thread 0. num_explored={result.num_explored}"
        )
        # The failing schedule should have thread 1 first
        assert result.counterexample is not None
        assert result.counterexample[0] == 1, f"Counterexample should start with thread 1, got: {result.counterexample}"

    def test_dining_philosophers_three_deadlock_without_diversity(self) -> None:
        """Three dining philosophers — DPOR finds deadlock via race detection alone.

        Standard DPOR race detection discovers the deadlock without needing
        to artificially rotate the initial thread.  The initial lock acquires
        are on different objects (independent), but later acquire attempts
        create conflicts that DPOR explores.
        """

        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    with s.forks[right]:
                        pass

            return philosopher

        result = frontrun.explore(
            setup=State,
            workers=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=1000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=False,
        )

        assert not result.property_holds, "Dining philosophers deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_dining_philosophers_four_deadlock_without_diversity(self) -> None:
        """Four dining philosophers — DPOR finds deadlock via race detection alone.

        With scheduling coarsening removing spurious scheduling points, N=4
        pure locks terminates quickly: ~121 interleavings exhaustively in ~1s.
        """

        num_philosophers = 4

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    with s.forks[right]:
                        pass

            return philosopher

        result = frontrun.explore(
            setup=State,
            workers=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=5000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=False,
        )

        assert not result.property_holds, "Dining philosophers deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_dining_philosophers_three_with_shared_write(self) -> None:
        """Three dining philosophers with a shared write inside the critical section.

        The ``s.x += 1`` generates both LOAD_ATTR (read) and STORE_ATTR (write)
        at the opcode level, creating read-write conflicts between all threads.
        This significantly expands the DPOR search tree compared to pure lock
        operations, so we use N=3 to keep the exploration tractable.
        """

        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]
                self.x = 0

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    s.x += 1
                    with s.forks[right]:
                        pass

            return philosopher

        result = frontrun.explore(
            setup=State,
            workers=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=50000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Dining philosophers deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Stable object ID tests
# ---------------------------------------------------------------------------


class TestStableObjectIds:
    """Object keys must be stable across executions within one explore() call.

    When explore() creates fresh state via setup() each execution, id(obj)
    changes because Python allocates new objects at different addresses.  The
    StableObjectIds class assigns monotonically increasing IDs so that the same
    logical object (accessed in the same deterministic order during replay)
    gets the same key across executions.
    """

    def test_stable_ids_deterministic_across_resets(self) -> None:
        """After reset_for_execution, re-accessing objects in the same order
        produces the same stable IDs."""
        from frontrun.dpor import StableObjectIds

        ids = StableObjectIds()

        # First execution: create objects and get stable IDs
        obj_a = [1, 2, 3]
        obj_b = {"key": "value"}
        id_a1 = ids.get(obj_a)
        id_b1 = ids.get(obj_b)

        ids.reset_for_execution()

        # Second execution: NEW objects (different id()), same access order
        obj_a2 = [4, 5, 6]
        obj_b2 = {"other": "dict"}
        id_a2 = ids.get(obj_a2)
        id_b2 = ids.get(obj_b2)

        assert id_a1 == id_a2, f"First-accessed object should get same stable ID: {id_a1} != {id_a2}"
        assert id_b1 == id_b2, f"Second-accessed object should get same stable ID: {id_b1} != {id_b2}"

    def test_stable_ids_monotonically_increasing(self) -> None:
        """Stable IDs are assigned in first-access order, starting from 0."""
        from frontrun.dpor import StableObjectIds

        ids = StableObjectIds()
        obj1 = object()
        obj2 = object()
        obj3 = object()

        assert ids.get(obj1) == 0
        assert ids.get(obj2) == 1
        assert ids.get(obj3) == 2
        # Same object returns same ID
        assert ids.get(obj1) == 0

    def test_stable_ids_same_object_same_id(self) -> None:
        """Within one execution, the same object always returns the same ID."""
        from frontrun.dpor import StableObjectIds

        ids = StableObjectIds()
        obj = object()
        first = ids.get(obj)
        assert ids.get(obj) == first
        assert ids.get(obj) == first

    def test_make_object_key_uses_stable_ids(self) -> None:
        """_make_object_key should produce identical keys for the same logical
        object across executions when StableObjectIds is used."""
        from frontrun.dpor import StableObjectIds, _make_object_key

        ids = StableObjectIds()

        # Execution 1
        obj1 = {"shared": True}
        key1 = _make_object_key(ids.get(obj1), "value")

        ids.reset_for_execution()

        # Execution 2 — different physical object, same access order
        obj2 = {"shared": False}
        key2 = _make_object_key(ids.get(obj2), "value")

        assert key1 == key2, f"Object keys should be stable across executions: {key1} != {key2}"


# ---------------------------------------------------------------------------
# LOAD_SPECIAL attribute name mapping
# ---------------------------------------------------------------------------


class TestLoadSpecialNameMapping:
    """Verify that the LOAD_SPECIAL opcode maps all context manager args correctly.

    Bug: _opcode_observer._process_opcode only maps {0: "__enter__", 1: "__exit__"}
    for LOAD_SPECIAL but Python 3.14 uses args 2 and 3 for __aenter__ and __aexit__
    in async with statements.
    """

    def test_load_special_maps_aenter_aexit(self) -> None:
        """LOAD_SPECIAL args 2 and 3 should resolve to __aenter__/__aexit__.

        Bug: The _special_names dict in _process_opcode only maps
        {0: "__enter__", 1: "__exit__"}, missing {2: "__aenter__", 3: "__aexit__"}.
        This causes the fallback name "__special_2__" / "__special_3__" to be
        used instead of the real dunder names, which degrades shadow stack
        accuracy and makes access reports unrecognizable.

        We test this by examining the _process_opcode source code for the
        LOAD_SPECIAL handler's name mapping.
        """
        import sys

        if sys.version_info < (3, 14):
            pytest.skip("LOAD_SPECIAL only exists on Python 3.14+")

        import inspect

        from frontrun._opcode_observer import _process_opcode

        src = inspect.getsource(_process_opcode)
        # Find the LOAD_SPECIAL handler and verify it maps args 2 and 3
        assert "__aenter__" in src, "_process_opcode's LOAD_SPECIAL handler should map arg 2 to '__aenter__'"
        assert "__aexit__" in src, "_process_opcode's LOAD_SPECIAL handler should map arg 3 to '__aexit__'"

    def test_async_context_manager_dpor_finds_race(self) -> None:
        """DPOR should find races in async context managers using __aenter__/__aexit__.

        This exercises LOAD_SPECIAL with args 2/3. If the opcode observer doesn't
        map these correctly, the shadow stack will have incorrect entries and
        DPOR may miss shared-access detection for the context manager object.
        """
        import asyncio
        import sys

        if sys.version_info < (3, 14):
            pytest.skip("LOAD_SPECIAL only exists on Python 3.14+")

        class SharedState:
            def __init__(self) -> None:
                self.value = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Container:
            def __init__(self) -> None:
                self.state = SharedState()

        async def worker(c: Container) -> None:
            async with c.state as s:
                temp = s.value
                await asyncio.sleep(0)
                s.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Container,
                workers=worker,
                count=2,
                invariant=lambda c: c.state.value == 2,
                strategy="dpor",
            )
        )
        # The race on s.value should still be found even with async context manager
        assert not result.property_holds, "DPOR should find the lost-update race through async with"
