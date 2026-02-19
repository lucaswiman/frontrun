"""
Tests for the Python DPOR prototype.

Covers:
- VersionVec operations
- Access dependency tracking
- Path exploration tree
- DporEngine core operations
- High-level model checking (explore_dpor)
- Classic concurrency bugs: lost update, bank account transfer
- Cooperative primitives (SharedVar, CooperativeLock)
"""

from __future__ import annotations

import copy
import sys
import os

# Add the prototype directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from dpor_prototype import (
    AccessKind,
    Branch,
    CooperativeLock,
    DporEngine,
    ExplorationResult,
    LockAcquire,
    LockRelease,
    ObjectState,
    Path,
    SharedVar,
    Step,
    ThreadSpawn,
    ThreadStatus,
    VersionVec,
    explore_dpor,
)


# ---------------------------------------------------------------------------
# VersionVec tests
# ---------------------------------------------------------------------------

class TestVersionVec:
    def test_new_is_zero(self) -> None:
        vv = VersionVec(3)
        assert vv.get(0) == 0
        assert vv.get(1) == 0
        assert vv.get(2) == 0

    def test_increment(self) -> None:
        vv = VersionVec(3)
        vv.increment(1)
        assert vv.get(1) == 1
        vv.increment(1)
        assert vv.get(1) == 2
        assert vv.get(0) == 0

    def test_join(self) -> None:
        a = VersionVec(3)
        a.set(0, 2)
        a.set(1, 1)
        b = VersionVec(3)
        b.set(0, 1)
        b.set(1, 3)
        b.set(2, 2)
        a.join(b)
        assert a.get(0) == 2
        assert a.get(1) == 3
        assert a.get(2) == 2

    def test_partial_le_equal(self) -> None:
        a = VersionVec(3)
        assert a.partial_le(a)

    def test_partial_le_strictly_less(self) -> None:
        a = VersionVec(2)
        a.set(0, 1)
        a.set(1, 2)
        b = VersionVec(2)
        b.set(0, 2)
        b.set(1, 3)
        assert a.partial_le(b)
        assert not b.partial_le(a)

    def test_concurrent(self) -> None:
        a = VersionVec(2)
        a.set(0, 2)
        a.set(1, 1)
        b = VersionVec(2)
        b.set(0, 1)
        b.set(1, 2)
        assert a.concurrent_with(b)
        assert not a.partial_le(b)
        assert not b.partial_le(a)

    def test_not_concurrent_when_ordered(self) -> None:
        a = VersionVec(2)
        a.set(0, 1)
        a.set(1, 1)
        b = VersionVec(2)
        b.set(0, 2)
        b.set(1, 2)
        assert not a.concurrent_with(b)

    def test_clone(self) -> None:
        a = VersionVec(2)
        a.set(0, 5)
        b = a.clone()
        assert a == b
        b.increment(0)
        assert a.get(0) == 5
        assert b.get(0) == 6


# ---------------------------------------------------------------------------
# ObjectState tests
# ---------------------------------------------------------------------------

class TestObjectState:
    def test_read_depends_on_write(self) -> None:
        state = ObjectState()
        vv = VersionVec(2)
        from dpor_prototype import Access

        state.record_access(Access(0, vv.clone(), 0), AccessKind.WRITE)
        assert state.last_dependent_access(AccessKind.READ) is not None

    def test_read_does_not_depend_on_read(self) -> None:
        state = ObjectState()
        vv = VersionVec(2)
        from dpor_prototype import Access

        state.record_access(Access(0, vv.clone(), 0), AccessKind.READ)
        assert state.last_dependent_access(AccessKind.READ) is None

    def test_write_depends_on_any(self) -> None:
        state = ObjectState()
        vv = VersionVec(2)
        from dpor_prototype import Access

        state.record_access(Access(0, vv.clone(), 0), AccessKind.READ)
        assert state.last_dependent_access(AccessKind.WRITE) is not None


# ---------------------------------------------------------------------------
# Path tests
# ---------------------------------------------------------------------------

class TestPath:
    def test_schedule_first_branch(self) -> None:
        path = Path()
        chosen = path.schedule([0, 1], 0, 2)
        assert chosen == 0

    def test_schedule_prefers_current(self) -> None:
        path = Path()
        chosen = path.schedule([0, 1, 2], 1, 3)
        assert chosen == 1

    def test_backtrack_and_step(self) -> None:
        path = Path()
        path.schedule([0, 1], 0, 2)
        path.backtrack(0, 1)
        assert path.step()
        # Replay should now choose thread 1
        chosen = path.schedule([0, 1], 0, 2)
        assert chosen == 1

    def test_step_exhausted(self) -> None:
        path = Path()
        path.schedule([0, 1], 0, 2)
        assert not path.step()

    def test_full_exploration_two_threads(self) -> None:
        path = Path()
        executions = []

        chosen = path.schedule([0, 1], 0, 2)
        executions.append([chosen])
        path.backtrack(0, 1)

        assert path.step()
        chosen = path.schedule([0, 1], 0, 2)
        executions.append([chosen])

        assert not path.step()
        assert executions == [[0], [1]]


# ---------------------------------------------------------------------------
# DporEngine core tests
# ---------------------------------------------------------------------------

class TestDporEngine:
    def test_two_threads_no_conflict(self) -> None:
        engine = DporEngine(2)
        execution = engine.begin_execution()

        t0 = engine.schedule(execution)
        assert t0 == 0
        engine.process_access(execution, 0, 1, AccessKind.WRITE)
        execution.finish_thread(0)

        t1 = engine.schedule(execution)
        assert t1 == 1
        engine.process_access(execution, 1, 2, AccessKind.WRITE)
        execution.finish_thread(1)

        assert not engine.next_execution()
        assert engine.executions_completed == 1

    def test_two_threads_write_write_conflict(self) -> None:
        engine = DporEngine(2)
        exec_count = 0

        while True:
            execution = engine.begin_execution()
            first = engine.schedule(execution)
            engine.process_access(execution, first, 1, AccessKind.WRITE)
            execution.finish_thread(first)

            second = engine.schedule(execution)
            engine.process_access(execution, second, 1, AccessKind.WRITE)
            execution.finish_thread(second)

            exec_count += 1
            if not engine.next_execution():
                break

        assert exec_count == 2

    def test_read_read_no_conflict(self) -> None:
        engine = DporEngine(2)
        execution = engine.begin_execution()

        first = engine.schedule(execution)
        engine.process_access(execution, first, 1, AccessKind.READ)
        execution.finish_thread(first)

        second = engine.schedule(execution)
        engine.process_access(execution, second, 1, AccessKind.READ)
        execution.finish_thread(second)

        assert not engine.next_execution()

    def test_max_executions_limit(self) -> None:
        engine = DporEngine(2, max_executions=1)
        execution = engine.begin_execution()

        engine.schedule(execution)
        engine.process_access(execution, 0, 1, AccessKind.WRITE)
        execution.finish_thread(0)

        engine.schedule(execution)
        engine.process_access(execution, 1, 1, AccessKind.WRITE)
        execution.finish_thread(1)

        assert not engine.next_execution()
        assert engine.executions_completed == 1


# ---------------------------------------------------------------------------
# explore_dpor high-level API tests
# ---------------------------------------------------------------------------

class TestExploreDpor:
    def test_lost_update_bug(self) -> None:
        """Two threads doing read-modify-write on a shared counter.
        DPOR should detect the lost-update interleaving.
        """
        result = explore_dpor(
            setup=lambda: {"counter": 0, "local": [0, 0]},
            thread_steps=[
                [
                    Step(0, AccessKind.READ,
                         lambda s: s.update(local=[s["counter"], s["local"][1]])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("counter", s["local"][0] + 1)),
                ],
                [
                    Step(0, AccessKind.READ,
                         lambda s: s.update(local=[s["local"][0], s["counter"]])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("counter", s["local"][1] + 1)),
                ],
            ],
            invariant=lambda s: s["counter"] == 2,
            max_executions=500,
        )
        assert not result.all_passed
        assert len(result.failures) > 0

    def test_atomic_increment_no_bug(self) -> None:
        """Each thread does a single atomic increment. No bug possible."""
        result = explore_dpor(
            setup=lambda: [0],  # mutable counter in a list
            thread_steps=[
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
            ],
            invariant=lambda s: s[0] == 2,
        )
        assert result.all_passed

    def test_independent_threads(self) -> None:
        """Threads accessing different objects need only one execution."""
        result = explore_dpor(
            setup=lambda: {"a": 0, "b": 0},
            thread_steps=[
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__("a", 1))],
                [Step(1, AccessKind.WRITE, lambda s: s.__setitem__("b", 1))],
            ],
            invariant=lambda s: s["a"] == 1 and s["b"] == 1,
        )
        assert result.all_passed
        assert result.executions_explored == 1

    def test_bank_account_transfer(self) -> None:
        """Classic bank account transfer race.

        Two threads each transfer 50 from account A to B.
        Total should always be 200 but the race causes violations.
        """

        def make_state():
            return {"a": 100, "b": 100, "la": [0, 0], "lb": [0, 0]}

        result = explore_dpor(
            setup=make_state,
            thread_steps=[
                [
                    Step(0, AccessKind.READ,
                         lambda s: s.__setitem__("la", [s["a"], s["la"][1]])),
                    Step(1, AccessKind.READ,
                         lambda s: s.__setitem__("lb", [s["b"], s["lb"][1]])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("a", s["la"][0] - 50)),
                    Step(1, AccessKind.WRITE,
                         lambda s: s.__setitem__("b", s["lb"][0] + 50)),
                ],
                [
                    Step(0, AccessKind.READ,
                         lambda s: s.__setitem__("la", [s["la"][0], s["a"]])),
                    Step(1, AccessKind.READ,
                         lambda s: s.__setitem__("lb", [s["lb"][0], s["b"]])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("a", s["la"][1] - 50)),
                    Step(1, AccessKind.WRITE,
                         lambda s: s.__setitem__("b", s["lb"][1] + 50)),
                ],
            ],
            invariant=lambda s: s["a"] + s["b"] == 200,
            max_executions=500,
        )
        assert not result.all_passed

    def test_three_threads_counter(self) -> None:
        """Three threads incrementing a shared counter (read-modify-write).
        Should find that some interleavings produce counter < 3.
        """

        def make_state():
            return {"c": 0, "l": [0, 0, 0]}

        result = explore_dpor(
            setup=make_state,
            thread_steps=[
                [
                    Step(0, AccessKind.READ,
                         lambda s: s["l"].__setitem__(0, s["c"])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("c", s["l"][0] + 1)),
                ],
                [
                    Step(0, AccessKind.READ,
                         lambda s: s["l"].__setitem__(1, s["c"])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("c", s["l"][1] + 1)),
                ],
                [
                    Step(0, AccessKind.READ,
                         lambda s: s["l"].__setitem__(2, s["c"])),
                    Step(0, AccessKind.WRITE,
                         lambda s: s.__setitem__("c", s["l"][2] + 1)),
                ],
            ],
            invariant=lambda s: s["c"] == 3,
            max_executions=500,
        )
        assert not result.all_passed
        assert result.executions_explored >= 2

    def test_preemption_bound_zero(self) -> None:
        """With preemption_bound=0, only non-preemptive schedules are explored."""
        result = explore_dpor(
            setup=lambda: [0],
            thread_steps=[
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
            ],
            invariant=lambda s: s[0] == 2,
            preemption_bound=0,
        )
        # With bound=0, no preemptive interleavings explored
        assert result.all_passed


# ---------------------------------------------------------------------------
# Cooperative primitives tests
# ---------------------------------------------------------------------------

class TestSharedVar:
    def test_read_write(self) -> None:
        engine = DporEngine(2)
        execution = engine.begin_execution()
        engine.schedule(execution)

        var = SharedVar(42, object_id=0, engine=engine, execution=execution)
        val = var.read(thread_id=0)
        assert val == 42
        var.write(99, thread_id=0)
        assert var.value == 99

    def test_without_engine(self) -> None:
        """SharedVar works without an engine for plain usage."""
        var = SharedVar(10, object_id=0)
        assert var.read() == 10
        var.write(20)
        assert var.value == 20


class TestCooperativeLock:
    def test_acquire_release(self) -> None:
        engine = DporEngine(2)
        execution = engine.begin_execution()
        engine.schedule(execution)

        lock = CooperativeLock(lock_id=1, engine=engine, execution=execution)
        lock.acquire(thread_id=0)
        lock.release(thread_id=0)
        # Should not crash

    def test_without_engine(self) -> None:
        lock = CooperativeLock(lock_id=1)
        lock.acquire(thread_id=0)
        lock.release(thread_id=0)


# ---------------------------------------------------------------------------
# Integration: model-checking with cooperative primitives
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_counter_with_shared_var(self) -> None:
        """Use SharedVar in a model-checking loop to detect a lost update.

        Each thread does: local = counter.read(); counter.write(local + 1)
        We interleave at each operation using the DPOR scheduler.
        """
        engine = DporEngine(2, max_executions=100)
        found_bug = False

        # Define thread operations as (object_id, kind, apply) tuples
        # Thread 0: read counter into local[0], write local[0]+1 to counter
        # Thread 1: read counter into local[1], write local[1]+1 to counter

        while True:
            execution = engine.begin_execution()
            counter_val = [0]  # shared counter
            locals_ = [0, 0]   # per-thread locals
            thread_pcs = [0, 0]  # program counters
            # Each thread has 2 ops
            num_ops = [2, 2]

            while True:
                # Mark finished threads
                for i in range(2):
                    if thread_pcs[i] >= num_ops[i]:
                        execution.finish_thread(i)

                if not execution.runnable_threads():
                    break

                chosen = engine.schedule(execution)
                if chosen is None:
                    break

                pc = thread_pcs[chosen]
                if pc >= num_ops[chosen]:
                    break

                if pc == 0:
                    # Read: report read access, store value locally
                    engine.process_access(execution, chosen, 0, AccessKind.READ)
                    locals_[chosen] = counter_val[0]
                elif pc == 1:
                    # Write: report write access, update counter
                    engine.process_access(execution, chosen, 0, AccessKind.WRITE)
                    counter_val[0] = locals_[chosen] + 1

                thread_pcs[chosen] += 1

            if counter_val[0] != 2:
                found_bug = True

            if not engine.next_execution():
                break

        assert found_bug, "DPOR should detect the lost-update bug"

    def test_counter_with_lock_no_bug(self) -> None:
        """Counter protected by a lock should have no bugs.

        Note: In this simplified model, we manually sequence the operations
        rather than having real blocking. The lock's happens-before edges
        are what matters for DPOR to reduce exploration.
        """
        engine = DporEngine(2, max_executions=100)
        all_correct = True

        while True:
            execution = engine.begin_execution()
            counter = SharedVar(0, object_id=0, engine=engine, execution=execution)
            lock = CooperativeLock(lock_id=99, engine=engine, execution=execution)

            # Thread 0: acquire, increment, release
            engine.schedule(execution)
            lock.acquire(0)
            val0 = counter.read(thread_id=0)
            counter.write(val0 + 1, thread_id=0)
            lock.release(0)
            execution.finish_thread(0)

            # Thread 1: acquire, increment, release
            engine.schedule(execution)
            lock.acquire(1)
            val1 = counter.read(thread_id=1)
            counter.write(val1 + 1, thread_id=1)
            lock.release(1)
            execution.finish_thread(1)

            if counter.value != 2:
                all_correct = False

            if not engine.next_execution():
                break

        assert all_correct, "Lock-protected counter should always be correct"


# ---------------------------------------------------------------------------
# Edge cases and stress tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_thread(self) -> None:
        """Single thread: only one execution, always passes."""
        result = explore_dpor(
            setup=lambda: [0],
            thread_steps=[
                [
                    Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, 1)),
                    Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, 2)),
                ],
            ],
            invariant=lambda s: s[0] == 2,
        )
        assert result.all_passed
        assert result.executions_explored == 1

    def test_empty_threads(self) -> None:
        """Threads with no operations: trivially passes."""
        result = explore_dpor(
            setup=lambda: None,
            thread_steps=[[], []],
            invariant=lambda s: True,
        )
        assert result.all_passed

    def test_one_thread_many_ops(self) -> None:
        """One thread with many operations: single execution."""
        steps = [
            Step(0, AccessKind.WRITE, lambda s, i=i: s.__setitem__(0, i))
            for i in range(20)
        ]
        result = explore_dpor(
            setup=lambda: [0],
            thread_steps=[steps],
            invariant=lambda s: s[0] == 19,
        )
        assert result.all_passed
        assert result.executions_explored == 1

    def test_many_independent_threads(self) -> None:
        """Four threads on independent objects: only one execution."""
        result = explore_dpor(
            setup=lambda: {"a": 0, "b": 0, "c": 0, "d": 0},
            thread_steps=[
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__("a", 1))],
                [Step(1, AccessKind.WRITE, lambda s: s.__setitem__("b", 1))],
                [Step(2, AccessKind.WRITE, lambda s: s.__setitem__("c", 1))],
                [Step(3, AccessKind.WRITE, lambda s: s.__setitem__("d", 1))],
            ],
            invariant=lambda s: all(v == 1 for v in s.values()),
        )
        assert result.all_passed
        assert result.executions_explored == 1

    def test_exploration_reports_failure_schedule(self) -> None:
        """Failures should include the schedule trace for debugging."""
        result = explore_dpor(
            setup=lambda: [0],
            thread_steps=[
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
                [Step(0, AccessKind.WRITE, lambda s: s.__setitem__(0, s[0] + 1))],
            ],
            invariant=lambda s: False,  # Always fails
        )
        assert not result.all_passed
        for exec_num, schedule in result.failures:
            assert isinstance(exec_num, int)
            assert isinstance(schedule, list)
            assert all(isinstance(t, int) for t in schedule)


if __name__ == "__main__":
    # Run with pytest or directly
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=os.path.dirname(__file__),
    )
    sys.exit(result.returncode)
