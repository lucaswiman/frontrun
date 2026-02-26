"""Tests for safe concurrent patterns — DPOR and bytecode fuzzing should NOT report false positives.

These tests exercise code patterns that ARE thread-safe and verify that both
``explore_dpor()`` and ``explore_interleavings()`` correctly identify them as
safe (``result.property_holds is True``).  Any test failure here represents a
false positive in the concurrency testing infrastructure.

Categories:
1. Lock-based safe patterns (Lock, RLock, Semaphore, try/finally)
2. Truly independent operations (no sharing possible)
3. Tricky patterns that look racy but are properly synchronized
4. Stress tests for the cooperative lock / scheduler infrastructure
5. Edge cases that could confuse the shadow stack
"""

from __future__ import annotations

import math
import queue
import threading

import pytest

from frontrun.bytecode import explore_interleavings
from frontrun.dpor import explore_dpor

# ============================================================================
# Category 1: Lock-based safe patterns
# ============================================================================


# -- 1. Lock-protected counter -----------------------------------------------


class _LockProtectedState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLockProtectedCounter:
    """Lock-protected counter should never report a race."""

    def test_dpor_safe(self) -> None:
        def inc(state: _LockProtectedState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_dpor(
            setup=_LockProtectedState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on lock-protected counter: {result}"

    def test_bytecode_safe(self) -> None:
        def inc(state: _LockProtectedState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_interleavings(
            setup=_LockProtectedState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on lock-protected counter: {result}"


# -- 2. RLock reentrant acquisition ------------------------------------------


class _RLockReentrantState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.RLock()


class TestRLockReentrant:
    """RLock acquired multiple times by the same thread in nested calls."""

    def test_dpor_safe(self) -> None:
        def outer(state: _RLockReentrantState) -> None:
            with state.lock:
                inner(state)

        def inner(state: _RLockReentrantState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_dpor(
            setup=_RLockReentrantState,
            threads=[outer, outer],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on RLock reentrant: {result}"

    def test_bytecode_safe(self) -> None:
        def outer(state: _RLockReentrantState) -> None:
            with state.lock:
                with state.lock:
                    temp = state.value
                    state.value = temp + 1

        result = explore_interleavings(
            setup=_RLockReentrantState,
            threads=[outer, outer],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on RLock reentrant: {result}"


# -- 3. Semaphore-protected access -------------------------------------------


class _SemaphoreState:
    def __init__(self) -> None:
        self.value = 0
        self.sem = threading.Semaphore(1)


class TestSemaphoreProtected:
    """Binary semaphore guarding a critical section."""

    def test_dpor_safe(self) -> None:
        def inc(state: _SemaphoreState) -> None:
            state.sem.acquire()
            try:
                temp = state.value
                state.value = temp + 1
            finally:
                state.sem.release()

        result = explore_dpor(
            setup=_SemaphoreState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on semaphore-protected: {result}"

    def test_bytecode_safe(self) -> None:
        def inc(state: _SemaphoreState) -> None:
            state.sem.acquire()
            try:
                temp = state.value
                state.value = temp + 1
            finally:
                state.sem.release()

        result = explore_interleavings(
            setup=_SemaphoreState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on semaphore-protected: {result}"


# -- 4. Lock with complex control flow (try/finally + exception) -------------


class _TryFinallyState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()
        self.error_count = 0


class TestLockTryFinally:
    """Lock acquired in try/finally with exceptions raised in between."""

    def test_dpor_safe(self) -> None:
        def inc_with_exception(state: _TryFinallyState) -> None:
            state.lock.acquire()
            try:
                temp = state.value
                # Exception that gets caught inside critical section
                try:
                    _ = 1 / 1  # no actual error
                except ZeroDivisionError:
                    state.error_count += 1
                state.value = temp + 1
            finally:
                state.lock.release()

        result = explore_dpor(
            setup=_TryFinallyState,
            threads=[inc_with_exception, inc_with_exception],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on try/finally lock: {result}"

    def test_bytecode_safe(self) -> None:
        def inc_with_exception(state: _TryFinallyState) -> None:
            with state.lock:
                temp = state.value
                try:
                    _ = 1 / 1
                except ZeroDivisionError:
                    state.error_count += 1
                state.value = temp + 1

        result = explore_interleavings(
            setup=_TryFinallyState,
            threads=[inc_with_exception, inc_with_exception],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on try/finally lock: {result}"


# -- 5. Multiple locks in correct ordering -----------------------------------


class _TwoLocksState:
    def __init__(self) -> None:
        self.a = 0
        self.b = 0
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()


class TestMultipleLocksCorrectOrder:
    """Two locks always acquired in the same order — no deadlock possible."""

    def test_dpor_safe(self) -> None:
        def transfer(state: _TwoLocksState) -> None:
            with state.lock_a:
                with state.lock_b:
                    state.a += 1
                    state.b += 1

        result = explore_dpor(
            setup=_TwoLocksState,
            threads=[transfer, transfer],
            invariant=lambda s: s.a == 2 and s.b == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on ordered locks: {result}"

    def test_bytecode_safe(self) -> None:
        def transfer(state: _TwoLocksState) -> None:
            with state.lock_a:
                with state.lock_b:
                    state.a += 1
                    state.b += 1

        result = explore_interleavings(
            setup=_TwoLocksState,
            threads=[transfer, transfer],
            invariant=lambda s: s.a == 2 and s.b == 2,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on ordered locks: {result}"


# -- 6. Lock-protected closure (nonlocal variable) ---------------------------


class _ClosureLockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        count = 0

        def get() -> int:
            return count

        def inc() -> None:
            nonlocal count
            with self.lock:
                temp = count
                count = temp + 1

        self.get = get
        self.inc = inc


class TestLockProtectedClosure:
    """nonlocal variable protected by lock in closure."""

    def test_dpor_safe(self) -> None:
        result = explore_dpor(
            setup=_ClosureLockState,
            threads=[lambda s: s.inc(), lambda s: s.inc()],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on lock-protected closure: {result}"

    def test_bytecode_safe(self) -> None:
        result = explore_interleavings(
            setup=_ClosureLockState,
            threads=[lambda s: s.inc(), lambda s: s.inc()],
            invariant=lambda s: s.get() == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on lock-protected closure: {result}"


# ============================================================================
# Category 2: Truly independent operations (no conflict possible)
# ============================================================================


# -- 7. Thread-local storage (each writes own index) -------------------------


class _ThreadLocalState:
    def __init__(self) -> None:
        self.slots = [0, 0]


class TestThreadLocalStorage:
    """Each thread writes to its own index in a list — no sharing."""

    def test_dpor_safe(self) -> None:
        def write_0(state: _ThreadLocalState) -> None:
            state.slots[0] = 42

        def write_1(state: _ThreadLocalState) -> None:
            state.slots[1] = 99

        result = explore_dpor(
            setup=_ThreadLocalState,
            threads=[write_0, write_1],
            invariant=lambda s: s.slots[0] == 42 and s.slots[1] == 99,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on thread-local storage: {result}"

    def test_bytecode_safe(self) -> None:
        def write_0(state: _ThreadLocalState) -> None:
            state.slots[0] = 42

        def write_1(state: _ThreadLocalState) -> None:
            state.slots[1] = 99

        result = explore_interleavings(
            setup=_ThreadLocalState,
            threads=[write_0, write_1],
            invariant=lambda s: s.slots[0] == 42 and s.slots[1] == 99,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on thread-local storage: {result}"


# -- 8. Independent attributes -----------------------------------------------


class _IndependentAttrsState:
    def __init__(self) -> None:
        self.a = 0
        self.b = 0


class TestIndependentAttributes:
    """Thread 1 writes state.a, Thread 2 writes state.b — no conflict."""

    def test_dpor_safe(self) -> None:
        result = explore_dpor(
            setup=_IndependentAttrsState,
            threads=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 1),
            ],
            invariant=lambda s: s.a == 1 and s.b == 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on independent attrs: {result}"

    def test_bytecode_safe(self) -> None:
        result = explore_interleavings(
            setup=_IndependentAttrsState,
            threads=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 1),
            ],
            invariant=lambda s: s.a == 1 and s.b == 1,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on independent attrs: {result}"


# -- 9. Independent dicts (each thread has its own) --------------------------


class _IndependentDictsState:
    def __init__(self) -> None:
        self.dict_a: dict[str, int] = {}
        self.dict_b: dict[str, int] = {}


class TestIndependentDicts:
    """Each thread operates on its own dict — no sharing."""

    def test_dpor_safe(self) -> None:
        def fill_a(state: _IndependentDictsState) -> None:
            state.dict_a["x"] = 1
            state.dict_a["y"] = 2

        def fill_b(state: _IndependentDictsState) -> None:
            state.dict_b["x"] = 10
            state.dict_b["y"] = 20

        result = explore_dpor(
            setup=_IndependentDictsState,
            threads=[fill_a, fill_b],
            invariant=lambda s: s.dict_a.get("x") == 1 and s.dict_b.get("x") == 10,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on independent dicts: {result}"

    def test_bytecode_safe(self) -> None:
        def fill_a(state: _IndependentDictsState) -> None:
            state.dict_a["x"] = 1
            state.dict_a["y"] = 2

        def fill_b(state: _IndependentDictsState) -> None:
            state.dict_b["x"] = 10
            state.dict_b["y"] = 20

        result = explore_interleavings(
            setup=_IndependentDictsState,
            threads=[fill_a, fill_b],
            invariant=lambda s: s.dict_a.get("x") == 1 and s.dict_b.get("x") == 10,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on independent dicts: {result}"


# -- 10. Independent closures ------------------------------------------------


class _IndependentClosuresState:
    def __init__(self) -> None:
        self.result_a = 0
        self.result_b = 0


class TestIndependentClosures:
    """Each thread operates on its own closure — no sharing of nonlocal state."""

    def test_dpor_safe(self) -> None:
        def work_a(state: _IndependentClosuresState) -> None:
            accum = 0
            for i in range(3):
                accum += i
            state.result_a = accum

        def work_b(state: _IndependentClosuresState) -> None:
            accum = 0
            for i in range(4):
                accum += i
            state.result_b = accum

        result = explore_dpor(
            setup=_IndependentClosuresState,
            threads=[work_a, work_b],
            invariant=lambda s: s.result_a == 3 and s.result_b == 6,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on independent closures: {result}"


# -- 11. Immutable type operations -------------------------------------------


class _ImmutableOpsState:
    def __init__(self) -> None:
        self.result_a = ""
        self.result_b: tuple[int, ...] = ()


class TestImmutableTypeOperations:
    """Threads operating on immutable types (strings, tuples) — no mutation possible."""

    def test_dpor_safe(self) -> None:
        def string_work(state: _ImmutableOpsState) -> None:
            s = "hello"
            s = s + " world"
            s = s.upper()
            state.result_a = s

        def tuple_work(state: _ImmutableOpsState) -> None:
            t = (1, 2, 3)
            t = t + (4, 5)
            state.result_b = t

        result = explore_dpor(
            setup=_ImmutableOpsState,
            threads=[string_work, tuple_work],
            invariant=lambda s: s.result_a == "HELLO WORLD" and s.result_b == (1, 2, 3, 4, 5),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on immutable ops: {result}"

    def test_bytecode_safe(self) -> None:
        def string_work(state: _ImmutableOpsState) -> None:
            s = "hello"
            s = s + " world"
            state.result_a = s

        def tuple_work(state: _ImmutableOpsState) -> None:
            t = (1, 2)
            t = t + (3,)
            state.result_b = t

        result = explore_interleavings(
            setup=_ImmutableOpsState,
            threads=[string_work, tuple_work],
            invariant=lambda s: s.result_a == "hello world" and s.result_b == (1, 2, 3),
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on immutable ops: {result}"


# ============================================================================
# Category 3: Tricky patterns that look racy but aren't
# ============================================================================


# -- 12. Happens-before through Event ----------------------------------------


class _EventSyncState:
    def __init__(self) -> None:
        self.value = 0
        self.event = threading.Event()


class TestHappensBeforeEvent:
    """Thread 1 sets event after write, Thread 2 waits for event before read.

    BUG: The DPOR test deadlocks on free-threaded Python because cooperative
    Event patching interferes with Thread._started.wait() during thread
    startup.  The cooperative Event's internal Condition/Lock are patched
    versions, causing the main thread to block in Thread.start().
    """

    @pytest.mark.timeout(10)
    def test_dpor_safe(self) -> None:
        def writer(state: _EventSyncState) -> None:
            state.value = 42
            state.event.set()

        def reader(state: _EventSyncState) -> None:
            state.event.wait()
            _ = state.value

        result = explore_dpor(
            setup=_EventSyncState,
            threads=[writer, reader],
            invariant=lambda s: s.value == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on event sync: {result}"

    def test_bytecode_safe(self) -> None:
        def writer(state: _EventSyncState) -> None:
            state.value = 42
            state.event.set()

        def reader(state: _EventSyncState) -> None:
            state.event.wait()
            _ = state.value

        result = explore_interleavings(
            setup=_EventSyncState,
            threads=[writer, reader],
            invariant=lambda s: s.value == 42,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on event sync: {result}"


# -- 13. Queue-based communication -------------------------------------------


class _QueueCommState:
    def __init__(self) -> None:
        self.q: queue.Queue[int] = queue.Queue()
        self.received = 0


class TestQueueCommunication:
    """Thread 1 puts to queue, Thread 2 gets — linearized by queue."""

    def test_dpor_safe(self) -> None:
        def producer(state: _QueueCommState) -> None:
            state.q.put(42)

        def consumer(state: _QueueCommState) -> None:
            state.received = state.q.get()

        result = explore_dpor(
            setup=_QueueCommState,
            threads=[producer, consumer],
            invariant=lambda s: s.received == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on queue comm: {result}"

    def test_bytecode_safe(self) -> None:
        def producer(state: _QueueCommState) -> None:
            state.q.put(42)

        def consumer(state: _QueueCommState) -> None:
            state.received = state.q.get()

        result = explore_interleavings(
            setup=_QueueCommState,
            threads=[producer, consumer],
            invariant=lambda s: s.received == 42,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on queue comm: {result}"


# -- 14. Lock handoff pattern ------------------------------------------------


class _LockHandoffState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLockHandoff:
    """Thread 1 writes under lock, Thread 2 reads under same lock — serialized."""

    def test_dpor_safe(self) -> None:
        def writer(state: _LockHandoffState) -> None:
            with state.lock:
                state.value = 42

        def reader(state: _LockHandoffState) -> None:
            with state.lock:
                _ = state.value

        result = explore_dpor(
            setup=_LockHandoffState,
            threads=[writer, reader],
            invariant=lambda s: s.value == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on lock handoff: {result}"

    def test_bytecode_safe(self) -> None:
        def writer(state: _LockHandoffState) -> None:
            with state.lock:
                state.value = 42

        def reader(state: _LockHandoffState) -> None:
            with state.lock:
                _ = state.value

        result = explore_interleavings(
            setup=_LockHandoffState,
            threads=[writer, reader],
            invariant=lambda s: s.value == 42,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on lock handoff: {result}"


# -- 15. Condition variable signaling ----------------------------------------


class _CondVarState:
    def __init__(self) -> None:
        self.value = 0
        self.ready = False
        self.cond = threading.Condition()


class TestConditionVariableSignaling:
    """Producer writes state under lock, signals condition; consumer waits and reads."""

    def test_dpor_safe(self) -> None:
        def producer(state: _CondVarState) -> None:
            with state.cond:
                state.value = 42
                state.ready = True
                state.cond.notify()

        def consumer(state: _CondVarState) -> None:
            with state.cond:
                while not state.ready:
                    state.cond.wait()
                _ = state.value

        result = explore_dpor(
            setup=_CondVarState,
            threads=[producer, consumer],
            invariant=lambda s: s.value == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on condvar: {result}"

    def test_bytecode_safe(self) -> None:
        def producer(state: _CondVarState) -> None:
            with state.cond:
                state.value = 42
                state.ready = True
                state.cond.notify()

        def consumer(state: _CondVarState) -> None:
            with state.cond:
                while not state.ready:
                    state.cond.wait()
                _ = state.value

        result = explore_interleavings(
            setup=_CondVarState,
            threads=[producer, consumer],
            invariant=lambda s: s.value == 42,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on condvar: {result}"


# ============================================================================
# Category 4: Stress tests for the infrastructure itself
# ============================================================================


# -- 16. Many nested locks ---------------------------------------------------


class _NestedLocksState:
    def __init__(self) -> None:
        self.value = 0
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()
        self.lock3 = threading.Lock()


class TestManyNestedLocks:
    """Deeply nested lock acquisition — tests cooperative lock stack depth."""

    def test_dpor_safe(self) -> None:
        def deep_inc(state: _NestedLocksState) -> None:
            with state.lock1:
                with state.lock2:
                    with state.lock3:
                        temp = state.value
                        state.value = temp + 1

        result = explore_dpor(
            setup=_NestedLocksState,
            threads=[deep_inc, deep_inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on nested locks: {result}"

    def test_bytecode_safe(self) -> None:
        def deep_inc(state: _NestedLocksState) -> None:
            with state.lock1:
                with state.lock2:
                    with state.lock3:
                        temp = state.value
                        state.value = temp + 1

        result = explore_interleavings(
            setup=_NestedLocksState,
            threads=[deep_inc, deep_inc],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=400,
            seed=42,
        )
        assert result.property_holds, f"False positive on nested locks: {result}"


# -- 17. Lock acquire/release in loop ----------------------------------------


class _LockLoopState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLockAcquireReleaseLoop:
    """Rapid lock cycling — tests spin-yield performance."""

    def test_dpor_safe(self) -> None:
        def loop_inc(state: _LockLoopState) -> None:
            for _ in range(3):
                with state.lock:
                    state.value += 1

        result = explore_dpor(
            setup=_LockLoopState,
            threads=[loop_inc, loop_inc],
            invariant=lambda s: s.value == 6,
            detect_io=False,
            deadlock_timeout=5.0,
            max_executions=100,
        )
        assert result.property_holds, f"False positive on lock loop: {result}"

    def test_bytecode_safe(self) -> None:
        def loop_inc(state: _LockLoopState) -> None:
            for _ in range(3):
                with state.lock:
                    state.value += 1

        result = explore_interleavings(
            setup=_LockLoopState,
            threads=[loop_inc, loop_inc],
            invariant=lambda s: s.value == 6,
            max_attempts=50,
            max_ops=500,
            seed=42,
        )
        assert result.property_holds, f"False positive on lock loop: {result}"


# -- 18. Many threads independent -------------------------------------------


class _ManyThreadsState:
    def __init__(self) -> None:
        self.a = 0
        self.b = 0
        self.c = 0


class TestManyThreadsIndependent:
    """Three threads all writing to independent state."""

    def test_dpor_safe(self) -> None:
        result = explore_dpor(
            setup=_ManyThreadsState,
            threads=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 2),
                lambda s: setattr(s, "c", 3),
            ],
            invariant=lambda s: s.a == 1 and s.b == 2 and s.c == 3,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on many independent threads: {result}"

    def test_bytecode_safe(self) -> None:
        result = explore_interleavings(
            setup=_ManyThreadsState,
            threads=[
                lambda s: setattr(s, "a", 1),
                lambda s: setattr(s, "b", 2),
                lambda s: setattr(s, "c", 3),
            ],
            invariant=lambda s: s.a == 1 and s.b == 2 and s.c == 3,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on many independent threads: {result}"


# -- 19. Exception in thread with lock (tests cleanup) ----------------------


class _ExceptionInLockState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()
        self.error_handled = False


class TestExceptionInThreadWithLock:
    """Thread raises and catches exception while holding lock — tests lock cleanup."""

    def test_dpor_safe(self) -> None:
        def safe_inc_with_error(state: _ExceptionInLockState) -> None:
            with state.lock:
                try:
                    temp = state.value
                    raise ValueError("test")
                except ValueError:
                    state.error_handled = True
                    state.value = temp + 1

        def normal_inc(state: _ExceptionInLockState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_dpor(
            setup=_ExceptionInLockState,
            threads=[safe_inc_with_error, normal_inc],
            invariant=lambda s: s.value == 2 and s.error_handled,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on exception in lock: {result}"

    def test_bytecode_safe(self) -> None:
        def safe_inc_with_error(state: _ExceptionInLockState) -> None:
            with state.lock:
                try:
                    temp = state.value
                    raise ValueError("test")
                except ValueError:
                    state.error_handled = True
                    state.value = temp + 1

        def normal_inc(state: _ExceptionInLockState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_interleavings(
            setup=_ExceptionInLockState,
            threads=[safe_inc_with_error, normal_inc],
            invariant=lambda s: s.value == 2 and s.error_handled,
            max_attempts=50,
            max_ops=300,
            seed=42,
        )
        assert result.property_holds, f"False positive on exception in lock: {result}"


# -- 20. Large computation between lock ops ----------------------------------


class _HeavyComputeState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLargeComputationBetweenLockOps:
    """Heavy CPU work between lock acquire and release."""

    def test_dpor_safe(self) -> None:
        def compute_and_store(state: _HeavyComputeState) -> None:
            with state.lock:
                # Non-trivial local computation
                total = 0
                for i in range(10):
                    total += i * i
                state.value += total

        result = explore_dpor(
            setup=_HeavyComputeState,
            threads=[compute_and_store, compute_and_store],
            invariant=lambda s: s.value == 570,  # 2 * sum(i*i for i in range(10)) = 2 * 285
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on heavy compute: {result}"


# -- 21. Mixed sync primitives (Lock + Event + Queue) -----------------------


class _MixedSyncState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.q: queue.Queue[int] = queue.Queue()
        self.value = 0


class TestMixedSyncPrimitives:
    """Lock + Event + Queue all used together correctly."""

    def test_dpor_safe(self) -> None:
        def producer(state: _MixedSyncState) -> None:
            with state.lock:
                state.value = 42
            state.q.put(state.value)
            state.event.set()

        def consumer(state: _MixedSyncState) -> None:
            state.event.wait()
            val = state.q.get()
            with state.lock:
                _ = val

        result = explore_dpor(
            setup=_MixedSyncState,
            threads=[producer, consumer],
            invariant=lambda s: s.value == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on mixed sync: {result}"

    def test_bytecode_safe(self) -> None:
        def producer(state: _MixedSyncState) -> None:
            with state.lock:
                state.value = 42
            state.q.put(state.value)
            state.event.set()

        def consumer(state: _MixedSyncState) -> None:
            state.event.wait()
            val = state.q.get()
            with state.lock:
                _ = val

        result = explore_interleavings(
            setup=_MixedSyncState,
            threads=[producer, consumer],
            invariant=lambda s: s.value == 42,
            max_attempts=50,
            max_ops=400,
            seed=42,
        )
        assert result.property_holds, f"False positive on mixed sync: {result}"


# ============================================================================
# Category 5: Edge cases that could confuse the shadow stack
# ============================================================================


# -- 22. Calling pure functions (no shared state) ----------------------------


class _PureFunctionState:
    def __init__(self) -> None:
        self.result_a = 0.0
        self.result_b = 0.0


class TestCallingPureFunctions:
    """Threads call pure functions (math.sqrt, etc.) — no shared state modified."""

    def test_dpor_safe(self) -> None:
        def compute_a(state: _PureFunctionState) -> None:
            val = math.sqrt(144.0)
            val = math.floor(val)
            state.result_a = float(val)

        def compute_b(state: _PureFunctionState) -> None:
            val = math.sqrt(256.0)
            val = math.ceil(val)
            state.result_b = float(val)

        result = explore_dpor(
            setup=_PureFunctionState,
            threads=[compute_a, compute_b],
            invariant=lambda s: s.result_a == 12.0 and s.result_b == 16.0,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on pure functions: {result}"

    def test_bytecode_safe(self) -> None:
        def compute_a(state: _PureFunctionState) -> None:
            val = math.sqrt(144.0)
            state.result_a = val

        def compute_b(state: _PureFunctionState) -> None:
            val = math.sqrt(256.0)
            state.result_b = val

        result = explore_interleavings(
            setup=_PureFunctionState,
            threads=[compute_a, compute_b],
            invariant=lambda s: s.result_a == 12.0 and s.result_b == 16.0,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on pure functions: {result}"


# -- 23. Creating new objects per thread (no sharing) -------------------------


class _FreshObjectsState:
    def __init__(self) -> None:
        self.len_a = 0
        self.len_b = 0


class TestCreatingFreshObjects:
    """Each thread creates fresh objects, no sharing."""

    def test_dpor_safe(self) -> None:
        def create_list(state: _FreshObjectsState) -> None:
            local_list = [1, 2, 3]
            local_list.append(4)
            state.len_a = len(local_list)

        def create_dict(state: _FreshObjectsState) -> None:
            local_dict = {"a": 1, "b": 2}
            local_dict["c"] = 3
            state.len_b = len(local_dict)

        result = explore_dpor(
            setup=_FreshObjectsState,
            threads=[create_list, create_dict],
            invariant=lambda s: s.len_a == 4 and s.len_b == 3,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on fresh objects: {result}"

    def test_bytecode_safe(self) -> None:
        def create_list(state: _FreshObjectsState) -> None:
            local_list = [1, 2, 3]
            local_list.append(4)
            state.len_a = len(local_list)

        def create_dict(state: _FreshObjectsState) -> None:
            local_dict = {"a": 1, "b": 2}
            local_dict["c"] = 3
            state.len_b = len(local_dict)

        result = explore_interleavings(
            setup=_FreshObjectsState,
            threads=[create_list, create_dict],
            invariant=lambda s: s.len_a == 4 and s.len_b == 3,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on fresh objects: {result}"


# -- 24. Lock-protected augmented assignment ----------------------------------


class _AugAssignLockState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLockProtectedAugmentedAssignment:
    """The += pattern under lock — should be safe despite being a non-atomic operation."""

    def test_dpor_safe(self) -> None:
        def inc(state: _AugAssignLockState) -> None:
            with state.lock:
                state.value += 1

        result = explore_dpor(
            setup=_AugAssignLockState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on aug assign under lock: {result}"

    def test_bytecode_safe(self) -> None:
        def inc(state: _AugAssignLockState) -> None:
            with state.lock:
                state.value += 1

        result = explore_interleavings(
            setup=_AugAssignLockState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on aug assign under lock: {result}"


# -- 25. Lock-protected dict subscript operations ----------------------------


class _DictSubscriptLockState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"count": 0}
        self.lock = threading.Lock()


class TestLockProtectedDictSubscript:
    """Dict subscript read-modify-write under lock — safe."""

    def test_dpor_safe(self) -> None:
        def inc(state: _DictSubscriptLockState) -> None:
            with state.lock:
                state.data["count"] = state.data["count"] + 1

        result = explore_dpor(
            setup=_DictSubscriptLockState,
            threads=[inc, inc],
            invariant=lambda s: s.data["count"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on dict subscript under lock: {result}"

    def test_bytecode_safe(self) -> None:
        def inc(state: _DictSubscriptLockState) -> None:
            with state.lock:
                state.data["count"] = state.data["count"] + 1

        result = explore_interleavings(
            setup=_DictSubscriptLockState,
            threads=[inc, inc],
            invariant=lambda s: s.data["count"] == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on dict subscript under lock: {result}"


# -- 26. Lock-protected list append with length check -------------------------


class _ListAppendLockState:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.max_size = 2
        self.lock = threading.Lock()


class TestLockProtectedListAppend:
    """Check-then-act on list under lock — safe because lock serializes."""

    def test_dpor_safe(self) -> None:
        def safe_append(state: _ListAppendLockState) -> None:
            with state.lock:
                if len(state.items) < state.max_size:
                    state.items.append("item")

        result = explore_dpor(
            setup=_ListAppendLockState,
            threads=[safe_append, safe_append],
            invariant=lambda s: len(s.items) <= s.max_size,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on list append under lock: {result}"

    def test_bytecode_safe(self) -> None:
        def safe_append(state: _ListAppendLockState) -> None:
            with state.lock:
                if len(state.items) < state.max_size:
                    state.items.append("item")

        result = explore_interleavings(
            setup=_ListAppendLockState,
            threads=[safe_append, safe_append],
            invariant=lambda s: len(s.items) <= s.max_size,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on list append under lock: {result}"


# -- 27. Lock-protected property access (descriptor protocol) ----------------


class _PropertyLockState:
    def __init__(self) -> None:
        self._value = 0
        self.lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = v


class TestLockProtectedProperty:
    """Property getter/setter under lock — safe despite descriptor protocol."""

    def test_dpor_safe(self) -> None:
        def inc(state: _PropertyLockState) -> None:
            with state.lock:
                temp = state.value
                state.value = temp + 1

        result = explore_dpor(
            setup=_PropertyLockState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on property under lock: {result}"


# -- 28. Lock-protected global variable --------------------------------------


_safe_global_counter = 0
_safe_global_lock = threading.Lock()


class _SafeGlobalState:
    def __init__(self) -> None:
        global _safe_global_counter
        _safe_global_counter = 0


def _safe_global_inc(_state: _SafeGlobalState) -> None:
    global _safe_global_counter
    with _safe_global_lock:
        tmp = _safe_global_counter
        _safe_global_counter = tmp + 1


def _safe_global_check(_state: _SafeGlobalState) -> bool:
    return _safe_global_counter == 2


class TestLockProtectedGlobal:
    """Module-level global protected by a lock — safe."""

    def test_dpor_safe(self) -> None:
        result = explore_dpor(
            setup=_SafeGlobalState,
            threads=[_safe_global_inc, _safe_global_inc],
            invariant=_safe_global_check,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on lock-protected global: {result}"


# -- 29. Single writer, invariant is order-independent ------------------------


class _SingleWriterState:
    def __init__(self) -> None:
        self.value = 0


class TestSingleWriterMultipleValues:
    """One thread writes, one reads — invariant accepts either order."""

    def test_dpor_safe(self) -> None:
        def writer(state: _SingleWriterState) -> None:
            state.value = 42

        def noop(state: _SingleWriterState) -> None:
            pass

        result = explore_dpor(
            setup=_SingleWriterState,
            threads=[writer, noop],
            invariant=lambda s: s.value == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on single writer: {result}"

    def test_bytecode_safe(self) -> None:
        def writer(state: _SingleWriterState) -> None:
            state.value = 42

        def noop(state: _SingleWriterState) -> None:
            pass

        result = explore_interleavings(
            setup=_SingleWriterState,
            threads=[writer, noop],
            invariant=lambda s: s.value == 42,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on single writer: {result}"


# -- 30. Lock-protected complex data structure mutation ----------------------


class _ComplexMutationState:
    def __init__(self) -> None:
        self.data: dict[str, list[int]] = {"items": []}
        self.lock = threading.Lock()


class TestLockProtectedComplexMutation:
    """Complex dict + list mutation under lock — safe."""

    def test_dpor_safe(self) -> None:
        def append_and_sort(state: _ComplexMutationState) -> None:
            with state.lock:
                state.data["items"].append(1)

        result = explore_dpor(
            setup=_ComplexMutationState,
            threads=[append_and_sort, append_and_sort],
            invariant=lambda s: len(s.data["items"]) == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on complex mutation under lock: {result}"

    def test_bytecode_safe(self) -> None:
        def append_and_sort(state: _ComplexMutationState) -> None:
            with state.lock:
                state.data["items"].append(1)

        result = explore_interleavings(
            setup=_ComplexMutationState,
            threads=[append_and_sort, append_and_sort],
            invariant=lambda s: len(s.data["items"]) == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on complex mutation under lock: {result}"


# -- 31. Semaphore with count > 1 (concurrent access allowed) ----------------


class _SemaphoreMultiState:
    def __init__(self) -> None:
        self.result_a = 0
        self.result_b = 0
        self.sem = threading.Semaphore(2)  # allows both threads in


class TestSemaphoreMultiPermit:
    """Semaphore(2) with 2 threads — both can enter simultaneously."""

    def test_dpor_safe(self) -> None:
        def work_a(state: _SemaphoreMultiState) -> None:
            state.sem.acquire()
            try:
                state.result_a = 42
            finally:
                state.sem.release()

        def work_b(state: _SemaphoreMultiState) -> None:
            state.sem.acquire()
            try:
                state.result_b = 99
            finally:
                state.sem.release()

        result = explore_dpor(
            setup=_SemaphoreMultiState,
            threads=[work_a, work_b],
            invariant=lambda s: s.result_a == 42 and s.result_b == 99,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on semaphore multi-permit: {result}"


# -- 32. Lock-protected swap -------------------------------------------------


class _SwapLockState:
    def __init__(self) -> None:
        self.a = 1
        self.b = 2
        self.lock = threading.Lock()


class TestLockProtectedSwap:
    """Tuple swap under lock — safe despite reading two attributes."""

    def test_dpor_safe(self) -> None:
        def swap(state: _SwapLockState) -> None:
            with state.lock:
                state.a, state.b = state.b, state.a

        def read(state: _SwapLockState) -> None:
            with state.lock:
                _ = state.a + state.b

        result = explore_dpor(
            setup=_SwapLockState,
            threads=[swap, read],
            invariant=lambda s: s.a + s.b == 3,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on lock-protected swap: {result}"

    def test_bytecode_safe(self) -> None:
        def swap(state: _SwapLockState) -> None:
            with state.lock:
                state.a, state.b = state.b, state.a

        def read(state: _SwapLockState) -> None:
            with state.lock:
                _ = state.a + state.b

        result = explore_interleavings(
            setup=_SwapLockState,
            threads=[swap, read],
            invariant=lambda s: s.a + s.b == 3,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on lock-protected swap: {result}"


# -- 33. Local-only computation (no shared state access at all) ---------------


class _LocalOnlyState:
    def __init__(self) -> None:
        self.done_a = False
        self.done_b = False


class TestLocalOnlyComputation:
    """Threads do purely local computation, then write result to independent attribute."""

    def test_dpor_safe(self) -> None:
        def work_a(state: _LocalOnlyState) -> None:
            x = 0
            for i in range(5):
                x += i
            _ = x * 2
            state.done_a = True

        def work_b(state: _LocalOnlyState) -> None:
            y = 1
            for i in range(1, 4):
                y *= i
            _ = y + 1
            state.done_b = True

        result = explore_dpor(
            setup=_LocalOnlyState,
            threads=[work_a, work_b],
            invariant=lambda s: s.done_a and s.done_b,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on local-only computation: {result}"

    def test_bytecode_safe(self) -> None:
        def work_a(state: _LocalOnlyState) -> None:
            x = sum(range(5))
            _ = x
            state.done_a = True

        def work_b(state: _LocalOnlyState) -> None:
            y = 1
            for i in range(1, 4):
                y *= i
            _ = y
            state.done_b = True

        result = explore_interleavings(
            setup=_LocalOnlyState,
            threads=[work_a, work_b],
            invariant=lambda s: s.done_a and s.done_b,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on local-only computation: {result}"


# -- 34. Lock-protected closure cell variable ---------------------------------


def _make_safe_closure() -> tuple[threading.Lock, object, object]:
    lock = threading.Lock()
    count = 0

    def increment() -> None:
        nonlocal count
        with lock:
            temp = count
            count = temp + 1

    def get() -> int:
        return count

    return lock, increment, get


class _SafeClosureCellState:
    def __init__(self) -> None:
        _, self.increment, self.get = _make_safe_closure()


class TestLockProtectedClosureCell:
    """nonlocal variable (closure cell) protected by lock — safe."""

    def test_dpor_safe(self) -> None:
        result = explore_dpor(
            setup=_SafeClosureCellState,
            threads=[lambda s: s.increment(), lambda s: s.increment()],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on closure cell under lock: {result}"

    def test_bytecode_safe(self) -> None:
        result = explore_interleavings(
            setup=_SafeClosureCellState,
            threads=[lambda s: s.increment(), lambda s: s.increment()],
            invariant=lambda s: s.get() == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on closure cell under lock: {result}"


# -- 35. Lock-protected setattr/getattr calls --------------------------------


class _SetattrLockState:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()


class TestLockProtectedSetattrGetattr:
    """setattr/getattr under lock — safe despite passthrough builtin tracking."""

    def test_dpor_safe(self) -> None:
        def inc(state: _SetattrLockState) -> None:
            with state.lock:
                temp = getattr(state, "value")
                setattr(state, "value", temp + 1)

        result = explore_dpor(
            setup=_SetattrLockState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on setattr/getattr under lock: {result}"

    def test_bytecode_safe(self) -> None:
        def inc(state: _SetattrLockState) -> None:
            with state.lock:
                temp = getattr(state, "value")
                setattr(state, "value", temp + 1)

        result = explore_interleavings(
            setup=_SetattrLockState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on setattr/getattr under lock: {result}"


# -- 36. Independent attributes with multi-step computation ------------------


class _MultiStepIndependentState:
    def __init__(self) -> None:
        self.sum_a = 0
        self.sum_b = 0


class TestMultiStepIndependent:
    """Each thread does multi-step computation on its own attribute."""

    def test_dpor_safe(self) -> None:
        def compute_a(state: _MultiStepIndependentState) -> None:
            state.sum_a = 0
            state.sum_a += 10
            state.sum_a += 20
            state.sum_a += 30

        def compute_b(state: _MultiStepIndependentState) -> None:
            state.sum_b = 0
            state.sum_b += 100
            state.sum_b += 200

        result = explore_dpor(
            setup=_MultiStepIndependentState,
            threads=[compute_a, compute_b],
            invariant=lambda s: s.sum_a == 60 and s.sum_b == 300,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert result.property_holds, f"False positive on multi-step independent: {result}"

    def test_bytecode_safe(self) -> None:
        def compute_a(state: _MultiStepIndependentState) -> None:
            state.sum_a = 0
            state.sum_a += 10
            state.sum_a += 20
            state.sum_a += 30

        def compute_b(state: _MultiStepIndependentState) -> None:
            state.sum_b = 0
            state.sum_b += 100
            state.sum_b += 200

        result = explore_interleavings(
            setup=_MultiStepIndependentState,
            threads=[compute_a, compute_b],
            invariant=lambda s: s.sum_a == 60 and s.sum_b == 300,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )
        assert result.property_holds, f"False positive on multi-step independent: {result}"
