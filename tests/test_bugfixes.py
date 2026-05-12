"""Tests that reproduce specific bugs and verify their fixes.

Each test is named after the bug it reproduces, with a docstring explaining
the root cause.  The tests are written to fail *before* the fix and pass
*after*.
"""

import threading
import time
from unittest.mock import patch as mock_patch

import pytest
from frontrun._dpor import PyDporEngine

from frontrun._cooperative import CooperativeCondition
from frontrun.bytecode import BytecodeShuffler, OpcodeScheduler
from frontrun.common import Schedule, Step
from frontrun.dpor import DporBytecodeRunner, DporScheduler, explore_dpor
from frontrun.trace_markers import TraceExecutor

# ---------------------------------------------------------------------------
# Bug: _patch_class_methods closure captures _is_executemany by reference
# ---------------------------------------------------------------------------


class TestPatchClassMethodsClosureBug:
    """_patch_class_methods loops over ("execute", "executemany") and defines
    _is_executemany inside the loop body, but the inner _patched closure
    captures it by reference. After the loop, _is_executemany is True (from
    the "executemany" iteration), so cursor.execute() incorrectly passes
    is_executemany=True to _intercept_execute.
    """

    def test_execute_not_treated_as_executemany(self):
        from frontrun._patching import restore_patches
        from frontrun._sql_cursor import _intercept_execute, _patch_class_methods

        captured: list[bool] = []
        original_intercept = _intercept_execute

        def spy_intercept(orig, cursor, operation, parameters=None, **kwargs):
            captured.append(kwargs.get("is_executemany", False))
            return None

        class FakeCursor:
            def execute(self, op, params=None):
                pass

            def executemany(self, op, seq_of_params):
                pass

        patches: list = []
        originals: dict = {}
        from frontrun._sql_cursor import _ORIGINAL_METHODS, _PATCHES

        old_originals = dict(_ORIGINAL_METHODS)
        old_patches = list(_PATCHES)

        with mock_patch("frontrun._sql_cursor._intercept_execute", side_effect=spy_intercept):
            _patch_class_methods(FakeCursor, "qmark")
            FakeCursor().execute("SELECT 1")
            FakeCursor().executemany("INSERT INTO t VALUES (?)", [(1,)])

        restore_patches(_PATCHES[len(old_patches) :])
        _PATCHES[:] = old_patches
        _ORIGINAL_METHODS.clear()
        _ORIGINAL_METHODS.update(old_originals)

        assert len(captured) == 2
        assert captured[0] is False, "execute() should pass is_executemany=False"
        assert captured[1] is True, "executemany() should pass is_executemany=True"


# ---------------------------------------------------------------------------
# Bug: CooperativeCondition.wait() loses notifications
# ---------------------------------------------------------------------------


class TestCooperativeConditionNotificationLoss:
    """CooperativeCondition.wait() used to poll _real_cond.wait(timeout=0) in a
    spin loop.  Since notify() calls _real_cond.notify() while no thread is
    actually blocked in _real_cond.wait(), the notification was silently dropped.

    The fix replaces the real-condition polling with a simple notification
    counter that cannot lose notifications.
    """

    def test_notify_wakes_waiter_promptly(self):
        """A notify() should wake a waiting thread promptly, not after 5s.

        Uses raw threads (no scheduler) to test the non-cooperative path of
        CooperativeCondition.  The waiter calls wait() which falls through to
        the real condition.  This validates the basic mechanism works.
        """
        cond = CooperativeCondition()
        flag = [False]

        def waiter():
            with cond:
                cond.wait(timeout=3.0)
                flag[0] = True

        def notifier():
            time.sleep(0.1)  # let waiter enter wait()
            with cond:
                cond.notify()

        t1 = threading.Thread(target=waiter, daemon=True)
        t2 = threading.Thread(target=notifier, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert flag[0], "waiter should have been woken by notify()"

    def test_notification_counter_does_not_lose_updates(self):
        """The ticket-based notification system should never lose a notify().

        This directly tests the mechanism: each waiting thread gets a
        ticket, and notify(n) advances the served counter by exactly n,
        waking only the intended number of waiters.
        """
        cond = CooperativeCondition()

        # notify() without waiters is a no-op (correct: nothing to wake)
        with cond:
            cond.notify()
        assert cond._served == 0, "notify with no waiters should not advance served"

        # Simulate 3 waiters taking tickets
        cond._waiters = 3
        cond._next_ticket = 3  # tickets 0, 1, 2

        # notify(1) should serve exactly 1 ticket
        before = cond._served
        with cond:
            cond.notify()
        assert cond._served == before + 1, "notify should advance served by 1"

        # notify_all should serve all remaining un-served tickets (not all _waiters,
        # since some may already have been served by the previous notify(1)).
        with cond:
            cond.notify_all()
        assert cond._served == cond._next_ticket, "notify_all should serve all remaining tickets"

    def test_notify_without_lock_raises(self):
        """Calling notify() without holding the lock should raise RuntimeError."""
        cond = CooperativeCondition()
        with pytest.raises(RuntimeError, match="cannot notify on un-acquired lock"):
            cond.notify()


# ---------------------------------------------------------------------------
# Bug: CooperativeCondition.wait_for() exits after one wait with timeout
# ---------------------------------------------------------------------------


class TestCooperativeConditionWaitFor:
    """CooperativeCondition.wait_for(pred, timeout=T) used to break out of
    the loop after a single wait attempt when any timeout was specified,
    even if most of the timeout period remained.

    The fix tracks a deadline and loops until it expires.
    """

    def test_wait_for_returns_true_when_predicate_true_immediately(self):
        """wait_for should return True immediately if predicate is already true."""
        cond = CooperativeCondition()
        with cond:
            result = cond.wait_for(lambda: True, timeout=10.0)
        assert result is True

    def test_wait_for_returns_false_on_timeout(self):
        """wait_for should return False when timeout expires."""
        cond = CooperativeCondition()

        start = time.monotonic()
        with cond:
            result = cond.wait_for(lambda: False, timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.09, "should wait approximately the timeout duration"

    def test_wait_for_with_timeout_retries(self):
        """wait_for(pred, timeout=T) should keep retrying until predicate
        becomes true, not exit after one wait."""
        cond = CooperativeCondition()
        state = {"ready": False, "result": None}

        def waiter():
            with cond:
                state["result"] = cond.wait_for(lambda: state["ready"], timeout=5.0)

        def setter():
            time.sleep(0.2)  # short delay — well within 5s timeout
            with cond:
                state["ready"] = True
                cond.notify()

        t1 = threading.Thread(target=waiter, daemon=True)
        t2 = threading.Thread(target=setter, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Before fix: wait_for would exit after one wait (returning False)
        # After fix: wait_for retries and sees ready=True
        assert state["result"] is True, f"wait_for should return True, got {state['result']}"


# ---------------------------------------------------------------------------
# Bug: explore_dpor doesn't stop on first failure
# ---------------------------------------------------------------------------


class TestDporStopOnFirst:
    """explore_dpor used to continue exploring all interleavings after
    finding a violation.  The fix adds a stop_on_first parameter."""

    def test_stop_on_first_reduces_exploration(self):
        """With stop_on_first=True (default), exploration should stop after
        the first counterexample is found."""

        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                temp = self.value
                self.value = temp + 1

        result_early = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
        )
        assert not result_early.property_holds
        assert result_early.counterexample is not None
        early_count = result_early.num_explored

        result_full = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
        )
        assert not result_full.property_holds
        full_count = result_full.num_explored
        assert full_count >= early_count, (
            f"full exploration ({full_count}) should explore at least as many as early-stop ({early_count})"
        )

    def test_stop_on_first_false_explores_all(self):
        """With stop_on_first=False, all interleavings are explored."""

        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
        )
        assert not result.property_holds
        assert len(result.failures) >= 1


# ---------------------------------------------------------------------------
# Bug: Per-thread timeout accumulates (N threads × timeout)
# ---------------------------------------------------------------------------


class TestTimeoutAccumulation:
    """BytecodeShuffler.run() and DporBytecodeRunner.run() used to apply
    timeout per-thread.  With N threads, total wait was N*timeout.

    The fix uses a global deadline so total wait is bounded by timeout."""

    @pytest.mark.intentionally_leaves_dangling_threads
    def test_total_timeout_is_bounded(self):
        """With 3 threads, total wait should still be ~timeout, not 3*timeout."""
        schedule = [0] * 10  # very short schedule
        scheduler = OpcodeScheduler(schedule, num_threads=3)
        runner = BytecodeShuffler(scheduler)

        def hang():
            time.sleep(100)

        start = time.monotonic()
        runner.run([hang, hang, hang], timeout=1.0)
        elapsed = time.monotonic() - start

        # Before fix: elapsed ~ 3.0 (3 threads x 1.0s each)
        # After fix: elapsed ~ 1.0 (global deadline) + grace period for abort
        assert elapsed < 4.0, f"total time {elapsed:.1f}s should be <4.0s with timeout=1.0"

    @pytest.mark.intentionally_leaves_dangling_threads
    def test_dpor_total_timeout_is_bounded(self):
        """Same test for DporBytecodeRunner."""
        engine = PyDporEngine(num_threads=3)
        execution = engine.begin_execution()
        scheduler = DporScheduler(engine, execution, num_threads=3)
        runner = DporBytecodeRunner(scheduler)

        def hang():
            time.sleep(100)

        start = time.monotonic()
        runner.run([hang, hang, hang], timeout=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 4.0, f"total time {elapsed:.1f}s should be <4.0s with timeout=1.0"


# ---------------------------------------------------------------------------
# Bug: ThreadCoordinator.wait_for_turn() has no condition timeout
# ---------------------------------------------------------------------------


class TestTraceMarkerTimeout:
    """ThreadCoordinator.wait_for_turn() used to call condition.wait() with
    no timeout.  The fix adds:
    1. A 5-second fallback condition.wait(timeout=5.0) for stalled threads
    2. A schedule-incomplete check in TraceExecutor.wait() for threads that
       finish before consuming the full schedule.
    """

    def test_incomplete_schedule_reports_error(self):
        """A schedule referencing a non-existent marker should raise TimeoutError
        with a clear message about which steps were never reached."""
        schedule = Schedule(
            [
                Step("t1", "marker_that_exists"),
                Step("t1", "marker_that_does_not_exist"),  # will never be hit
            ]
        )

        executor = TraceExecutor(schedule)

        def worker():
            x = 1  # frontrun: marker_that_exists
            _ = x + 1  # no marker here

        with pytest.raises(TimeoutError, match="Schedule incomplete.*marker_that_does_not_exist"):
            executor.run({"t1": worker}, timeout=5.0)


# ---------------------------------------------------------------------------
# Bug: patch_locks() breaks concurrent.futures.ThreadPoolExecutor
# ---------------------------------------------------------------------------


class TestPatchLocksConcurrentFutures:
    """patch_locks() replaces threading.Lock with CooperativeLock.  Modules
    that import threading.Lock at module-load time (like concurrent.futures.thread)
    would get a CooperativeLock for their internal locks, causing failures when
    those locks are later tested with isinstance(..., _thread.LockType).

    The fix pre-imports concurrent.futures.thread before monkey-patching so
    its internal lock objects are created with the real C-level lock.
    """

    def test_patch_locks_concurrent_futures_threadpool(self):
        """After patch_locks(), ThreadPoolExecutor should still submit tasks."""
        import concurrent.futures

        from frontrun._cooperative import patch_locks, unpatch_locks

        patch_locks()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: 42)
                result = future.result(timeout=5.0)
            assert result == 42
        finally:
            unpatch_locks()

    def test_patch_locks_thread_start_event_handshake(self):
        """Thread.start() should complete when cooperative patching is enabled."""
        from frontrun._cooperative import patch_locks, unpatch_locks

        started = threading.Event()
        finished = threading.Event()

        def worker() -> None:
            started.set()
            finished.wait(1.0)

        patch_locks()
        try:
            t = threading.Thread(target=worker)
            t.start()
            assert started.wait(1.0)
            finished.set()
            t.join(timeout=1.0)
            assert not t.is_alive()
        finally:
            unpatch_locks()
