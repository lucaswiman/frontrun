"""Regression tests for sound I/O deferral while holding locks.

Problem statement
-----------------

Sync DPOR buffers I/O in ``pending_io`` and reports it to the Rust engine at
the next scheduling point. Deferring every I/O until ``lock_depth == 0`` keeps
same-lock critical sections compact, but it can hide a real race window when a
different thread reaches a competing I/O operation before the lock is released.

The intended rule is:

  - A thread keeps its own pending I/O buffered while it is inside a lock.
  - When another thread reaches a real I/O boundary, deferred I/O from
    lock-held threads must be flushed before the current thread reports its own
    I/O.
"""

from __future__ import annotations

import threading

from frontrun._deadlock import install_wait_for_graph, uninstall_wait_for_graph
from frontrun.dpor import DporBytecodeRunner, DporScheduler, _dpor_tls


class _FakeExecution:
    def __init__(self, runnable: list[int]) -> None:
        self._runnable = list(runnable)

    def runnable_threads(self) -> list[int]:
        return list(self._runnable)


class _FakeEngine:
    def __init__(self) -> None:
        self.io_calls: list[tuple[int, int, str]] = []
        self.schedule_calls = 0

    def schedule(self, execution: _FakeExecution) -> int | None:
        self.schedule_calls += 1
        runnable = execution.runnable_threads()
        return runnable[0] if runnable else None

    def report_io_access(self, execution: _FakeExecution, thread_id: int, object_id: int, kind: str) -> None:
        self.io_calls.append((thread_id, object_id, kind))

    def report_synced_io_access(self, execution: _FakeExecution, thread_id: int, object_id: int, kind: str) -> None:
        self.io_calls.append((thread_id, object_id, kind))


class TestLockDepthIoWindow:
    def teardown_method(self) -> None:
        _dpor_tls.pending_io = []
        _dpor_tls.lock_depth = 0
        uninstall_wait_for_graph()

    def test_flushes_current_thread_pending_io_immediately_outside_lock(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0])
        scheduler = DporScheduler(engine, execution, num_threads=1)

        pending_io = [(123, "write", False)]
        scheduler._pending_io_by_thread[0] = pending_io
        scheduler._lock_depth_by_thread[0] = 0
        _dpor_tls.pending_io = pending_io
        _dpor_tls.lock_depth = 0

        assert scheduler._report_and_wait(None, 0)

        assert engine.io_calls == [(0, 123, "write")]
        assert _dpor_tls.pending_io == []

    def test_keeps_current_thread_pending_io_buffered_inside_lock(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0, 1])
        scheduler = DporScheduler(engine, execution, num_threads=2)

        pending_io = [(456, "read", False)]
        scheduler._pending_io_by_thread[0] = pending_io
        scheduler._lock_depth_by_thread[0] = 1
        _dpor_tls.pending_io = pending_io
        _dpor_tls.lock_depth = 1

        assert scheduler._report_and_wait(None, 0)

        assert engine.io_calls == []
        assert _dpor_tls.pending_io == [(456, "read", False)]

    def test_flushes_other_threads_deferred_io_when_current_thread_reaches_io_boundary(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0, 1])
        scheduler = DporScheduler(engine, execution, num_threads=2)

        deferred_other = [(789, "write", False)]
        current_io = [(999, "read", False)]
        scheduler._pending_io_by_thread[0] = deferred_other
        scheduler._pending_io_by_thread[1] = current_io
        scheduler._lock_depth_by_thread[0] = 1
        scheduler._lock_depth_by_thread[1] = 0
        scheduler._current_thread = 1
        _dpor_tls.pending_io = current_io
        _dpor_tls.lock_depth = 0

        assert scheduler._report_and_wait(None, 1)

        assert engine.io_calls == [(0, 789, "write"), (1, 999, "read")]
        assert scheduler._pending_io_by_thread[0] == []
        assert _dpor_tls.pending_io == []

    def test_teardown_serializes_orphan_pending_io_flush(self) -> None:
        """Teardown and a cross-thread flush must not report the same event twice."""
        engine = _FakeEngine()
        execution = _FakeExecution([0, 1])
        scheduler = DporScheduler(engine, execution, num_threads=2)
        runner = DporBytecodeRunner(scheduler, detect_io=False)
        pending_io = [(123, "write", False)]
        scheduler._pending_io_by_thread[0] = pending_io
        scheduler._lock_depth_by_thread[0] = 0

        flush_boundary = threading.Event()
        condition_held = threading.Event()
        competitor_done = threading.Event()
        errors: list[BaseException] = []
        real_condition = scheduler._condition

        class SignalingCondition:
            def __enter__(self) -> object:
                if threading.current_thread().name == "teardown":
                    flush_boundary.set()
                return real_condition.__enter__()

            def __exit__(self, *args: object) -> None:
                real_condition.__exit__(*args)

        class HandoffEngineLock:
            def __enter__(self) -> None:
                if threading.current_thread().name == "teardown":
                    flush_boundary.set()
                    if not competitor_done.wait(2):
                        raise TimeoutError("competitor did not flush pending I/O")

            def __exit__(self, *args: object) -> None:
                pass

        scheduler._condition = SignalingCondition()  # type: ignore[assignment]
        scheduler._engine_lock = HandoffEngineLock()  # type: ignore[assignment]

        def competitor() -> None:
            try:
                with scheduler._condition:
                    condition_held.set()
                    if not flush_boundary.wait(2):
                        raise TimeoutError("teardown did not reach the flush boundary")
                    scheduler._flush_pending_io_for_unlocked(0, allow_inside_lock=True)
                    competitor_done.set()
            except BaseException as exc:
                errors.append(exc)
                competitor_done.set()

        def teardown() -> None:
            try:
                _dpor_tls.thread_id = 0
                _dpor_tls.engine = engine
                _dpor_tls.execution = execution
                _dpor_tls.pending_io = pending_io
                _dpor_tls.lock_depth = 0
                runner._teardown_dpor_tls()
            except BaseException as exc:
                errors.append(exc)

        competitor_thread = threading.Thread(target=competitor, name="competitor")
        teardown_thread = threading.Thread(target=teardown, name="teardown")
        competitor_thread.start()
        assert condition_held.wait(2)
        teardown_thread.start()
        competitor_thread.join(3)
        teardown_thread.join(3)

        assert not competitor_thread.is_alive()
        assert not teardown_thread.is_alive()
        assert errors == []
        assert engine.io_calls == [(0, 123, "write")]

    def test_mark_done_sets_dpor_machinery_guard(self) -> None:
        """``mark_done`` holds the (non-reentrant) scheduler condition while
        finalizing a thread at teardown — exactly when GC ``__del__`` chains
        fire.  It must set the ``_in_dpor_machinery`` guard for its critical
        section (like ``_report_and_wait``/``before_io`` do) so a cooperative
        lock released by such a ``__del__`` falls back to the real lock instead
        of re-entering the scheduler and self-deadlocking (defect #7).
        """
        from frontrun._cooperative import _in_dpor_machinery, _scheduler_tls

        seen: list[bool] = []

        class _SpyExecution(_FakeExecution):
            def finish_thread(self, thread_id: int) -> None:
                # Runs inside mark_done's critical section.
                seen.append(_in_dpor_machinery())

        engine = _FakeEngine()
        execution = _SpyExecution([0])
        scheduler = DporScheduler(engine, execution, num_threads=1)

        try:
            scheduler.mark_done(0)
        finally:
            _scheduler_tls._in_dpor_machinery = False

        assert seen == [True], (
            "mark_done must set the _in_dpor_machinery guard around its condition-holding critical section"
        )

    def test_skips_io_scheduling_point_when_all_other_threads_wait_on_held_locks(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0, 1])
        scheduler = DporScheduler(engine, execution, num_threads=2)
        graph = install_wait_for_graph()
        graph.add_holding(0, 123, kind="lock")
        graph.add_waiting(1, 123, kind="lock")

        pending_io = [(321, "write", False)]
        scheduler._pending_io_by_thread[0] = pending_io
        scheduler._lock_depth_by_thread[0] = 1
        scheduler._current_thread = 0
        _dpor_tls.pending_io = pending_io
        _dpor_tls.lock_depth = 1
        baseline_schedule_calls = engine.schedule_calls

        assert scheduler._report_and_wait(None, 0)

        assert engine.schedule_calls == baseline_schedule_calls, (
            "When all other live threads are transitively blocked behind the "
            "current thread's held locks, the I/O boundary should not create a "
            "new scheduling point."
        )
        assert engine.io_calls == []
        assert _dpor_tls.pending_io == [(321, "write", False)]
