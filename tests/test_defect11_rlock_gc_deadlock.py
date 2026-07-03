"""Defect #11: GC destructor deadlock with CooperativeRLock.

CooperativeRLock.release() is missing the _in_dpor_machinery() reentrancy
guard that CooperativeLock.release() has.  When GC fires during
_report_and_wait (which holds scheduler._condition and sets
_in_dpor_machinery=True), redis.Redis.__del__ can trigger
connection_pool.disconnect() which uses an RLock.  The RLock.release()
calls _report("lock_release") which tries to acquire scheduler._condition,
causing a deadlock.

Fix: add the same guard to CooperativeRLock.release().
"""

from __future__ import annotations

from frontrun._cooperative import (
    CooperativeRLock,
    _scheduler_tls,
    set_context,
    set_sync_reporter,
)
from frontrun._deadlock import install_wait_for_graph, uninstall_wait_for_graph


class TestRLockGCReentrancyGuard:
    """CooperativeRLock must not deadlock when released during DPOR machinery."""

    def test_rlock_release_inside_dpor_machinery(self) -> None:
        """Releasing a CooperativeRLock while _in_dpor_machinery should not deadlock.

        Simulates: _report_and_wait holds scheduler._condition →
        GC fires → redis.__del__ → connection_pool.disconnect() →
        with self._lock (CooperativeRLock) → __exit__ → release() →
        _report("lock_release") → reporter() → with scheduler._condition
        → DEADLOCK.
        """
        rlock = CooperativeRLock()
        released_ok = False
        release_reporter_called = False

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()

        try:
            set_context(scheduler, 0)  # type: ignore[arg-type]

            def reporter(event: str, obj_id: int, lock_obj: object) -> None:
                nonlocal release_reporter_called
                if event == "lock_release":
                    release_reporter_called = True

            set_sync_reporter(reporter)

            # Acquire the RLock normally (this calls _report("lock_acquire"),
            # which is expected and fine).
            rlock.acquire()

            # Simulate being inside DPOR machinery (as if _report_and_wait
            # holds scheduler._condition)
            _scheduler_tls._in_dpor_machinery = True

            # Release should NOT call _report() since we're in DPOR machinery.
            # Without the fix, this deadlocks (or in this test, calls reporter
            # which would try to acquire scheduler._condition).
            rlock.release()
            released_ok = True
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        assert released_ok, "RLock release deadlocked inside DPOR machinery"
        # The guard should skip _report("lock_release") entirely
        assert not release_reporter_called, (
            "CooperativeRLock.release() should skip _report() when _in_dpor_machinery() "
            "is True, but lock_release reporter was called"
        )

    def test_rlock_release_normal_reports(self) -> None:
        """CooperativeRLock.release() should still call _report() normally."""
        rlock = CooperativeRLock()
        events: list[str] = []

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()

        try:
            set_context(scheduler, 0)  # type: ignore[arg-type]

            def reporter(event: str, obj_id: int, lock_obj: object) -> None:
                events.append(event)

            set_sync_reporter(reporter)

            rlock.acquire()
            rlock.release()
        finally:
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        assert "lock_release" in events, f"Expected lock_release event, got {events}"

    def test_rlock_machinery_release_removes_holding_edge(self) -> None:
        """Releasing a normally-acquired RLock during machinery must remove its
        wait-for-graph holding edge, or a stale edge can fabricate a deadlock.

        Scenario: a thread acquires the RLock normally (adding a
        ``lock -> holder`` edge), then a GC ``__del__`` releases it while
        ``_in_dpor_machinery()`` is True.  ``CooperativeLock.release()`` scrubs
        the edge on this path; ``CooperativeRLock.release()`` must do the same.
        A leaked edge lets ``_check_lock_cycle`` find a cycle through a lock
        nobody actually holds and raise a spurious ``DeadlockError``.
        """
        rlock = CooperativeRLock()

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()
        lock_node = ("lock", rlock._object_id)

        try:
            set_context(scheduler, 0)  # type: ignore[arg-type]
            set_sync_reporter(lambda event, obj_id, lock_obj: None)

            # Acquire normally: records a holding edge lock -> thread 0.
            rlock.acquire()
            edge_before_release = ("thread", 0) in graph._edges.get(lock_node, set())

            # GC __del__ fires during DPOR machinery and releases the lock.
            _scheduler_tls._in_dpor_machinery = True
            rlock.release()
            # Capture the edge state *before* the finally uninstalls (and clears)
            # the graph — otherwise the assertion is vacuously true.
            edge_after_release = ("thread", 0) in graph._edges.get(lock_node, set())
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        assert edge_before_release, "precondition: holding edge recorded on acquire"
        assert not edge_after_release, (
            "CooperativeRLock.release() left a stale holding edge in the wait-for graph "
            "when released during DPOR machinery"
        )

    def test_rlock_reentrant_release_inner_during_machinery(self) -> None:
        """Reentrant RLock: inner release during machinery should not deadlock.

        Simulates nested acquisition where the outer release happens normally
        but the inner release happens during DPOR machinery.
        """
        rlock = CooperativeRLock()

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()

        try:
            set_context(scheduler, 0)  # type: ignore[arg-type]
            set_sync_reporter(lambda event, obj_id, lock_obj: None)

            # Acquire twice (reentrant)
            rlock.acquire()
            rlock.acquire()

            # Release inner (just decrements count, doesn't release underlying lock)
            rlock.release()

            # Now simulate DPOR machinery
            _scheduler_tls._in_dpor_machinery = True

            # Release outer — this is the final release that touches the real lock
            rlock.release()
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        # Should complete without deadlock
        assert not rlock._is_owned()

    def test_rlock_acquire_contested_inside_dpor_machinery_falls_back(self) -> None:
        """CooperativeRLock.acquire() inside DPOR machinery should fall back to real blocking.

        Regression: CooperativeLock.acquire() has an _in_dpor_machinery() guard
        that falls back to the real lock, but CooperativeRLock.acquire() is
        missing this guard.  When GC fires during DPOR machinery and triggers
        a __del__ that acquires a contested RLock, the RLock.acquire() enters
        the slow path and calls scheduler.wait_for_turn(), which is unsafe.

        We use a real thread to hold the lock, then try to acquire from the
        main thread with blocking=True and a short timeout while in machinery
        mode.  The test checks that scheduler.wait_for_turn is NOT called.
        """
        import threading

        rlock = CooperativeRLock()
        wait_for_turn_called = False
        holder_ready = threading.Event()
        holder_release = threading.Event()

        class FakeScheduler:
            _finished = False
            _error = None

            def wait_for_turn(self, thread_id: int) -> None:
                nonlocal wait_for_turn_called
                wait_for_turn_called = True

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()

        def holder() -> None:
            """Hold the underlying lock from another thread."""
            rlock._lock.acquire()
            holder_ready.set()
            holder_release.wait(timeout=5.0)
            rlock._lock.release()

        t = threading.Thread(target=holder)
        t.start()
        holder_ready.wait(timeout=5.0)

        try:
            set_context(scheduler, 0)  # type: ignore[arg-type]
            set_sync_reporter(lambda event, obj_id, lock_obj: None)

            # Simulate being inside DPOR machinery
            _scheduler_tls._in_dpor_machinery = True

            # Blocking acquire with timeout — should fall back to real blocking
            # and NOT call scheduler.wait_for_turn()
            result = rlock.acquire(blocking=True, timeout=0.1)
            # Primary assertion: scheduler should NOT be involved during machinery
            assert not wait_for_turn_called, (
                "CooperativeRLock.acquire() should fall back to real blocking when "
                "_in_dpor_machinery() is True, but scheduler.wait_for_turn was called"
            )
            if result:
                rlock._owner = None
                rlock._count = 0
                rlock._lock.release()
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            holder_release.set()
            t.join(timeout=5.0)
            uninstall_wait_for_graph()
