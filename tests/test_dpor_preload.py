"""Tests for DPOR detection of races via LD_PRELOAD C-level I/O interception.

These tests verify that the IOEventDispatcher → _PreloadBridge → DPOR engine
pipeline works: C-level socket/file events intercepted by the LD_PRELOAD
library are routed to the correct DPOR thread and reported as I/O accesses,
enabling DPOR to explore interleavings of threads that share an external
resource (e.g. a database via TCP).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from frontrun._preload_io import PreloadIOEvent
from frontrun.dpor import _PreloadBridge, explore_dpor

# ---------------------------------------------------------------------------
# Unit tests for _PreloadBridge
# ---------------------------------------------------------------------------


class TestPreloadBridge:
    """Verify that _PreloadBridge correctly routes events by OS TID."""

    def test_routes_events_by_tid(self) -> None:
        """Events are buffered per DPOR thread based on OS TID mapping."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        bridge.register_thread(os_tid=2000, dpor_id=1)

        ev0 = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=1000)
        ev1 = PreloadIOEvent(kind="read", resource_id="socket:127.0.0.1:5432", fd=6, pid=1, tid=2000)

        bridge.listener(ev0)
        bridge.listener(ev1)

        events_0 = bridge.drain(0)
        events_1 = bridge.drain(1)
        assert len(events_0) == 1, "Thread 0 should have 1 event"
        assert len(events_1) == 1, "Thread 1 should have 1 event"
        # Events preserve their original kind (write stays write, read stays read)
        assert events_0[0][1] == "write"
        assert events_1[0][1] == "read"

    def test_ignores_unregistered_tids(self) -> None:
        """Events from unknown TIDs (setup, invariant, reader thread) are ignored."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)

        ev = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=9999)
        bridge.listener(ev)

        assert bridge.drain(0) == []

    def test_clear_resets_state(self) -> None:
        """clear() removes all mappings and pending events."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        ev = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=1000)
        bridge.listener(ev)

        bridge.clear()
        assert bridge.drain(0) == []

    def test_unregister_thread(self) -> None:
        """After unregistering, events from that TID are ignored."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        bridge.unregister_thread(os_tid=1000)

        ev = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=1000)
        bridge.listener(ev)
        assert bridge.drain(0) == []

    def test_same_resource_yields_same_object_key(self) -> None:
        """Two events for the same resource_id should produce identical object keys."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        bridge.register_thread(os_tid=2000, dpor_id=1)

        ev0 = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=1000)
        ev1 = PreloadIOEvent(kind="write", resource_id="socket:127.0.0.1:5432", fd=6, pid=1, tid=2000)
        bridge.listener(ev0)
        bridge.listener(ev1)

        key_0 = bridge.drain(0)[0][0]
        key_1 = bridge.drain(1)[0][0]
        assert key_0 == key_1, "Same resource_id should map to the same DPOR object key"

    def test_redis_suppressed_thread_file_io_passes_through(self) -> None:
        """File I/O from a Redis-suppressed thread must not be dropped.

        SQL suppression is gated on ``is_socket`` so non-socket events
        always pass through.  Redis suppression should follow the same
        pattern — only suppress socket-level events, not file I/O.
        """
        from unittest.mock import patch as mock_patch

        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)

        file_event = PreloadIOEvent(kind="write", resource_id="file:/tmp/log.txt", fd=7, pid=1, tid=1000)

        with mock_patch("frontrun._dpor_runtime.preload_bridge.is_redis_tid_suppressed", return_value=True):
            bridge.listener(file_event)

        events = bridge.drain(0)
        assert len(events) == 1, "File I/O from Redis-suppressed thread should NOT be dropped"


# ---------------------------------------------------------------------------
# Integration: DPOR detects C-level file I/O races via the preload bridge
# ---------------------------------------------------------------------------


class TestRowLockRegistry:
    """Unit tests for DporScheduler.acquire_row_locks / release_row_locks."""

    def _make_scheduler(self) -> Any:
        """Create a minimal object with the real row-lock methods from DporScheduler."""
        from frontrun._cooperative import real_condition, real_lock
        from frontrun._dpor_core import ReplayEngine, ReplayExecution, RowLockRegistry
        from frontrun.dpor import DporScheduler

        class RowLockHost:
            """Lightweight host carrying only the row-lock subset of DporScheduler."""

            def __init__(self) -> None:
                self.deadlock_timeout = 1.0
                self._lock = real_lock()
                self._condition = real_condition(self._lock)
                self._engine_lock = real_lock()
                self.engine = ReplayEngine()
                self.execution = ReplayExecution()
                self._finished = False
                self._error: Exception | None = None
                self._current_thread: int | None = None
                self._row_lock_registry = RowLockRegistry()
                self._active_row_locks: dict[str, int] = self._row_lock_registry._active_row_locks
                self._thread_row_locks: dict[int, set[str]] = self._row_lock_registry._task_row_locks
                self._row_lock_ids: dict[str, int] = self._row_lock_registry._row_lock_ids
                self._row_lock_blocked: dict[int, int] = {}

            # Bind the real methods from DporScheduler so tests exercise production code.
            acquire_row_locks = DporScheduler.acquire_row_locks
            release_row_locks = DporScheduler.release_row_locks
            _release_row_locks_unlocked = DporScheduler._release_row_locks_unlocked
            _row_lock_int_id = DporScheduler._row_lock_int_id
            _engine_block_thread = DporScheduler._engine_block_thread
            _engine_unblock_thread = DporScheduler._engine_unblock_thread

        return RowLockHost()

    def test_acquire_single_lock(self) -> None:
        """A single thread can acquire a row lock."""
        sched = self._make_scheduler()
        sched.acquire_row_locks(0, ["sql:users:(('id', 42))"])
        assert sched._active_row_locks.get("sql:users:(('id', 42))") == 0

    def test_same_thread_reacquire(self) -> None:
        """The same thread re-acquiring a lock it already holds is a no-op."""
        sched = self._make_scheduler()
        sched.acquire_row_locks(0, ["sql:users:(('id', 42))"])
        sched.acquire_row_locks(0, ["sql:users:(('id', 42))"])
        assert sched._active_row_locks.get("sql:users:(('id', 42))") == 0

    def test_release_unlocks_waiters(self) -> None:
        """Releasing a lock unblocks a waiting thread."""
        import threading

        sched = self._make_scheduler()
        sched.acquire_row_locks(0, ["sql:users:(('id', 42))"])

        acquired = threading.Event()
        error: list[Exception] = []

        def waiter() -> None:
            try:
                sched.acquire_row_locks(1, ["sql:users:(('id', 42))"])
                acquired.set()
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=waiter)
        t.start()

        # Give the waiter time to block
        import time

        time.sleep(0.1)
        assert not acquired.is_set(), "Waiter should still be blocked"

        sched.release_row_locks(0)
        t.join(timeout=2.0)
        assert acquired.is_set(), "Waiter should have acquired after release"
        assert sched._active_row_locks.get("sql:users:(('id', 42))") == 1

    def test_release_no_locks_held(self) -> None:
        """Releasing when no locks are held is a no-op."""
        sched = self._make_scheduler()
        sched.release_row_locks(0)  # Should not raise

    def test_multiple_resources(self) -> None:
        """acquire_row_locks handles multiple resource IDs."""
        sched = self._make_scheduler()
        sched.acquire_row_locks(0, ["sql:users:(('id', 1))", "sql:users:(('id', 2))"])
        assert sched._active_row_locks.get("sql:users:(('id', 1))") == 0
        assert sched._active_row_locks.get("sql:users:(('id', 2))") == 0

    def test_timeout_on_held_lock(self) -> None:
        """acquire_row_locks times out when the holder doesn't release."""
        sched = self._make_scheduler()
        sched.deadlock_timeout = 0.2  # Fast timeout for test
        sched.acquire_row_locks(0, ["sql:users:(('id', 42))"])

        # Thread 1 tries to acquire, should timeout
        sched.acquire_row_locks(1, ["sql:users:(('id', 42))"])
        # After timeout, thread 0 still holds the lock
        assert sched._active_row_locks.get("sql:users:(('id', 42))") == 0

    def _make_scheduler_with_graph(self) -> Any:
        """RowLockHost with WaitForGraph integration fields bound."""
        from frontrun._cooperative import real_condition, real_lock
        from frontrun._dpor_core import ReplayEngine, ReplayExecution, RowLockRegistry
        from frontrun.dpor import DporScheduler

        class RowLockHost:
            def __init__(self) -> None:
                self.deadlock_timeout = 2.0
                self._lock = real_lock()
                self._condition = real_condition(self._lock)
                self._engine_lock = real_lock()
                self.engine = ReplayEngine()
                self.execution = ReplayExecution()
                self._finished = False
                self._error: Exception | None = None
                self._current_thread: int | None = None
                self._row_lock_registry = RowLockRegistry()
                self._active_row_locks: dict[str, int] = self._row_lock_registry._active_row_locks
                self._thread_row_locks: dict[int, set[str]] = self._row_lock_registry._task_row_locks
                self._row_lock_ids: dict[str, int] = self._row_lock_registry._row_lock_ids
                self._row_lock_blocked: dict[int, int] = {}

            acquire_row_locks = DporScheduler.acquire_row_locks
            release_row_locks = DporScheduler.release_row_locks
            _release_row_locks_unlocked = DporScheduler._release_row_locks_unlocked
            _row_lock_int_id = DporScheduler._row_lock_int_id
            _engine_block_thread = DporScheduler._engine_block_thread
            _engine_unblock_thread = DporScheduler._engine_unblock_thread

        return RowLockHost()

    def test_deadlock_detected_via_wait_for_graph(self) -> None:
        """Two-thread row-lock deadlock is detected instantly via WaitForGraph."""
        import threading
        import time

        from frontrun._deadlock import (
            DeadlockError,
            SchedulerAbort,
            install_wait_for_graph,
            uninstall_wait_for_graph,
        )

        install_wait_for_graph()
        try:
            sched = self._make_scheduler_with_graph()

            t0_has_x = threading.Event()
            t1_has_y = threading.Event()
            t0_waiting_for_y = threading.Event()
            errors: list[Exception] = []

            def t0() -> None:
                try:
                    sched.acquire_row_locks(0, ["row_X"])
                    t0_has_x.set()
                    t1_has_y.wait()
                    t0_waiting_for_y.set()
                    sched.acquire_row_locks(0, ["row_Y"])  # blocks; killed by deadlock signal
                except SchedulerAbort:
                    pass
                except Exception as e:
                    errors.append(e)

            def t1() -> None:
                try:
                    sched.acquire_row_locks(1, ["row_Y"])
                    t1_has_y.set()
                    t0_has_x.wait()
                    t0_waiting_for_y.wait()
                    time.sleep(0.05)  # let T0 reach condition.wait
                    sched.acquire_row_locks(1, ["row_X"])  # should detect cycle
                except SchedulerAbort:
                    pass
                except Exception as e:
                    errors.append(e)

            ta = threading.Thread(target=t0)
            tb = threading.Thread(target=t1)
            ta.start()
            tb.start()
            ta.join(timeout=5.0)
            tb.join(timeout=5.0)

            assert not errors, f"Unexpected errors: {errors}"
            assert isinstance(sched._error, DeadlockError), f"Expected DeadlockError, got {sched._error!r}"
            assert "row_X" in sched._error.cycle_description or "row_Y" in sched._error.cycle_description
        finally:
            uninstall_wait_for_graph()

    def test_holding_edges_removed_on_release(self) -> None:
        """_release_row_locks_unlocked removes holding edges from WaitForGraph."""
        from frontrun._deadlock import install_wait_for_graph, uninstall_wait_for_graph

        install_wait_for_graph()
        try:
            sched = self._make_scheduler_with_graph()
            sched.acquire_row_locks(0, ["row_A"])
            lid = sched._row_lock_ids.get("row_A")
            assert lid is not None

            # Verify holding edge exists: ("row_lock", lid) -> ("thread", 0)
            from frontrun._deadlock import get_wait_for_graph

            graph = get_wait_for_graph()
            assert graph is not None
            assert ("thread", 0) in graph._edges.get(("row_lock", lid), set())

            # Release and verify edge is gone
            sched.release_row_locks(0)
            assert ("thread", 0) not in graph._edges.get(("row_lock", lid), set())
        finally:
            uninstall_wait_for_graph()


class TestPreloadDporIntegration:
    """End-to-end: LD_PRELOAD events → IOEventDispatcher → _PreloadBridge → DPOR engine.

    These tests use ``os.open()``/``os.read()``/``os.write()`` (raw POSIX
    file descriptors) instead of ``builtins.open()``.  The Python-level
    I/O monkey-patches in ``_io_detection.py`` do NOT intercept these calls.
    The ONLY way DPOR can see this I/O is through the LD_PRELOAD library
    (which intercepts libc ``read()``, ``write()``, ``close()``) via the
    ``IOEventDispatcher`` → ``_PreloadBridge`` pipeline.

    This proves the integration works for C extensions like psycopg2
    that call libc directly without going through Python's socket module.
    """

    def test_c_level_file_lost_update(self) -> None:
        """Lost update on a file counter using raw POSIX I/O (os.read/write).

        Two threads each read-modify-write a counter stored in a file.
        The individual read and write are atomic (via a lock), but the
        compound sequence is not.  The LD_PRELOAD library intercepts the
        C-level read()/write() calls and the preload bridge routes them
        to DPOR, which detects the write-write conflict and explores the
        interleaving that causes the lost update.
        """
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "counter.txt").encode()

        class State:
            def __init__(self) -> None:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                os.write(fd, b"0")
                os.close(fd)

        def increment(state: State) -> None:
            # Read current value using raw POSIX I/O
            fd = os.open(path, os.O_RDONLY)
            data = os.read(fd, 100)
            os.close(fd)
            val = int(data) if data else 0
            # Write incremented value using raw POSIX I/O
            fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
            os.write(fd, str(val + 1).encode())
            os.close(fd)

        def check_invariant(state: State) -> bool:
            fd = os.open(path, os.O_RDONLY)
            data = os.read(fd, 100)
            os.close(fd)
            return data != b"" and int(data) == 2

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=check_invariant,
            detect_io=True,
            max_executions=30,
            reproduce_on_failure=0,
        )
        # DPOR should detect the lost-update race: the LD_PRELOAD bridge
        # is the ONLY source of I/O events here (os.read/write bypass
        # the Python-level patches).
        assert not result.property_holds, (
            f"DPOR should detect lost-update via LD_PRELOAD C-level I/O; explored {result.num_explored} interleavings"
        )

    def test_c_level_file_no_race_when_locked(self) -> None:
        """Locked read-modify-write using raw POSIX I/O — no race."""
        import threading

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "counter.txt").encode()

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                os.write(fd, b"0")
                os.close(fd)

        def increment(state: State) -> None:
            with state.lock:
                fd = os.open(path, os.O_RDONLY)
                data = os.read(fd, 100)
                os.close(fd)
                val = int(data)
                fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
                os.write(fd, str(val + 1).encode())
                os.close(fd)

        def check_invariant(state: State) -> bool:
            fd = os.open(path, os.O_RDONLY)
            data = os.read(fd, 100)
            os.close(fd)
            return int(data) == 2

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=check_invariant,
            detect_io=True,
            max_executions=30,
            reproduce_on_failure=0,
        )
        assert result.property_holds, (
            f"Lock-protected counter should not have a race; explanation: {result.explanation}"
        )
