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


# ---------------------------------------------------------------------------
# Integration: DPOR detects C-level file I/O races via the preload bridge
# ---------------------------------------------------------------------------


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
            val = int(data)
            # Write incremented value using raw POSIX I/O
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
