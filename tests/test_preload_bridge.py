"""Tests for _PreloadBridge event-kind mapping (LD_PRELOAD -> DPOR access kind)."""

from __future__ import annotations

import threading

from frontrun._preload_io import PreloadIOEvent
from frontrun._sql_endpoint_suppression import clear_permanent_suppressions, suppress_sql_write
from frontrun.dpor import _PreloadBridge


class TestPreloadBridgeKindMapping:
    """The bridge must map libc op kinds to the correct DPOR access kind."""

    def test_sql_write_maps_to_write(self) -> None:
        """A sql_write event is a SQL *send* and must bucket as 'write'.

        Previously the bridge used ``"write" if kind == "write" else "read"``,
        which dropped sql_write into the read bucket — so two threads' SQL
        sends never conflicted through this path.  The resource for a sql_write
        event is raw SQL text (not a ``socket:`` endpoint), so the socket
        suppression filters do not apply.
        """
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)

        ev = PreloadIOEvent(
            kind="sql_write",
            resource_id="UPDATE accounts SET balance = balance - 1 WHERE id = 1",
            fd=5,
            pid=1,
            tid=1000,
        )
        bridge.listener(ev)

        events = bridge.drain(0)
        assert len(events) == 1, "sql_write event should be buffered"
        # tuple is (obj_key, kind, resource_id, detail, call_chain)
        assert events[0][1] == "write", "sql_write must map to DPOR 'write', not 'read'"

    def test_suppressed_sql_write_is_dropped(self) -> None:
        """Raw SQL wire events are redundant once SQL parsing reported rows."""
        clear_permanent_suppressions()
        bridge = _PreloadBridge()
        tid = threading.get_native_id()
        bridge.register_thread(os_tid=tid, dpor_id=0)
        suppress_sql_write("SELECT value FROM accounts WHERE id = 1")

        ev = PreloadIOEvent(
            kind="sql_write",
            resource_id="SELECT value FROM accounts WHERE id = 1",
            fd=5,
            pid=1,
            tid=tid,
        )
        bridge.listener(ev)

        assert bridge.drain(0) == []

    def test_unsuppressed_sql_write_still_buffers_as_fallback(self) -> None:
        clear_permanent_suppressions()
        bridge = _PreloadBridge()
        tid = threading.get_native_id()
        bridge.register_thread(os_tid=tid, dpor_id=0)

        ev = PreloadIOEvent(kind="sql_write", resource_id="CALL unknown_proc()", fd=5, pid=1, tid=tid)
        bridge.listener(ev)

        events = bridge.drain(0)
        assert len(events) == 1
        assert events[0][1] == "write"

    def test_plain_write_still_maps_to_write(self) -> None:
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        ev = PreloadIOEvent(kind="write", resource_id="file:/tmp/x", fd=5, pid=1, tid=1000)
        bridge.listener(ev)
        events = bridge.drain(0)
        assert len(events) == 1
        assert events[0][1] == "write"

    def test_read_still_maps_to_read(self) -> None:
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        ev = PreloadIOEvent(kind="read", resource_id="file:/tmp/x", fd=5, pid=1, tid=1000)
        bridge.listener(ev)
        events = bridge.drain(0)
        assert len(events) == 1
        assert events[0][1] == "read"

    def test_connect_maps_to_read(self) -> None:
        """connect stays a 'read' (a connection is a read-like access point)."""
        bridge = _PreloadBridge()
        bridge.register_thread(os_tid=1000, dpor_id=0)
        ev = PreloadIOEvent(kind="connect", resource_id="socket:127.0.0.1:5432", fd=5, pid=1, tid=1000)
        bridge.listener(ev)
        events = bridge.drain(0)
        assert len(events) == 1
        assert events[0][1] == "read"
