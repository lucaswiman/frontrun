"""Tests for connection pooling awareness (reset_connection_state)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from frontrun._io_detection import _io_tls
from frontrun._sql_cursor import _intercept_execute, reset_connection_state
from frontrun._sql_insert_tracker import clear_insert_tracker, record_insert, resolve_alias


@pytest.fixture
def reporter():
    m = Mock()
    with patch("frontrun._sql_cursor.get_io_reporter", return_value=m):
        yield m


@pytest.fixture(autouse=True)
def clean_tls():
    clear_insert_tracker()
    for attr in ("_in_transaction", "_tx_buffer", "_tx_savepoints", "_sql_suppress"):
        if hasattr(_io_tls, attr):
            delattr(_io_tls, attr)
    yield
    clear_insert_tracker()
    for attr in ("_in_transaction", "_tx_buffer", "_tx_savepoints", "_sql_suppress"):
        if hasattr(_io_tls, attr):
            delattr(_io_tls, attr)


class MockCursor:
    def execute(self, op, params=None):
        pass


class TestResetConnectionState:
    def test_clears_in_transaction(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        assert getattr(_io_tls, "_in_transaction", False) is True

        reset_connection_state()
        assert getattr(_io_tls, "_in_transaction", False) is False

    def test_clears_tx_buffer(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        assert len(getattr(_io_tls, "_tx_buffer", [])) > 0

        reset_connection_state()
        assert not hasattr(_io_tls, "_tx_buffer")

    def test_clears_savepoints(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        assert "sp1" in getattr(_io_tls, "_tx_savepoints", {})

        reset_connection_state()
        assert not hasattr(_io_tls, "_tx_savepoints")

    def test_noop_when_clean(self):
        """reset_connection_state should be safe to call even when no transaction is active."""
        reset_connection_state()  # should not raise

    def test_subsequent_operations_work_after_reset(self, reporter):
        """After reset, new SQL operations should work normally (not buffered)."""
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        reporter.assert_not_called()  # buffered in transaction

        reset_connection_state()

        # Now a new INSERT outside transaction should report immediately
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t2 (id) VALUES (2)")
        reporter.assert_called()

    def test_leaked_transaction_cleared_on_pool_return(self, reporter):
        """Simulate pool return: transaction started but never committed/rolled back."""
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "UPDATE accounts SET balance = 100 WHERE id = 1")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        _intercept_execute(cursor.execute, cursor, "DELETE FROM logs WHERE id = 5")

        # Simulate pool checkin
        reset_connection_state()

        # Verify clean state
        assert getattr(_io_tls, "_in_transaction", False) is False
        assert not hasattr(_io_tls, "_tx_buffer")
        assert not hasattr(_io_tls, "_tx_savepoints")

        # Next user of the connection gets a clean state
        reporter.reset_mock()
        _intercept_execute(cursor.execute, cursor, "SELECT * FROM users WHERE id = 1")
        # SELECT reports database intent + table-level read + :seq read for
        # phantom detection.
        assert reporter.call_count == 3

    def test_reset_does_not_clear_insert_aliases_from_other_logical_sessions(self, reporter):
        """Pool reset is per-connection state; it must not wipe global INSERT alias tracking."""
        record_insert("users", 42)
        assert resolve_alias("users", 42) == "sql:users:setup_ins0"

        reset_connection_state()

        assert resolve_alias("users", 42) == "sql:users:setup_ins0"


class TestSavepointDispatch:
    """Pins the cursor-level dispatch behavior for SAVEPOINT / ROLLBACK TO / RELEASE.

    These exercise the three tagged-operation branches in _intercept_execute so the
    behavior is locked down before/after refactoring the dataclass shapes.
    """

    def test_savepoint_records_buffer_offset(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")

        savepoints = getattr(_io_tls, "_tx_savepoints", {})
        buffer = getattr(_io_tls, "_tx_buffer", [])
        assert "sp1" in savepoints
        assert savepoints["sp1"] == len(buffer)

    def test_rollback_to_truncates_buffer(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        buffer_at_sp = list(getattr(_io_tls, "_tx_buffer", []))
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t2 (id) VALUES (2)")
        assert len(getattr(_io_tls, "_tx_buffer", [])) > len(buffer_at_sp)

        _intercept_execute(cursor.execute, cursor, "ROLLBACK TO SAVEPOINT sp1")
        assert list(getattr(_io_tls, "_tx_buffer", [])) == buffer_at_sp

    def test_release_removes_savepoint(self, reporter):
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        assert "sp1" in getattr(_io_tls, "_tx_savepoints", {})

        _intercept_execute(cursor.execute, cursor, "RELEASE SAVEPOINT sp1")
        assert "sp1" not in getattr(_io_tls, "_tx_savepoints", {})

    def test_rollback_to_invalidates_nested_savepoints(self, reporter):
        """ROLLBACK TO sp1 should destroy savepoints created after sp1."""
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t2 (id) VALUES (2)")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp2")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t3 (id) VALUES (3)")

        _intercept_execute(cursor.execute, cursor, "ROLLBACK TO SAVEPOINT sp1")

        savepoints = getattr(_io_tls, "_tx_savepoints", {})
        assert "sp1" in savepoints, "sp1 should still exist"
        assert "sp2" not in savepoints, "sp2 should be destroyed by ROLLBACK TO sp1"

    def test_rollback_to_stale_nested_savepoint_safe(self, reporter):
        """After ROLLBACK TO sp1, ROLLBACK TO sp2 must not corrupt the buffer."""
        cursor = MockCursor()
        _intercept_execute(cursor.execute, cursor, "BEGIN")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp1")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t1 (id) VALUES (1)")
        _intercept_execute(cursor.execute, cursor, "SAVEPOINT sp2")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t2 (id) VALUES (2)")

        _intercept_execute(cursor.execute, cursor, "ROLLBACK TO SAVEPOINT sp1")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t3 (id) VALUES (3)")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t4 (id) VALUES (4)")
        _intercept_execute(cursor.execute, cursor, "INSERT INTO t5 (id) VALUES (5)")

        buf_before = list(getattr(_io_tls, "_tx_buffer", []))

        _intercept_execute(cursor.execute, cursor, "ROLLBACK TO SAVEPOINT sp2")
        buf_after = list(getattr(_io_tls, "_tx_buffer", []))
        assert buf_after == buf_before, "ROLLBACK TO destroyed sp2; buffer should be unchanged"
