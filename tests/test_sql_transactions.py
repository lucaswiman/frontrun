"""Tests for SQL transaction state management (_sql_transactions.py).

Covers:
- handle_connection_commit / handle_connection_rollback edge cases
"""

from __future__ import annotations

import pytest

from frontrun._io_detection import set_io_reporter, tx_store
from frontrun._sql_cursor import _intercept_execute, _run_connection_tx_method
from frontrun._sql_transactions import handle_connection_commit


@pytest.mark.parametrize("operation", ["COMMIT", "ROLLBACK"])
def test_failed_connection_tx_end_preserves_modeled_transaction(operation: str) -> None:
    """A failed physical tx end leaves the real transaction active or poisoned."""
    store = tx_store()
    store._in_transaction = True
    store._is_autobegin = True
    store._tx_buffer = [("sql:accounts", "write")]
    store._tx_savepoints = {"before": 0}
    store._held_row_locks = {"sql:accounts:id=1"}

    def fail() -> None:
        raise RuntimeError(f"physical {operation.lower()} failed")

    try:
        with pytest.raises(RuntimeError, match="physical"):
            _run_connection_tx_method(fail, operation)

        assert store._in_transaction is True
        assert store._is_autobegin is True
        assert store._tx_buffer == [("sql:accounts", "write")]
        assert store._tx_savepoints == {"before": 0}
        assert store._held_row_locks == {"sql:accounts:id=1"}
    finally:
        for attr in ("_in_transaction", "_is_autobegin", "_tx_buffer", "_tx_savepoints", "_held_row_locks"):
            if hasattr(store, attr):
                delattr(store, attr)


class TestHandleConnectionCommitReporterNone:
    """handle_connection_commit crashes when the reporter is None at commit time.

    When _in_transaction is True (set during BEGIN with a reporter active) but
    get_io_reporter() returns None at commit time, _handle_tx_op(None, COMMIT)
    crashes at line 167 calling None(res_id, kind) while flushing the buffer.

    The function should handle this gracefully instead of crashing.
    """

    def test_commit_with_none_reporter_and_buffered_items(self) -> None:
        """Should not crash when reporter is None but transaction has buffered items."""
        store = tx_store()
        # Simulate: a transaction was started when a reporter was active,
        # buffered some accesses, then the reporter was cleared before commit.
        store._in_transaction = True
        store._is_autobegin = False
        store._tx_buffer = [("sql:users", "write"), ("sql:orders", "read")]
        store._tx_savepoints = {}

        # Clear the reporter so get_io_reporter() returns None
        set_io_reporter(None)

        try:
            # This should not crash — but currently does because
            # _handle_tx_op calls reporter(res_id, kind) where reporter is None
            handle_connection_commit()
        finally:
            # Clean up transaction state regardless
            store._in_transaction = False
            store._tx_buffer = []
            store._tx_savepoints = {}


@pytest.fixture
def transaction_cursor():
    from unittest.mock import Mock

    from frontrun._io_detection import _io_tls

    reporter = Mock()
    set_io_reporter(reporter)
    for attr in ("_in_transaction", "_tx_buffer", "_tx_savepoints"):
        if hasattr(_io_tls, attr):
            delattr(_io_tls, attr)

    class Cursor:
        pass

    try:
        yield reporter, Cursor(), lambda self, op, params=None: None
    finally:
        set_io_reporter(None)


class TestInterceptedTransactionBoundaries:
    def test_transaction_grouping_begin_commit(self, transaction_cursor):
        (reporter, cursor, mock_orig) = transaction_cursor
        _intercept_execute(mock_orig, cursor, "BEGIN")
        reporter.assert_not_called()
        _intercept_execute(mock_orig, cursor, "SELECT * FROM accounts WHERE id = 1")
        reporter.assert_not_called()
        _intercept_execute(mock_orig, cursor, "UPDATE accounts SET balance = 0 WHERE id = 1")
        reporter.assert_not_called()
        _intercept_execute(mock_orig, cursor, "COMMIT")
        expected_id = "sql:accounts:(('id', '1'),)"
        reporter.assert_any_call(expected_id, "read")
        reporter.assert_any_call(expected_id, "write")

    def test_savepoint_tracking(self, transaction_cursor):
        (reporter, cursor, mock_orig) = transaction_cursor
        _intercept_execute(mock_orig, cursor, "BEGIN")
        _intercept_execute(mock_orig, cursor, "UPDATE t1 SET x=1")
        _intercept_execute(mock_orig, cursor, "SAVEPOINT sp1")
        _intercept_execute(mock_orig, cursor, "UPDATE t2 SET x=2")
        _intercept_execute(mock_orig, cursor, "ROLLBACK TO SAVEPOINT sp1")
        _intercept_execute(mock_orig, cursor, "COMMIT")
        reporter.assert_any_call("sql:t1", "write")
        for call in reporter.call_args_list:
            assert "sql:t2" not in call.args[0]

    def test_rollback_transaction_boundary(self, transaction_cursor):
        (reporter, cursor, mock_orig) = transaction_cursor
        _intercept_execute(mock_orig, cursor, "BEGIN")
        _intercept_execute(mock_orig, cursor, "DELETE FROM sensitive_data WHERE id = 1")
        _intercept_execute(mock_orig, cursor, "ROLLBACK")
        reporter.assert_not_called()
