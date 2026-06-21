"""Tests for SQL transaction state management (_sql_transactions.py).

Covers:
- handle_connection_commit / handle_connection_rollback edge cases
"""

from __future__ import annotations

from frontrun._io_detection import set_io_reporter, tx_store
from frontrun._sql_transactions import handle_connection_commit


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
