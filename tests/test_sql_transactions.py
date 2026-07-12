"""Tests for SQL transaction state management (_sql_transactions.py).

Covers:
- handle_connection_commit / handle_connection_rollback edge cases
"""

from __future__ import annotations

import pytest

from frontrun._io_detection import set_io_reporter, tx_store
from frontrun._sql_cursor import _run_connection_tx_method
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
