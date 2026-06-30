"""Regression test for DPOR scheduling points at SQL interception.

The bug: _intercept_execute lives in frontrun/ which is skipped by sys.settrace.
When SQL is called from library code (site-packages / frontrun internals), the
tracer never fires between cursor.execute() calls, so pending_io events from
intermediate SQL operations accumulate and are flushed together at the next
user-code opcode.  The Rust engine's record_io_access keeps only the FIRST
event per thread — so the UPDATE write is dropped and only the SELECT read is
recorded.  DPOR sees no write-write conflict and explores only 1 interleaving.

The fix: _intercept_execute calls report_and_wait(None, thread_id) directly,
forcing a scheduling point at each SQL call regardless of whether the caller
is traced user code or opaque library code.

Tests:
  1. Unit test — verifies report_and_wait is called once per reportable SQL op.
  2. Integration test — verifies DPOR explores >1 interleaving when SQL is
     called from untraced "library" code (simulated via a helper compiled with
     a frontrun/ filename so _should_trace_file returns False for it).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from frontrun import explore
from frontrun._sql_cursor import patch_sql, unpatch_sql
from frontrun._tracing import _FRONTRUN_DIR

# ---------------------------------------------------------------------------
# Helper: compile an "ORM wrapper" that looks like library code to the tracer
# ---------------------------------------------------------------------------

# Compile the helper with a filename inside _FRONTRUN_DIR so that
# _should_trace_file() returns False for it — exactly as it would for Django
# code in site-packages.  The tracer will not fire between the cursor.execute()
# calls inside this function.
_FAKE_LIB_FILENAME = os.path.join(_FRONTRUN_DIR, "_test_orm_helper.py")
_ORM_CODE = compile(
    """
def orm_increment(cursor):
    '''Simulate a Django-style ORM method: multiple SQL calls, no user opcodes between them.'''
    cursor.execute("SELECT val FROM counters WHERE id = 1")
    row = cursor.fetchone()
    cursor.execute("UPDATE counters SET val = ? WHERE id = 1", (row[0] + 1,))
""",
    _FAKE_LIB_FILENAME,
    "exec",
)
_orm_ns: dict[str, Any] = {}
exec(_ORM_CODE, _orm_ns)  # noqa: S102
_orm_increment = _orm_ns["orm_increment"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_db() -> sqlite3.Connection:
    """Shared in-memory SQLite DB accessible from multiple connections."""
    uri = "file:dpor_sched_test?mode=memory&cache=shared"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS counters (id INTEGER PRIMARY KEY, val INTEGER)")
    conn.execute("DELETE FROM counters")
    conn.execute("INSERT INTO counters VALUES (1, 0)")
    conn.commit()
    # Keep this connection open so the shared-cache DB stays alive
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Unit test: report_and_wait is called for each reportable SQL operation
# ---------------------------------------------------------------------------


class TestReportAndWaitCalledPerSqlOp:
    """_intercept_execute must call report_and_wait once per reportable SQL op."""

    def test_report_and_wait_called_when_dpor_active(self, mem_db: sqlite3.Connection) -> None:
        """When DPOR is active, each cursor.execute should trigger report_and_wait."""
        from frontrun._io_detection import (
            set_dpor_scheduler,
            set_dpor_thread_id,
            set_io_reporter,
        )

        # Set up a mock scheduler that records report_and_wait calls
        mock_scheduler = MagicMock()
        mock_scheduler.report_and_wait.return_value = True

        set_dpor_scheduler(mock_scheduler)
        set_dpor_thread_id(0)
        set_io_reporter(lambda r, k: None)  # so reported=True

        patch_sql()
        try:
            uri = "file:dpor_sched_test?mode=memory&cache=shared"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT val FROM counters WHERE id = 1")
            cur.execute("UPDATE counters SET val = 2 WHERE id = 1")
            conn.close()
        finally:
            unpatch_sql()
            set_dpor_scheduler(None)
            set_dpor_thread_id(None)
            set_io_reporter(None)

        calls = mock_scheduler.report_and_wait.call_args_list
        assert len(calls) >= 2, (
            f"Expected report_and_wait to be called at least once per SQL operation "
            f"(SELECT + UPDATE = 2), but got {len(calls)} call(s). "
            "The scheduling-point fix in _intercept_execute may have been removed."
        )
        # All calls must use frame=None (not a real frame)
        for call in calls:
            assert call.args[0] is None, (
                f"report_and_wait should be called with frame=None from _intercept_execute, got frame={call.args[0]!r}"
            )

    def test_report_and_wait_not_called_when_dpor_inactive(self, mem_db: sqlite3.Connection) -> None:
        """When DPOR is not active, _intercept_execute must not call report_and_wait."""
        from frontrun._io_detection import set_io_reporter

        set_io_reporter(lambda r, k: None)
        patch_sql()
        try:
            # No dpor scheduler/thread_id set — _get_dpor_context() returns None
            uri = "file:dpor_sched_test?mode=memory&cache=shared"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            cur = conn.cursor()
            # Should not raise; report_and_wait is skipped entirely
            cur.execute("SELECT val FROM counters WHERE id = 1")
        finally:
            unpatch_sql()
            set_io_reporter(None)
        # If we got here without error, the guard worked correctly


# ---------------------------------------------------------------------------
# Integration test: DPOR explores >1 interleaving for untraced library SQL
# ---------------------------------------------------------------------------


class TestDporSqlSchedulingPoints:
    """DPOR must explore >1 interleaving when SQL is called from untraced library code."""

    def test_orm_style_sql_more_interleavings_than_without_sql_events(self, mem_db: sqlite3.Connection) -> None:
        """SQL scheduling points cause DPOR to explore significantly more interleavings.

        _orm_increment is compiled with a filename inside _FRONTRUN_DIR so that
        _should_trace_file() returns False for it.  This simulates Django's
        cursor.execute() calls inside site-packages: the DPOR tracer never fires
        between the SELECT and UPDATE inside _orm_increment.

        With detect_io=True the SQL cursor is patched.  Each cursor.execute()
        call triggers _intercept_execute which:
          - reports the SQL access to the io_reporter (pending_io)
          - [WITH FIX] calls report_and_wait(None, thread_id) to flush pending_io
            immediately and create a scheduling point

        Without the fix, both SQL events are flushed together at the next
        user-code opcode (conn.close()).  record_io_access keeps only the first
        (SELECT/read), dropping the UPDATE/write.  DPOR finds far fewer
        interleavings (the write-write conflict is invisible).

        With the fix, each SQL operation has its own scheduling point, the
        UPDATE/write is recorded separately, and DPOR explores the write-write
        conflict ordering too.  We verify this by checking that detect_io=True
        (SQL patched, scheduling points active) produces more interleavings than
        a baseline where SQL is not patched (detect_io=False, no SQL events).
        """

        class State:
            pass

        def thread_fn(_state: State) -> None:
            # autocommit (isolation_level=None) so each statement commits immediately
            # and SQLite never holds a write lock across DPOR scheduling points.
            uri = "file:dpor_sched_test?mode=memory&cache=shared"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
            cur = conn.cursor()
            _orm_increment(cur)  # SQL happens inside "library code" — tracer skips it
            conn.close()

        # With SQL patching active: each cursor.execute creates a scheduling point
        result_with_sql = explore(
            setup=State,
            workers=[thread_fn, thread_fn],
            invariant=lambda _s: True,
            detect_io=True,
            reproduce_on_failure=0,
            max_executions=50,
            preemption_bound=2,
        )

        # Without SQL patching: cursors go directly to C, no SQL events, fewer points
        # (detect_io=False also disables patch_sql, so _intercept_execute is never called)
        result_without_sql = explore(
            setup=State,
            workers=[thread_fn, thread_fn],
            invariant=lambda _s: True,
            detect_io=False,
            reproduce_on_failure=0,
            max_executions=50,
            preemption_bound=2,
        )

        assert result_with_sql.num_explored > result_without_sql.num_explored, (
            f"Expected SQL scheduling points to enable more interleavings "
            f"(got detect_io=True: {result_with_sql.num_explored}, "
            f"detect_io=False: {result_without_sql.num_explored}). "
            "The report_and_wait fix in _intercept_execute may not be creating "
            "extra scheduling points for untraced ORM-style SQL code."
        )
