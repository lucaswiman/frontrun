"""Tests for DPOR phantom read detection (SELECT + INSERT conflicts).

The bug: DPOR uses row-level (or table-level fallback) conflict tracking.
When thread A performs a SELECT that returns existing rows (or checks a COUNT),
and thread B performs an INSERT that adds a new row to the same table,
DPOR does not detect a conflict if the SELECT falls back to table-level but
the INSERT has row-level predicates from its VALUES clause.

The SELECT reports READ on ``sql:<table>`` (table-level), while the INSERT
reports WRITE on ``sql:<table>:(('col','val'),...)`` (row-level) plus WRITE
on ``sql:<table>:seq`` and ``sql:<table>:t<N>_ins<M>`` (indexical alias).
Since ``sql:<table>`` != ``sql:<table>:(...)`` and ``sql:<table>:seq``,
no resource ID overlaps and DPOR sees no conflict.

This is a classic "phantom read" problem: the SELECT's result (e.g.,
COUNT=0 or EXISTS=False) depends on the *absence* of rows that the INSERT
later creates.

Tests:
  1. Unit test — verifies that SELECT on a table reports READ on the table's
     sequence resource (sql:<table>:seq), creating a conflict with INSERTs.
  2. Integration test — verifies DPOR detects a check-then-insert TOCTOU race
     (token limit bypass pattern from django-rest-knox).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any
from uuid import uuid4

import frontrun
from frontrun._sql_cursor import patch_sql, unpatch_sql
from frontrun._tracing import _FRONTRUN_DIR

# ---------------------------------------------------------------------------
# Helper: compile a "library ORM" wrapper invisible to the DPOR tracer
# ---------------------------------------------------------------------------

_FAKE_LIB_FILENAME = os.path.join(_FRONTRUN_DIR, "_test_phantom_helper.py")
_PHANTOM_CODE = compile(
    """
def check_and_insert_token(cursor, user_id, token_value, token_limit=1):
    '''Simulate a check-then-insert pattern (django-rest-knox style).

    1. COUNT all tokens (global limit check — no WHERE on user_id).
    2. If below limit, INSERT a new token.

    The SELECT has no WHERE clause, so it falls back to table-level
    conflict tracking (no row-level predicates).  The INSERT has
    row-level predicates from its VALUES clause.  DPOR must still
    detect the conflict.

    Returns True if a token was inserted, False if limit was reached.
    '''
    cursor.execute("SELECT COUNT(*) FROM tokens")
    count = cursor.fetchone()[0]
    if count >= token_limit:
        return False
    cursor.execute(
        "INSERT INTO tokens (user_id, token) VALUES (?, ?)",
        (user_id, token_value),
    )
    return True
""",
    _FAKE_LIB_FILENAME,
    "exec",
)
_phantom_ns: dict[str, Any] = {}
exec(_PHANTOM_CODE, _phantom_ns)  # noqa: S102
_check_and_insert_token = _phantom_ns["check_and_insert_token"]


# ---------------------------------------------------------------------------
# Unit test: SELECT reports READ on table sequence resource
# ---------------------------------------------------------------------------


class TestSelectReadsSequenceResource:
    """SELECT on a table must report READ on sql:<table>:seq to detect phantom reads."""

    def test_select_reports_read_on_seq_resource(self) -> None:
        """A SELECT COUNT on a table should report a READ on the :seq resource.

        This is required so that concurrent INSERTs (which WRITE to :seq)
        create a conflict that DPOR can detect.
        """
        from frontrun._io_detection import set_io_reporter
        from frontrun._sql_cursor import clear_sql_metadata

        clear_sql_metadata()
        patch_sql()
        try:
            db_uri = "file:phantom_unit_test?mode=memory&cache=shared"
            conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            conn.execute("CREATE TABLE IF NOT EXISTS tokens (id INTEGER PRIMARY KEY, user_id INTEGER, token TEXT)")
            conn.execute("DELETE FROM tokens")
            conn.commit()

            events: list[tuple[str, str]] = []

            def reporter(resource_id: str, kind: str) -> None:
                events.append((resource_id, kind))

            set_io_reporter(reporter)
            conn.execute("SELECT COUNT(*) FROM tokens WHERE user_id = 1")
            set_io_reporter(None)

            # The SELECT should report a READ on the :seq resource for phantom detection
            seq_reads = [(r, k) for r, k in events if r.endswith(":seq") and k == "read"]
            assert len(seq_reads) >= 1, (
                f"SELECT should report READ on sql:tokens:...:seq for phantom read detection, but got events: {events}"
            )

            conn.close()
        finally:
            unpatch_sql()

    def test_select_without_where_reports_read_on_seq_resource(self) -> None:
        """A bare SELECT (no WHERE) should also report READ on :seq."""
        from frontrun._io_detection import set_io_reporter
        from frontrun._sql_cursor import clear_sql_metadata

        clear_sql_metadata()
        patch_sql()
        try:
            db_uri = "file:phantom_unit_bare?mode=memory&cache=shared"
            conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            conn.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("DELETE FROM items")
            conn.commit()

            events: list[tuple[str, str]] = []

            def reporter(resource_id: str, kind: str) -> None:
                events.append((resource_id, kind))

            set_io_reporter(reporter)
            conn.execute("SELECT * FROM items")
            set_io_reporter(None)

            seq_reads = [(r, k) for r, k in events if r.endswith(":seq") and k == "read"]
            assert len(seq_reads) >= 1, (
                f"SELECT (no WHERE) should report READ on sql:items:...:seq for phantom read detection, "
                f"but got events: {events}"
            )

            conn.close()
        finally:
            unpatch_sql()


# ---------------------------------------------------------------------------
# Unit test: DELETE reports WRITE on table sequence resource
# ---------------------------------------------------------------------------


class TestDeleteWritesSequenceResource:
    """DELETE on a table must report WRITE on sql:<table>:seq for phantom detection."""

    def test_delete_reports_write_on_seq_resource(self) -> None:
        """DELETE should report a WRITE on the :seq resource.

        This is required so that concurrent SELECTs (which READ :seq)
        create a conflict that DPOR can detect — the DELETE changes
        the set of rows visible to the SELECT.
        """
        from frontrun._io_detection import set_io_reporter
        from frontrun._sql_cursor import clear_sql_metadata

        clear_sql_metadata()
        patch_sql()
        try:
            db_uri = "file:phantom_delete_test?mode=memory&cache=shared"
            conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            conn.execute("CREATE TABLE IF NOT EXISTS tokens (id INTEGER PRIMARY KEY, user_id INTEGER, token TEXT)")
            conn.execute("INSERT INTO tokens VALUES (1, 1, 'abc')")
            conn.commit()

            events: list[tuple[str, str]] = []

            def reporter(resource_id: str, kind: str) -> None:
                events.append((resource_id, kind))

            set_io_reporter(reporter)
            conn.execute("DELETE FROM tokens WHERE id = 1")
            set_io_reporter(None)

            seq_writes = [(r, k) for r, k in events if r.endswith(":seq") and k == "write"]
            assert len(seq_writes) >= 1, (
                f"DELETE should report WRITE on sql:tokens:...:seq for phantom read detection, but got events: {events}"
            )

            conn.close()
        finally:
            unpatch_sql()


# ---------------------------------------------------------------------------
# Integration test: DPOR detects check-then-insert TOCTOU race
# ---------------------------------------------------------------------------


class TestDporPhantomReadDetection:
    """DPOR must detect check-then-insert TOCTOU races (phantom reads)."""

    def test_token_limit_bypass_detected(self) -> None:
        """Two threads doing COUNT + INSERT can bypass a token limit.

        Simulates the django-rest-knox token limit pattern:
        1. Thread A: COUNT active tokens → 0 (below limit)
        2. Thread B: COUNT active tokens → 0 (below limit)
        3. Thread A: INSERT new token
        4. Thread B: INSERT new token → user now has 2 tokens, limit was 1

        DPOR must explore the interleaving where both SELECTs happen before
        either INSERT, detecting the invariant violation.
        """

        db_uri = f"file:phantom_dpor_test_{uuid4().hex}?mode=memory&cache=shared"

        # Keep a connection alive so the shared-cache DB survives across
        # setup/thread/invariant calls.
        _keeper = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
        try:
            _keeper.execute(
                "CREATE TABLE IF NOT EXISTS tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, token TEXT)"
            )

            class State:
                pass

            def thread_fn(state: State) -> None:
                conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
                cur = conn.cursor()
                _check_and_insert_token(cur, user_id=1, token_value=f"tok_{id(state)}")
                conn.close()

            def setup() -> State:
                _keeper.execute("DELETE FROM tokens")
                return State()

            def invariant(state: State) -> bool:
                conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM tokens WHERE user_id = 1")
                count = cur.fetchone()[0]
                conn.close()
                # Token limit is 1 — should never have more than 1 token
                return count <= 1

            result = frontrun.explore(
                setup=setup,
                workers=[thread_fn, thread_fn],
                invariant=invariant,
                detect_io=True,
                reproduce_on_failure=0,
                max_executions=50,
                preemption_bound=2,
            )

            # DPOR must find the interleaving where both threads pass the COUNT
            # check before either INSERT, violating the token limit invariant.
            assert not result.property_holds, (
                f"DPOR should detect the phantom read race (token limit bypass) "
                f"but explored {result.num_explored} interleavings without finding "
                f"an invariant violation. This indicates the SELECT + INSERT "
                f"conflict is not being detected."
            )
        finally:
            _keeper.close()
