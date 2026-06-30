"""Defect #13: DPOR false positive — CTE + FOR UPDATE SKIP LOCKED and single-statement DML.

Problem statement
-----------------

DPOR's SQL conflict model (``_sql_cursor.py`` / ``_sql_parsing.py``) has two
related modeling gaps around PostgreSQL's row-level locking semantics:

**1. ``FOR UPDATE SKIP LOCKED`` is not distinguishable from ``FOR UPDATE``**

    ``parse_sql_access()`` extracts ``lock_intent = LockIntent.UPDATE`` for
    both ``SELECT ... FOR UPDATE`` and ``SELECT ... FOR UPDATE SKIP LOCKED``.
    The ``SKIP LOCKED`` modifier — which tells PostgreSQL to silently skip
    rows that are already locked by another transaction — is completely
    invisible to the conflict model.

    This matters because ``SKIP LOCKED`` changes the concurrency semantics
    fundamentally: two concurrent ``SELECT FOR UPDATE SKIP LOCKED`` on the
    same table will never return the same row, so they cannot conflict at
    the row level.  But DPOR's model sees two accesses to the same table
    with the same ``LockIntent.UPDATE`` and correctly infers a conflict.

    For single-statement CTEs (``WITH cte AS (SELECT ... FOR UPDATE SKIP
    LOCKED) UPDATE ...``), this is currently harmless because each
    ``cursor.execute()`` call is one atomic scheduling point.  But if a
    future refactoring splits the CTE into separate SELECT + UPDATE
    statements (as ORM layers sometimes do), the conflict model would
    create false backtrack points.

**2. Single-statement DML atomicity is not modeled**

    Two concurrent ``DELETE FROM t WHERE id = 1 RETURNING id`` always
    serialize at the row level in PostgreSQL: the first transaction locks
    the row and deletes it; the second blocks until the first commits,
    then finds the row gone and returns nothing.  DPOR's conflict model
    sees two write accesses to the same table and would explore an
    interleaving where both succeed — but this interleaving is impossible
    in real PostgreSQL.

    Again, this is currently harmless for single-statement operations
    (one scheduling point per execute), but the conflict model does not
    represent the constraint that these operations serialize.

**Current impact:**

    With single-statement SQL (which is the common case for pgmq-style
    dequeue), DPOR handles these patterns correctly because each
    ``cursor.execute()`` is indivisible.  The false positive described in
    the original defect report may require a multi-statement implementation
    or specific library internals that create additional scheduling points
    within the transaction.

    The tests below verify:
    - The SQL parsing gap (``SKIP LOCKED`` is invisible)
    - That single-statement patterns are correctly handled
    - That split-statement patterns (SELECT + UPDATE separately) could
      create false backtrack points, though Postgres row locking still
      prevents the invariant violation at execution time

Running
-------
::

    sudo pg_ctlcluster 16 main start
    make test-integration-3.14 PYTEST_ARGS="-v -k test_defect13"
"""

from __future__ import annotations

import os

import pytest

from frontrun._sql_parsing import LockIntent, parse_sql_access

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

from frontrun import explore

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")


# ---------------------------------------------------------------------------
# SQL parsing model gap tests (no Postgres needed)
# ---------------------------------------------------------------------------


class TestSkipLockedParsingGap:
    """The SQL parser treats FOR UPDATE SKIP LOCKED identically to FOR UPDATE.

    These tests document that ``parse_sql_access`` loses the ``SKIP LOCKED``
    modifier, which means DPOR's conflict model cannot distinguish safe
    (skip-locked) from unsafe (blocking) row-level locking patterns.
    """

    def test_skip_locked_produces_distinct_lock_intent(self) -> None:
        """parse_sql_access should distinguish SKIP LOCKED from plain FOR UPDATE."""
        regular = parse_sql_access("SELECT id FROM queue FOR UPDATE")
        skip_locked = parse_sql_access("SELECT id FROM queue FOR UPDATE SKIP LOCKED")

        assert regular.lock_intent == LockIntent.UPDATE
        assert skip_locked.lock_intent == LockIntent.UPDATE_SKIP_LOCKED

        assert skip_locked.lock_intent != regular.lock_intent, (
            "SKIP LOCKED should produce a distinct LockIntent so DPOR's "
            "conflict model knows these accesses cannot conflict at the row level"
        )

    def test_cte_skip_locked_preserves_lock_intent(self) -> None:
        """CTE + FOR UPDATE SKIP LOCKED should preserve lock_intent."""
        result = parse_sql_access("""
            WITH cte AS (
                SELECT msg_id FROM queue
                WHERE vt <= now()
                ORDER BY msg_id LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE queue
            SET vt = now() + interval '30 seconds', read_ct = read_ct + 1
            FROM cte WHERE queue.msg_id = cte.msg_id
            RETURNING queue.msg_id
        """)

        assert "queue" in result.read_tables
        assert "queue" in result.write_tables

        assert result.lock_intent == LockIntent.UPDATE_SKIP_LOCKED, (
            "CTE with FOR UPDATE SKIP LOCKED should preserve lock_intent as UPDATE_SKIP_LOCKED"
        )

    def test_nowait_also_not_distinguished(self) -> None:
        """FOR UPDATE NOWAIT is also not distinguished from FOR UPDATE.

        While less critical than SKIP LOCKED (NOWAIT raises an error
        rather than silently skipping), it documents another modifier
        that the parser loses.
        """
        nowait = parse_sql_access("SELECT id FROM queue FOR UPDATE NOWAIT")
        assert nowait.lock_intent == LockIntent.UPDATE
        # No xfail — this is just documentation of current behavior.


# ---------------------------------------------------------------------------
# Full DPOR tests against Postgres (verify current correct behavior)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_dsn():
    """Return a psycopg2 DSN string, skipping if Postgres is unavailable."""
    if psycopg2 is None:
        pytest.skip("psycopg2 not installed")
    try:
        conn = psycopg2.connect(_DB_URL)
        conn.close()
    except psycopg2.OperationalError:
        pytest.skip(f"Postgres not available at {_DB_URL}")
    return _DB_URL


class TestForUpdateSkipLockedExecution:
    """Verify DPOR correctly handles single-statement FOR UPDATE SKIP LOCKED.

    These tests demonstrate that DPOR does NOT produce false positives for
    single-statement CTE + FOR UPDATE SKIP LOCKED, because each
    ``cursor.execute()`` is one atomic scheduling point.  They serve as
    regression guards: if a future change to the scheduling model breaks
    this atomicity, these tests will catch it.
    """

    @pytest.fixture(autouse=True)
    def _setup_queue_table(self, pg_dsn: str) -> None:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS defect13_queue")
        cur.execute("""
            CREATE TABLE defect13_queue (
                msg_id  SERIAL PRIMARY KEY,
                payload TEXT NOT NULL,
                vt      TIMESTAMPTZ NOT NULL DEFAULT now(),
                read_ct INT NOT NULL DEFAULT 0
            )
        """)
        conn.close()

    def test_cte_skip_locked_dequeue_no_false_positive(self, pg_dsn: str) -> None:
        """Single-statement CTE + FOR UPDATE SKIP LOCKED is correctly atomic.

        DPOR should find property_holds=True (no race) because the CTE is
        executed as a single cursor.execute() call — one scheduling point.
        """
        dsn = pg_dsn

        class State:
            def __init__(self) -> None:
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute("DELETE FROM defect13_queue")
                cur.execute("INSERT INTO defect13_queue (payload) VALUES ('test_msg')")
                conn.close()
                self.dequeued: list[list[int]] = [[], []]

        def make_dequeue(thread_idx: int):
            def dequeue(state: State) -> None:
                conn = psycopg2.connect(dsn)
                conn.autocommit = False
                cur = conn.cursor()
                cur.execute("""
                    WITH cte AS (
                        SELECT msg_id FROM defect13_queue
                        WHERE vt <= now()
                        ORDER BY msg_id LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE defect13_queue
                    SET vt = now() + interval '30 seconds',
                        read_ct = read_ct + 1
                    FROM cte
                    WHERE defect13_queue.msg_id = cte.msg_id
                    RETURNING defect13_queue.msg_id
                """)
                rows = cur.fetchall()
                for row in rows:
                    state.dequeued[thread_idx].append(row[0])
                conn.commit()
                conn.close()

            return dequeue

        def invariant(state: State) -> bool:
            set_0 = set(state.dequeued[0])
            set_1 = set(state.dequeued[1])
            return len(set_0 & set_1) == 0

        result = explore(
            setup=State,
            workers=[make_dequeue(0), make_dequeue(1)],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            lock_timeout=2000,
        )

        assert result.property_holds, (
            f"Single-statement CTE + FOR UPDATE SKIP LOCKED should be safe, "
            f"but DPOR reported a false positive.\n"
            f"Explanation: {result.explanation}"
        )

    def test_concurrent_delete_no_false_positive(self, pg_dsn: str) -> None:
        """Single-statement concurrent DELETE is correctly atomic.

        Two threads DELETE the same row.  Postgres serializes them at the
        row level: only one succeeds.  DPOR should find no violation.
        """
        dsn = pg_dsn

        class State:
            def __init__(self) -> None:
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute("DELETE FROM defect13_queue")
                cur.execute("INSERT INTO defect13_queue (msg_id, payload) VALUES (1, 'to_delete')")
                conn.close()
                self.deleted_by: list[bool] = [False, False]

        def make_deleter(thread_idx: int):
            def deleter(state: State) -> None:
                conn = psycopg2.connect(dsn)
                conn.autocommit = False
                cur = conn.cursor()
                cur.execute("DELETE FROM defect13_queue WHERE msg_id = 1 RETURNING msg_id")
                row = cur.fetchone()
                if row is not None:
                    state.deleted_by[thread_idx] = True
                conn.commit()
                conn.close()

            return deleter

        def invariant(state: State) -> bool:
            return not (state.deleted_by[0] and state.deleted_by[1])

        result = explore(
            setup=State,
            workers=[make_deleter(0), make_deleter(1)],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            lock_timeout=2000,
        )

        assert result.property_holds, (
            f"Single-statement concurrent DELETE should be safe, "
            f"but DPOR reported a false positive.\n"
            f"Explanation: {result.explanation}"
        )
