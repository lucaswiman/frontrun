"""Tests for defect #6: UPDATE-INSERT phantom race detection within transactions.

DPOR cannot detect races involving the pattern SELECT→UPDATE→INSERT within
a transaction, where two concurrent UPDATEs both match 0 rows and then both
INSERTs create rows that would have matched the UPDATE's WHERE clause.

Root cause (two parts):
1. Row-lock arbitration acquires scheduler-level locks for UPDATEs even when
   0 rows match. In real PostgreSQL, 0-row UPDATEs acquire no row locks.
   This over-serialization forces Thread B to wait for Thread A to commit.
2. UPDATE is excluded from :seq tracking, so DPOR has no conflict arc between
   the UPDATE (which depends on which rows exist) and the INSERT (which
   changes which rows exist).

Fix (two parts):
1. After cursor.execute(), check cursor.rowcount. If 0, release any row locks
   acquired for this UPDATE (matches PostgreSQL semantics).
2. UPDATEs report a READ on sql:<table>:seq, creating a conflict arc with
   INSERT's WRITE on :seq so DPOR explores interleavings where both UPDATEs
   precede both INSERTs.
"""

from __future__ import annotations

import os
import sqlite3
from uuid import uuid4

import pytest

import frontrun
from frontrun._sql_cursor import patch_sql, unpatch_sql

# ---------------------------------------------------------------------------
# Unit test: UPDATE reports READ on :seq resource
# ---------------------------------------------------------------------------


class TestUpdateReadsSequenceResource:
    """UPDATE on a table must report READ on sql:<table>:seq.

    UPDATE results depend on which rows exist (like SELECT), so concurrent
    INSERTs that add rows matching the UPDATE's WHERE clause are phantom reads.
    Without :seq tracking, DPOR misses the conflict.
    """

    def test_update_reports_read_on_seq_resource(self) -> None:
        """UPDATE should report a READ on the :seq resource.

        This creates a conflict arc with INSERT's WRITE on :seq, so DPOR
        knows to explore interleavings where both UPDATEs happen before
        either INSERT.
        """
        from frontrun._io_detection import set_io_reporter
        from frontrun._sql_cursor import clear_sql_metadata

        clear_sql_metadata()
        patch_sql()
        try:
            db_uri = "file:defect6_update_seq_test?mode=memory&cache=shared"
            conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, is_primary BOOLEAN DEFAULT 0)"
            )
            conn.execute("DELETE FROM items")

            events: list[tuple[str, str]] = []

            def reporter(resource_id: str, kind: str) -> None:
                events.append((resource_id, kind))

            set_io_reporter(reporter)
            conn.execute("UPDATE items SET is_primary = 0 WHERE is_primary = 1")
            set_io_reporter(None)

            seq_reads = [(r, k) for r, k in events if r.endswith(":seq") and k == "read"]
            assert len(seq_reads) >= 1, (
                f"UPDATE should report READ on sql:items:...:seq for phantom read detection, but got events: {events}"
            )

            conn.close()
        finally:
            unpatch_sql()


# ---------------------------------------------------------------------------
# Integration test: DPOR detects UPDATE-INSERT phantom race (sqlite, no tx)
# ---------------------------------------------------------------------------


class TestDporUpdateInsertPhantomRace:
    """DPOR must detect UPDATE-INSERT phantom races."""

    def test_update_insert_phantom_race_autocommit(self) -> None:
        """Two threads doing SELECT→UPDATE→INSERT can both INSERT with is_primary=TRUE.

        With autocommit (no transactions), row-lock arbitration is inactive.
        DPOR should detect the race via :seq conflict arcs between UPDATE (READ)
        and INSERT (WRITE).

        Interleaving:
        1. Thread A: SELECT (empty) → UPDATE (0 rows)
        2. Thread B: SELECT (empty) → UPDATE (0 rows)
        3. Thread A: INSERT (is_primary=TRUE)
        4. Thread B: INSERT (is_primary=TRUE)
        → Two rows with is_primary=TRUE (invariant violated)
        """
        db_uri = f"file:defect6_phantom_autocommit_{uuid4().hex}?mode=memory&cache=shared"

        _keeper = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
        try:
            _keeper.execute(
                "CREATE TABLE IF NOT EXISTS domains "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, is_primary INTEGER DEFAULT 0)"
            )

            def setup() -> object:
                _keeper.execute("DELETE FROM domains")
                return object()

            def make_thread_fn(name: str):
                def thread_fn(state: object) -> None:
                    conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
                    cur = conn.cursor()
                    cur.execute("SELECT EXISTS(SELECT 1 FROM domains WHERE is_primary = 1)")
                    cur.fetchone()
                    cur.execute("UPDATE domains SET is_primary = 0 WHERE is_primary = 1")
                    cur.execute(
                        "INSERT INTO domains (name, is_primary) VALUES (?, 1)",
                        (name,),
                    )
                    conn.close()

                return thread_fn

            def invariant(state: object) -> bool:
                conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, isolation_level=None)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM domains WHERE is_primary = 1")
                count = cur.fetchone()[0]
                conn.close()
                return count <= 1

            result = frontrun.explore(
                setup=setup,
                workers=[make_thread_fn("domain_a"), make_thread_fn("domain_b")],
                invariant=invariant,
                detect_io=True,
                reproduce_on_failure=0,
                max_executions=50,
                preemption_bound=2,
            )

            assert not result.property_holds, (
                f"DPOR should detect the UPDATE-INSERT phantom race but explored "
                f"{result.num_explored} interleavings without finding an invariant "
                f"violation. This indicates UPDATE is not generating :seq conflict "
                f"arcs with INSERT."
            )
        finally:
            _keeper.close()


# ---------------------------------------------------------------------------
# Integration test: PostgreSQL transactional UPDATE-INSERT phantom race
# ---------------------------------------------------------------------------


try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = f"dbname={_DB_NAME}"


def _pg_available() -> bool:
    if not _HAS_PSYCOPG2:
        return False
    try:
        conn = psycopg2.connect(_DSN)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def pg_table():
    if not _pg_available():
        pytest.skip("PostgreSQL not available or psycopg2 not installed")
    conn = psycopg2.connect(_DSN)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect6_domains")
        cur.execute("""
            CREATE TABLE defect6_domains (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                is_primary BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
    conn.close()
    yield
    conn = psycopg2.connect(_DSN)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect6_domains")
    conn.close()


class _PgState:
    def __init__(self) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DELETE FROM defect6_domains")
        conn.close()
        self.results: list[str | None] = [None, None]


def _make_pg_autocommit_thread_fn(idx: int, name: str):
    """SELECT-UPDATE-INSERT with autocommit=True (baseline, no row locks)."""

    def thread_fn(state: _PgState) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS(SELECT 1 FROM defect6_domains WHERE is_primary = TRUE)")
                cur.fetchone()
                cur.execute("UPDATE defect6_domains SET is_primary = FALSE WHERE is_primary = TRUE")
                cur.execute(
                    "INSERT INTO defect6_domains (name, is_primary) VALUES (%s, TRUE)",
                    (name,),
                )
            state.results[idx] = "ok"
        except Exception as exc:
            state.results[idx] = f"error: {type(exc).__name__}: {exc}"
        finally:
            conn.close()

    return thread_fn


def _make_pg_transactional_thread_fn(idx: int, name: str):
    """SELECT-UPDATE-INSERT within a transaction (autocommit=False).

    Mimics Django's @transaction.atomic: psycopg2 implicitly starts a
    transaction at the first statement, conn.commit() ends it.
    """

    def thread_fn(state: _PgState) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS(SELECT 1 FROM defect6_domains WHERE is_primary = TRUE)")
                cur.fetchone()
                cur.execute("UPDATE defect6_domains SET is_primary = FALSE WHERE is_primary = TRUE")
                cur.execute(
                    "INSERT INTO defect6_domains (name, is_primary) VALUES (%s, TRUE)",
                    (name,),
                )
            conn.commit()
            state.results[idx] = "ok"
        except Exception as exc:
            conn.rollback()
            state.results[idx] = f"error: {type(exc).__name__}: {exc}"
        finally:
            conn.close()

    return thread_fn


def _pg_invariant(state: _PgState) -> bool:
    """At most one domain should have is_primary=TRUE."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM defect6_domains WHERE is_primary = TRUE")
        primary_count = cur.fetchone()[0]
    conn.close()
    return primary_count <= 1


class TestPgAutocommitPhantomRace:
    """Baseline: autocommit mode should detect UPDATE-INSERT phantom race."""

    def test_autocommit_finds_race(self, pg_table) -> None:
        """With autocommit=True, row-lock arbitration is disabled, so DPOR
        can freely interleave both UPDATEs before either INSERT.
        """
        result = frontrun.explore(
            setup=_PgState,
            workers=[
                _make_pg_autocommit_thread_fn(0, "domain_a"),
                _make_pg_autocommit_thread_fn(1, "domain_b"),
            ],
            invariant=_pg_invariant,
            detect_io=True,
            timeout_per_run=10.0,
            deadlock_timeout=10.0,
            max_executions=50,
            preemption_bound=2,
            reproduce_on_failure=0,
        )

        assert not result.property_holds, (
            f"Expected DPOR to find the race with autocommit=True.\nnum_explored={result.num_explored}"
        )


def test_postgres_retains_prior_lock_after_zero_row_update(pg_table) -> None:
    """A later missing-row UPDATE does not release an earlier PostgreSQL row lock."""
    conn1 = psycopg2.connect(_DSN)
    conn2 = psycopg2.connect(_DSN)
    try:
        with conn1.cursor() as cur1:
            cur1.execute("INSERT INTO defect6_domains (name, is_primary) VALUES ('locked', FALSE) RETURNING id")
            row_id = cur1.fetchone()[0]
        conn1.commit()

        with conn1.cursor() as cur1:
            cur1.execute("UPDATE defect6_domains SET name = 'held' WHERE id = %s", (row_id,))
            assert cur1.rowcount == 1
            cur1.execute("UPDATE defect6_domains SET name = 'missing' WHERE id = -1")
            assert cur1.rowcount == 0

        with conn2.cursor() as cur2:
            cur2.execute("SET LOCAL lock_timeout = '100ms'")
            with pytest.raises(psycopg2.errors.LockNotAvailable):
                cur2.execute("UPDATE defect6_domains SET name = 'contender' WHERE id = %s", (row_id,))
    finally:
        conn1.rollback()
        conn2.rollback()
        conn1.close()
        conn2.close()


class TestPgTransactionalPhantomRace:
    """Defect #6: transactional mode must also detect UPDATE-INSERT phantom race.

    Before the fix, row-lock arbitration blocked Thread B's UPDATE until
    Thread A committed, preventing DPOR from exploring the interleaving
    where both 0-row UPDATEs execute before either INSERT.
    """

    def test_transactional_finds_race(self, pg_table) -> None:
        """With autocommit=False (transactions), DPOR must still detect the
        UPDATE-INSERT phantom race. 0-row UPDATEs should not acquire row
        locks (matching PostgreSQL semantics).
        """
        result = frontrun.explore(
            setup=_PgState,
            workers=[
                _make_pg_transactional_thread_fn(0, "domain_a"),
                _make_pg_transactional_thread_fn(1, "domain_b"),
            ],
            invariant=_pg_invariant,
            detect_io=True,
            timeout_per_run=10.0,
            deadlock_timeout=10.0,
            max_executions=50,
            preemption_bound=2,
            reproduce_on_failure=0,
        )

        assert not result.property_holds, (
            f"DPOR should detect the transactional UPDATE-INSERT phantom race "
            f"but explored {result.num_explored} interleavings without finding "
            f"an invariant violation. Defect #6 is not fixed."
        )
