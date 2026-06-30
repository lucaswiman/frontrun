"""Integration test: two threads sharing a single psycopg2 cursor.

psycopg2 cursors are **not** thread-safe (DBAPI2 threadsafety level 2 means
connections may be shared between threads, but cursors must not be).  When two
threads call ``cursor.execute()`` and ``cursor.fetchone()`` concurrently on the
same cursor object, the cursor's internal result-set pointer can be clobbered
so that one thread reads the other thread's rows.

This file contains two test cases:

1. **test_dpor_detects_shared_cursor_race** — the racy version.  Two threads
   share *one* psycopg2 cursor.  The invariant "each thread reads back the
   value it wrote" should be violated when the interleaving is bad.  We expect
   DPOR to find a schedule where ``cursor.fetchone()`` in one thread returns
   the row fetched by the other thread (or raises an error), falsifying the
   invariant.

2. **test_dpor_safe_with_per_thread_cursors** — the safe version.  Each thread
   creates its own cursor on the *same* connection.  psycopg2 documents that a
   single connection may be used by multiple threads, provided each thread uses
   its own cursor.  The same invariant must hold in every interleaving.

Requires::

    - PostgreSQL running and a ``frontrun_test`` database accessible without a
      password (e.g. ``sudo -u postgres createdb frontrun_test`` plus a role
      matching the OS user, or ``DATABASE_URL`` env var).
    - psycopg2-binary installed (``make build-integration-3.14``).

Running::

    make test-integration-3.14 PYTEST_ARGS="-v -k psycopg2_cursor"
"""

from __future__ import annotations

import os

import pytest

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

from frontrun import explore

pytestmark = pytest.mark.integration

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")

# ---------------------------------------------------------------------------
# Module-level fixture: verify Postgres is reachable and create a test table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _pg_conn():
    """Open a long-lived psycopg2 connection for fixture setup and tear-down.

    The connection is in AUTOCOMMIT mode so DDL doesn't require an explicit
    COMMIT and won't be rolled back if a test raises.
    """
    try:
        conn = psycopg2.connect(_DB_URL)
    except Exception as exc:
        pytest.skip(f"Postgres not available at {_DB_URL!r}: {exc}")

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS cursor_race_test")
        cur.execute("""
            CREATE TABLE cursor_race_test (
                thread_id  INTEGER PRIMARY KEY,
                value      INTEGER NOT NULL
            )
        """)
        # Seed two rows so each thread can SELECT its own row.
        cur.execute("INSERT INTO cursor_race_test VALUES (0, 100)")
        cur.execute("INSERT INTO cursor_race_test VALUES (1, 200)")

    yield conn

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS cursor_race_test")
    conn.close()


# ---------------------------------------------------------------------------
# Racy test: two threads share ONE cursor
# ---------------------------------------------------------------------------


class TestSharedCursorRace:
    """DPOR should detect the race when two threads share one cursor."""

    def test_dpor_detects_shared_cursor_race(self, _pg_conn) -> None:
        """Invariant: each thread reads back the value for its own thread_id.

        With a *shared* cursor the call sequence

            Thread-0: execute("SELECT value … WHERE thread_id=0")
            Thread-1: execute("SELECT value … WHERE thread_id=1")   # overwrites T0's results
            Thread-0: fetchone()  → gets 200 (T1's row!) or None

        violates the invariant.  DPOR should find this interleaving.

        NOTE: psycopg2 cursor.execute() + fetchone() are C-extension calls.
        The cursor's Python-visible state (__dict__, statusmessage, rowcount,
        description, …) changes between execute() and fetchone(), so DPOR can
        detect the conflict through Python-level attribute reads.  However,
        because the actual result buffer lives in C, DPOR may only observe the
        race via the ``statusmessage`` / ``rowcount`` attributes rather than
        the raw wire data.  The invariant check is therefore deliberately
        permissive — it accepts *None* or wrong values as a violation to
        surface any symptom of the sharing.
        """
        # Use a fresh connection per explore() run so we don't bleed state.
        shared_conn = psycopg2.connect(_DB_URL)

        class State:
            """Shared state created fresh for each explored interleaving."""

            def __init__(self) -> None:
                # One cursor shared by both threads — the unsafe pattern.
                self.cursor = shared_conn.cursor()
                # Each thread stores what it read.
                self.results: list[int | None] = [None, None]

        def make_thread(idx: int):
            def thread_fn(state: State) -> None:
                state.cursor.execute(
                    "SELECT value FROM cursor_race_test WHERE thread_id = %s",
                    (idx,),
                )
                row = state.cursor.fetchone()
                state.results[idx] = row[0] if row is not None else None

            return thread_fn

        def invariant(state: State) -> bool:
            """Each thread must read back its own value (100 or 200)."""
            r0, r1 = state.results
            # If either thread got None or the wrong value, the invariant fails.
            return r0 == 100 and r1 == 200

        try:
            result = explore(
                setup=State,
                workers=[make_thread(0), make_thread(1)],
                invariant=invariant,
                max_executions=30,
                deadlock_timeout=10.0,
                timeout_per_run=10.0,
                reproduce_on_failure=50,
                lock_timeout=2000,
            )
        finally:
            shared_conn.close()

        assert not result.property_holds, (
            "DPOR should have found a schedule where the shared cursor "
            f"returns wrong results, but explored {result.num_explored} "
            "interleavings without finding a violation."
        )
        assert result.explanation is not None
        assert result.num_explored >= 2, "Must explore at least 2 interleavings"
        assert result.reproduction_attempts == 50
        assert result.reproduction_successes == 50, (
            f"Expected 50/50 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )


# ---------------------------------------------------------------------------
# Safe test: two threads each use their OWN cursor on the same connection
# ---------------------------------------------------------------------------


class TestPerThreadCursorSafe:
    """DPOR must NOT flag a violation when each thread has its own cursor."""

    def test_dpor_safe_with_per_thread_cursors(self, _pg_conn) -> None:
        """Per-thread cursors on a shared connection: invariant must always hold."""
        shared_conn = psycopg2.connect(_DB_URL)

        class State:
            def __init__(self) -> None:
                # Each thread will create its own cursor in thread_fn.
                self.results: list[int | None] = [None, None]

        def make_thread(idx: int):
            def thread_fn(state: State) -> None:
                # Each thread gets its OWN cursor — the safe pattern.
                cur = shared_conn.cursor()
                cur.execute(
                    "SELECT value FROM cursor_race_test WHERE thread_id = %s",
                    (idx,),
                )
                row = cur.fetchone()
                state.results[idx] = row[0] if row is not None else None
                cur.close()

            return thread_fn

        def invariant(state: State) -> bool:
            r0, r1 = state.results
            return r0 == 100 and r1 == 200

        try:
            result = explore(
                setup=State,
                workers=[make_thread(0), make_thread(1)],
                invariant=invariant,
                max_executions=30,
                deadlock_timeout=10.0,
                timeout_per_run=10.0,
                reproduce_on_failure=0,
                lock_timeout=2000,
            )
        finally:
            shared_conn.close()

        assert result.property_holds, (
            f"Per-thread cursors should be safe, but DPOR found a violation: {result.explanation}"
        )
        assert result.num_explored >= 1
