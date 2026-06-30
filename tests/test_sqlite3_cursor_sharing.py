"""Test: two threads sharing a single sqlite3 cursor.

sqlite3 cursors are **not** thread-safe.  When two threads call
``cursor.execute()`` and ``cursor.fetchone()`` concurrently on the same
cursor object, the cursor's internal result-set iterator can be clobbered
so that one thread reads the other thread's rows.

This file contains two test cases:

1. **test_dpor_detects_shared_cursor_race** — the racy version.  Two threads
   share *one* sqlite3 cursor.  The invariant "each thread reads back the
   value it wrote" should be violated when the interleaving is bad.

2. **test_dpor_safe_with_per_thread_cursors** — the safe version.  Each thread
   creates its own cursor on the *same* connection.  The same invariant must
   hold in every interleaving.

No external services required — uses an in-memory sqlite3 database.

Running::

    make test-3.14 PYTEST_ARGS="-v -k sqlite3_cursor"
"""

from __future__ import annotations

import sqlite3

import frontrun

# ---------------------------------------------------------------------------
# Racy test: two threads share ONE cursor
# ---------------------------------------------------------------------------


class TestSqlite3SharedCursorRace:
    """DPOR should detect the race when two threads share one sqlite3 cursor."""

    def test_dpor_detects_shared_cursor_race(self) -> None:
        """Invariant: each thread reads back the value for its own thread_id.

        With a *shared* cursor the call sequence

            Thread-0: execute("SELECT value … WHERE thread_id=0")
            Thread-1: execute("SELECT value … WHERE thread_id=1")   # overwrites T0's results
            Thread-0: fetchone()  → gets 200 (T1's row!) or None

        violates the invariant.  DPOR should find this interleaving.
        """
        shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
        shared_conn.execute("""
            CREATE TABLE cursor_race_test (
                thread_id  INTEGER PRIMARY KEY,
                value      INTEGER NOT NULL
            )
        """)
        shared_conn.execute("INSERT INTO cursor_race_test VALUES (0, 100)")
        shared_conn.execute("INSERT INTO cursor_race_test VALUES (1, 200)")
        shared_conn.commit()

        class State:
            def __init__(self) -> None:
                self.cursor = shared_conn.cursor()
                self.results: list[int | None] = [None, None]

        def make_thread(idx: int):
            def thread_fn(state: State) -> None:
                state.cursor.execute(
                    "SELECT value FROM cursor_race_test WHERE thread_id = ?",
                    (idx,),
                )
                row = state.cursor.fetchone()
                state.results[idx] = row[0] if row is not None else None

            return thread_fn

        def invariant(state: State) -> bool:
            r0, r1 = state.results
            return r0 == 100 and r1 == 200

        try:
            result = frontrun.explore(
                setup=State,
                workers=[make_thread(0), make_thread(1)],
                invariant=invariant,
                max_executions=30,
                deadlock_timeout=10.0,
                timeout_per_run=10.0,
                reproduce_on_failure=50,
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


class TestSqlite3PerThreadCursorSafe:
    """DPOR must NOT flag a violation when each thread has its own cursor."""

    def test_dpor_safe_with_per_thread_cursors(self) -> None:
        """Per-thread cursors on a shared connection: invariant must always hold."""
        shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
        shared_conn.execute("""
            CREATE TABLE cursor_race_test (
                thread_id  INTEGER PRIMARY KEY,
                value      INTEGER NOT NULL
            )
        """)
        shared_conn.execute("INSERT INTO cursor_race_test VALUES (0, 100)")
        shared_conn.execute("INSERT INTO cursor_race_test VALUES (1, 200)")
        shared_conn.commit()

        class State:
            def __init__(self) -> None:
                self.results: list[int | None] = [None, None]

        def make_thread(idx: int):
            def thread_fn(state: State) -> None:
                cur = shared_conn.cursor()
                cur.execute(
                    "SELECT value FROM cursor_race_test WHERE thread_id = ?",
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
            result = frontrun.explore(
                setup=State,
                workers=[make_thread(0), make_thread(1)],
                invariant=invariant,
                max_executions=30,
                deadlock_timeout=10.0,
                timeout_per_run=10.0,
                reproduce_on_failure=0,
            )
        finally:
            shared_conn.close()

        assert result.property_holds, (
            f"Per-thread cursors should be safe, but DPOR found a violation: {result.explanation}"
        )
        assert result.num_explored >= 1
