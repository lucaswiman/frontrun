"""Regression test for FRONTRUN_DEFECTS.md #1: LD_PRELOAD shared socket deadlock.

When the LD_PRELOAD library is active, two DPOR threads that each open their
own psycopg2 connection to the same PostgreSQL server (via Unix socket) must
be treated as accessing a shared resource (the database) — but not incorrectly
flagged as sharing a single socket.

Previously, ``_PreloadBridge`` keyed its shared-fd tracking by resource_id
(the socket path) rather than fd, causing a spurious "shared socket" warning
and potential deadlocks when threads independently connected to the same
PostgreSQL server.
"""

from __future__ import annotations

import os

import pytest

try:
    import psycopg2
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

import frontrun
from frontrun.cli import require_active

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        conn = psycopg2.connect(_DB_URL)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _check_pg():
    if not _pg_available():
        pytest.skip(f"PostgreSQL not available at {_DB_URL}")


@pytest.fixture(autouse=True)
def _setup_table():
    """Create a simple test table, drop it after the test."""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS defect1_counter")
    cur.execute("CREATE TABLE defect1_counter (id INT PRIMARY KEY, val INT)")
    cur.execute("INSERT INTO defect1_counter VALUES (1, 0)")
    cur.close()
    conn.close()
    yield
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS defect1_counter")
    cur.close()
    conn.close()


class TestSharedSocketDeadlock:
    """Regression test for defect #1: LD_PRELOAD + per-thread psycopg2 connections."""

    def test_two_threads_own_connections_with_preload(self) -> None:
        """Two DPOR threads each opening their own psycopg2 connection.

        This simulates what django_dpor does: each thread opens a fresh
        connection to the same PostgreSQL server.  With LD_PRELOAD active,
        DPOR should detect the shared database resource and explore
        interleavings without deadlocking or emitting spurious warnings.
        """
        require_active("test_two_threads_own_connections_with_preload")

        class State:
            pass

        def thread_fn(state: State) -> None:
            conn = psycopg2.connect(_DB_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("UPDATE defect1_counter SET val = val + 1 WHERE id = 1")
            cur.close()
            conn.close()

        def invariant(state: State) -> bool:
            conn = psycopg2.connect(_DB_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT val FROM defect1_counter WHERE id = 1")
            val = cur.fetchone()[0]
            cur.close()
            conn.close()
            return val == 2

        result = frontrun.explore(
            setup=State,
            workers=[thread_fn, thread_fn],
            invariant=invariant,
            detect_io=True,
            timeout_per_run=5.0,
            total_timeout=15.0,
            max_executions=5,
            reproduce_on_failure=0,
        )

        # The key assertion: DPOR explored multiple interleavings without
        # deadlocking.  Before the fix, the spurious shared-socket warning
        # indicated incorrect conflict tracking.
        assert result.num_explored > 1, (
            f"DPOR explored only {result.num_explored} interleaving(s); "
            f"expected >1 with LD_PRELOAD detecting database I/O conflicts"
        )
