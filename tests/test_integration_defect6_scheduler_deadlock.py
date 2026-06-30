"""Minimal reproduction of frontrun defect #6: DPOR cooperative scheduler
deadlocks with PostgreSQL row locks.

This is a general explore() (DPOR) bug, NOT Django-specific. Any explore()
test where two threads contend on the same PostgreSQL row lock will deadlock.

The scenario:
1. Thread A INSERTs a row (acquires PG row lock via UNIQUE constraint)
2. DPOR suspends thread A at the next scheduling point
3. DPOR resumes thread B
4. Thread B INSERTs the same row — PG blocks thread B waiting for A's lock
5. Thread B never reaches a DPOR scheduling point (blocked in kernel)
6. DPOR waits for thread B to yield — deadlock

Without lock_timeout, this test hangs forever (or until deadlock_timeout).
With lock_timeout, PG raises an error and the thread returns to DPOR.

Running:

    # This will HANG (demonstrating the bug):
    PYTHONPATH=libraries .venv/bin/frontrun pytest \
      frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_deadlock_without_lock_timeout \
      -v --timeout=30

    # This WORKS (demonstrating the workaround):
    PYTHONPATH=libraries .venv/bin/frontrun pytest \
      frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_workaround_with_lock_timeout \
      -v --timeout=30
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
_DSN = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")


@pytest.fixture(scope="module")
def _pg_available():
    """Create test table in Postgres."""
    try:
        conn = psycopg2.connect(_DSN)
    except Exception:
        pytest.skip(f"Postgres not available at {_DSN}")

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect6_test")
        cur.execute("""
            CREATE TABLE defect6_test (
                id TEXT PRIMARY KEY,
                value TEXT
            )
        """)
    conn.close()
    yield


class _State:
    """Shared state: two threads race to INSERT the same row."""

    def __init__(self) -> None:
        conn = psycopg2.connect(_DSN)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM defect6_test")
        conn.close()
        self.results: list[str | None] = [None, None]


def _make_thread_fn(idx: int, use_lock_timeout: bool = False):
    """Each thread opens its own connection and tries to INSERT the same row.

    This is the minimal TOCTOU: SELECT to check existence, then INSERT.
    """

    def _thread_fn(state: _State) -> None:
        conn = psycopg2.connect(_DSN)
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                if use_lock_timeout:
                    cur.execute("SET lock_timeout = '2s'")

                # CHECK: does the row exist?
                cur.execute("SELECT id FROM defect6_test WHERE id = %s", ("row1",))
                row = cur.fetchone()
                if row is not None:
                    state.results[idx] = "already_exists"
                    conn.rollback()
                    return

                # ACT: insert the row
                try:
                    cur.execute(
                        "INSERT INTO defect6_test (id, value) VALUES (%s, %s)",
                        ("row1", f"thread_{idx}"),
                    )
                    conn.commit()
                    state.results[idx] = "inserted"
                except Exception as exc:
                    conn.rollback()
                    state.results[idx] = f"error: {type(exc).__name__}: {exc}"
        finally:
            conn.close()

    return _thread_fn


def _invariant(state: _State) -> bool:
    """Invariant: no errors should occur if the TOCTOU is properly serialized.

    When the race is triggered, one thread gets a PG error (IntegrityError
    or LockNotAvailable with lock_timeout).
    """
    r0, r1 = state.results
    has_error = (r0 is not None and r0.startswith("error")) or (r1 is not None and r1.startswith("error"))
    return not has_error


def test_deadlock_without_lock_timeout(_pg_available) -> None:
    """Verifies defect #6 fix: DPOR no longer deadlocks with PG row locks.

    Previously, this test would HANG because DPOR's cooperative scheduler
    suspended thread A (holding a PG row lock), then resumed thread B which
    blocked in the kernel waiting for A's lock. Thread B never reached a DPOR
    scheduling point.

    The fix has two parts:
    1. DPOR row lock arbitration: write-kind SQL accesses inside transactions
       now acquire DPOR row locks, preventing the C-level PG deadlock.
    2. Replay safety: reproduce_on_failure replays always get a safety
       lock_timeout to prevent deadlocks in OpcodeScheduler (which lacks
       row lock arbitration).
    """
    result = explore(
        setup=_State,
        workers=[_make_thread_fn(0, use_lock_timeout=False), _make_thread_fn(1, use_lock_timeout=False)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
    )

    # The test completes without hanging — defect #6 is fixed.
    assert result.num_explored >= 1, "Expected at least 1 interleaving explored"


def test_workaround_with_lock_timeout(_pg_available) -> None:
    """Demonstrates the lock_timeout workaround: PG fails fast instead of
    blocking, allowing thread B to return to a DPOR scheduling point.

    This test should PASS (detecting the race) without hanging.
    """
    result = explore(
        setup=_State,
        workers=[_make_thread_fn(0, use_lock_timeout=True), _make_thread_fn(1, use_lock_timeout=True)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
    )

    assert not result.property_holds, (
        f"Expected race to be detected (invariant violated by PG error).\n"
        f"num_explored={result.num_explored}\n{result.explanation}"
    )


def test_explore_lock_timeout_parameter(_pg_available) -> None:
    """Test that explore()'s lock_timeout parameter automatically injects
    SET lock_timeout on PostgreSQL connections, preventing the DPOR/PG deadlock.

    Threads do NOT set lock_timeout themselves — explore() handles it.
    Without the lock_timeout parameter, this scenario hangs (see
    test_deadlock_without_lock_timeout). With lock_timeout, PG raises an error
    on blocked threads, allowing them to return to a DPOR scheduling point.

    The key assertion is that explore() completes without hanging. Whether
    DPOR detects the race depends on conflict analysis; the lock_timeout
    parameter's job is to prevent the cooperative scheduler deadlock.
    """
    result = explore(
        setup=_State,
        workers=[_make_thread_fn(0, use_lock_timeout=False), _make_thread_fn(1, use_lock_timeout=False)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
        lock_timeout=2000,
    )

    # The test completes without hanging — defect #6 is fixed.
    # DPOR may or may not detect the race depending on conflict analysis,
    # but the deadlock is prevented.
    assert result.num_explored >= 1, "Expected at least 1 interleaving explored"


def test_explore_dpor_lock_timeout_injects_on_connections(_pg_available) -> None:
    """Verify that lock_timeout is actually injected on new PG connections."""

    observed_lock_timeouts: list[str] = []

    def _check_thread_fn(state: _State) -> None:
        conn = psycopg2.connect(_DSN)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SHOW lock_timeout")
                lt = cur.fetchone()
                if lt:
                    observed_lock_timeouts.append(lt[0])
            state.results[0] = "ok"
        finally:
            conn.close()

    result = explore(
        setup=_State,
        workers=[_check_thread_fn],
        invariant=lambda s: True,
        deadlock_timeout=5.0,
        timeout_per_run=10.0,
        lock_timeout=2000,
    )

    assert result.num_explored >= 1
    assert any(lt == "2s" for lt in observed_lock_timeouts), (
        f"Expected lock_timeout='2s' on new connections, got {observed_lock_timeouts}"
    )


def test_counterexample_replay_uses_dpor_row_lock_arbitration(_pg_available) -> None:
    """Counterexample replay should reproduce under the DPOR runner.

    This regression specifically covers the old `0/10` behavior: DPOR found
    the failing schedule, but replay delegated to OpcodeScheduler, which lacks
    DPOR's PostgreSQL row-lock arbitration and therefore could not re-run the
    path reliably.
    """
    result = explore(
        setup=_State,
        workers=[_make_thread_fn(0, use_lock_timeout=False), _make_thread_fn(1, use_lock_timeout=False)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
        reproduce_on_failure=3,
        max_executions=50,
        preemption_bound=2,
    )

    assert not result.property_holds, "Expected DPOR to detect the insert race"
    assert result.reproduction_attempts == 3
    assert result.reproduction_successes > 0, (
        f"Expected DPOR-native replay to reproduce at least once, got "
        f"{result.reproduction_successes}/{result.reproduction_attempts}"
    )
