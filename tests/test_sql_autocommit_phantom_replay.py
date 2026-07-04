"""Regression: autocommit UPDATE-then-INSERT phantom races must detect AND replay >= 8/10.

Two threads each run the check-then-act pattern against real PostgreSQL
(autocommit, one connection per thread, table with NO unique constraint):

    UPDATE t SET v = v + 1 WHERE k = %s     -- 0 rows (no row yet)
    if rowcount == 0:
        INSERT INTO t (k, v) VALUES (%s, 1)

The phantom interleaving (both UPDATEs before either INSERT) produces two
rows for one key.  This is the exact shape of the django-watson
``SearchEngine._update_obj_index_iter`` bug (UPDATE-then-INSERT with no
UNIQUE backstop, autocommit, real DB socket).

History: a django-watson audit reported this pattern as a frontrun
reproduction-reliability defect ("DPOR finds the counterexample but
``reproduce_on_failure`` replays it at only ~3-4/10" — see the discussion
around FRONTRUN_DEFECTS.md defect #14 in the oss-bug-analysis repo).
Investigation showed frontrun handles this pattern deterministically: the
race is found within ~2 interleavings and replays 10/10, both with raw
psycopg2 and through Django's connection layer.  The audit's flaky rates
came from the watson *test's* setup — ``User.objects.create_user()`` fires
watson's ``post_save`` receiver, which pre-created the SearchEntry row, so
the phantom window did not exist during the measured runs (any red runs
were cross-run contamination, hence the nondeterministic "reproduction").

This test pins the correct behavior so a real regression in SQL phantom
detection or schedule replay (e.g. in the ``:seq`` conflict arcs from
defects #2/#6, or the replay machinery from defects #15/#16) is caught.
"""

from __future__ import annotations

import os

import pytest

import frontrun

psycopg2 = pytest.importorskip("psycopg2")

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = f"dbname={_DB_NAME}"
_TABLE = "autocommit_phantom_replay"


def _pg_available() -> bool:
    try:
        conn = psycopg2.connect(_DSN)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def pg_table():
    if not _pg_available():
        pytest.skip("PostgreSQL not available")
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        # Deliberately NO unique constraint on k — that absence is the
        # application-level bug whose detection/replay we are pinning.
        cur.execute(f"CREATE TABLE {_TABLE} (id SERIAL PRIMARY KEY, k TEXT NOT NULL, v INTEGER NOT NULL)")
    conn.close()
    yield
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
    conn.close()


class _State:
    def __init__(self) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {_TABLE}")
        conn.close()


def _thread_fn(state: _State) -> None:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {_TABLE} SET v = v + 1 WHERE k = %s", ("key",))
            if cur.rowcount == 0:
                cur.execute(f"INSERT INTO {_TABLE} (k, v) VALUES (%s, 1)", ("key",))
    finally:
        conn.close()


def _invariant(state: _State) -> bool:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {_TABLE} WHERE k = %s", ("key",))
        count = cur.fetchone()[0]
    conn.close()
    return count == 1


class TestAutocommitPhantomReplayReliability:
    def test_phantom_counterexample_replays_reliably(self, pg_table) -> None:
        """DPOR must both FIND the phantom race and REPLAY it >= 8/10."""
        result = frontrun.explore(
            setup=_State,
            workers=[_thread_fn, _thread_fn],
            invariant=_invariant,
            detect_io=True,
            timeout_per_run=10.0,
            deadlock_timeout=10.0,
            max_executions=50,
            preemption_bound=2,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, (
            f"DPOR should detect the autocommit UPDATE-then-INSERT phantom race "
            f"but explored {result.num_explored} interleavings without a violation."
        )
        assert result.reproduction_attempts == 10
        assert result.reproduction_successes >= 8, (
            f"Counterexample replayed only {result.reproduction_successes}/10 times; "
            f"SQL phantom counterexamples must replay deterministically.\n"
            f"{result.explanation}"
        )
