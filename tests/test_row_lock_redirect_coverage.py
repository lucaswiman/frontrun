"""Fail-closed coverage demotion for row-lock-blocked redirects (issue #250).

When the DPOR engine schedules a worker whose row-lock acquire then blocks on
a modeled row lock, the engine step was already committed (``before_sync_retry``
runs before ``acquire_row_locks`` can redirect execution to the lock holder) —
both in the xproc relay's ACQUIRE_LOCKS path and in the in-process SQL path
(``_sql_cursor._dpor_schedule_and_suppress_sync``). The engine's schedule and
the physical execution can therefore diverge at row-lock boundaries, and
derived executions seeded from the desynced trace may be silently pruned.

Until the protocol fix (defer engine-step commitment until row-lock
arbitration decides) lands, any exploration that observed such a redirect must
not certify coverage: ``exhausted`` is demoted to ``False`` — sticky across
executions within one exploration — while ok/failure reporting is untouched.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import frontrun
from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher
from frontrun._io_detection import tx_store
from frontrun._sql_cursor import _dpor_schedule_and_suppress_sync
from frontrun._sql_row_locks import _release_dpor_row_locks

ROW = "sql:accounts:id=1"


class _DB:
    def __init__(self) -> None:
        self.balance = 0

    def reset(self) -> None:
        self.balance = 0


def _row_locked_worker(db: _DB):
    """xproc worker: RMW on a shared value under a modeled row lock."""

    def worker(proxy: Any) -> None:
        proxy.acquire_row_locks(0, [ROW])
        proxy.report_and_wait(None, 0)
        current = db.balance
        proxy.io_report(ROW, "read")
        proxy.report_and_wait(None, 0)
        db.balance = current + 100
        proxy.io_report(ROW, "write")
        proxy.release_row_locks(0)

    return worker


class _NoRedirectFailureCoordinator(DporCrossProcessCoordinator):
    """Suppress the per-execution redirect hard-fail to isolate the coverage claim.

    ``_evaluate`` already fails a redirected execution closed with
    ``failure_kind="nondeterministic"`` (which itself demotes ``exhausted``).
    This subclass simulates the exact state the issue #250 audit describes —
    the engine believes the search tree was fully explored and no failure is
    surfaced — so the test pins the *sticky* coverage-claim demotion
    independently of failure reporting.
    """

    def _evaluate(self, *args: Any, **kwargs: Any) -> CrossProcessResult | None:
        result = super()._evaluate(*args, **kwargs)
        if result is not None and result.failure_kind == "nondeterministic":
            return None
        return result


def test_xproc_redirect_demotes_exhausted_even_without_failure() -> None:
    """A row-lock redirect must demote exhausted even when no failure is reported.

    Both workers take the same modeled row lock, so DPOR's lock-order
    reversals deterministically drive one worker's ACQUIRE_LOCKS into
    contention: its engine step is committed, then execution is redirected to
    the holder. With the per-execution hard-fail filtered out, the engine
    finishes the (unbounded) search believing it covered everything — the
    coordinator must still refuse to certify coverage.
    """
    db = _DB()
    worker = _row_locked_worker(db)
    coord = _NoRedirectFailureCoordinator(
        num_workers=2, deadlock_timeout=5.0, preemption_bound=None, stop_on_first=False
    )
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: True,
    )
    # Fail-closed is about the coverage claim only: ok/failure reporting for
    # the executions that ran must not be demoted by the sticky flag.
    assert result.ok
    assert result.exhausted is False, (
        "engine-vs-physical divergence at a row-lock boundary (issue #250) must demote the coverage claim"
    )


def test_xproc_redirect_hard_fail_result_is_unchanged() -> None:
    """Regression guard: the released per-execution fail-closed result stays.

    The sticky exhausted demotion must not weaken the existing behavior of
    refusing to certify a redirected trace as a counterexample-quality result
    (``failure_kind="nondeterministic"``).
    """
    db = _DB()
    worker = _row_locked_worker(db)
    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=5.0, preemption_bound=None)
    result = coord.explore(
        worker_set=ThreadLauncher([worker, worker]),
        setup=db.reset,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "nondeterministic"
    assert result.exhausted is False


def _locked_increment(state: _DB) -> None:
    """Thread-mode worker taking the real in-process SQL row-lock seam.

    Pre-declares a pending row lock (what ``_sql_transactions`` does for
    ``SELECT ... FOR UPDATE`` / in-transaction DML) and runs it through
    ``_sql_cursor._dpor_schedule_and_suppress_sync``: the engine step is
    committed by ``before_sync_retry`` BEFORE ``_acquire_pending_row_locks``
    can discover contention and redirect execution to the holder — the exact
    issue #250 seam — all without needing a real database driver.
    """
    tx_store()._pending_row_locks = [ROW]
    _dpor_schedule_and_suppress_sync(
        False, "SELECT balance FROM accounts WHERE id = 1 FOR UPDATE", None, "qmark", lambda: None
    )
    current = state.balance
    state.balance = current + 100
    _release_dpor_row_locks(None)


def test_thread_mode_redirect_demotes_exhausted() -> None:
    """In-process DPOR hits the same seam and must demote its coverage claim.

    The row lock serializes the read-modify-write, so the invariant holds in
    every interleaving — but at least one explored execution schedules a
    worker into row-lock contention after committing its engine step, so the
    exploration must not leave its exhaustiveness claim intact.
    """
    result = frontrun.explore(
        setup=_DB,
        workers=[_locked_increment, _locked_increment],
        invariant=lambda db: db.balance == 200,
        preemption_bound=None,
        reproduce_on_failure=0,
        deadlock_timeout=5.0,
    )
    assert result.property_holds, result.explanation
    assert result.exhausted is False, (
        "a row-lock-blocked redirect occurred (issue #250); thread-mode must fail closed on coverage"
    )


# ---------------------------------------------------------------------------
# Postgres integration: real row-lock contention hits the redirect seam
# (SQLite never reaches it — it has no SELECT ... FOR UPDATE row locks).
# ---------------------------------------------------------------------------

try:
    import psycopg2
except ImportError:  # pragma: no cover - integration dependency
    psycopg2 = None  # type: ignore[assignment]

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")
_PG_TABLE = "xproc_redirect_accounts"


@pytest.fixture(scope="module")
def pg_accounts() -> str:
    """Create the shared accounts table, skipping when Postgres is unavailable."""
    if psycopg2 is None:
        pytest.skip("psycopg2 not installed")
    try:
        conn = psycopg2.connect(_DSN)
    except Exception:
        pytest.skip(f"Postgres not available at {_DSN}")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_PG_TABLE}")
        cur.execute(f"CREATE TABLE {_PG_TABLE} (id INT PRIMARY KEY, balance INT NOT NULL)")
    conn.close()
    return _DSN


def _pg_reset() -> None:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {_PG_TABLE}")
        cur.execute(f"INSERT INTO {_PG_TABLE} (id, balance) VALUES (1, 0)")
    conn.close()


def _pg_read_balance() -> int:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SELECT balance FROM {_PG_TABLE} WHERE id = 1")
        row = cur.fetchone()
    conn.close()
    assert row is not None
    return int(row[0])


class _PgState:
    def __init__(self) -> None:
        _pg_reset()


def _pg_locked_increment(_state: _PgState) -> None:
    conn = psycopg2.connect(_DSN)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(f"SELECT balance FROM {_PG_TABLE} WHERE id = 1 FOR UPDATE")
        row = cur.fetchone()
        assert row is not None
        cur.execute(f"UPDATE {_PG_TABLE} SET balance = %s WHERE id = 1", (row[0] + 100,))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
def test_postgres_thread_mode_redirect_demotes_exhausted(pg_accounts: str) -> None:
    """Real Postgres FOR UPDATE contention must demote thread-mode coverage.

    Two threads take the same row lock; frontrun's modeled row-lock
    arbitration redirects the engine-granted waiter to the holder after its
    engine step was committed (issue #250). The lock serializes the RMW so
    the invariant holds, but the exploration must not leave its
    exhaustiveness claim intact.
    """
    result = frontrun.explore(
        setup=_PgState,
        workers=[_pg_locked_increment, _pg_locked_increment],
        invariant=lambda _s: _pg_read_balance() == 200,
        detect_io=True,
        lock_timeout=2000,
        deadlock_timeout=15.0,
        timeout_per_run=30.0,
        reproduce_on_failure=0,
    )
    assert result.property_holds, result.explanation
    assert result.exhausted is False, (
        "Postgres row-lock contention redirected execution (issue #250); coverage must not be certified"
    )


@pytest.mark.integration
@pytest.mark.e2e
def test_postgres_xproc_redirect_never_certifies_coverage(pg_accounts: str) -> None:
    """Cross-process Postgres row-lock contention must never certify coverage.

    This is the audited issue #250 scenario: separate OS processes contend on
    the same Postgres row via SELECT ... FOR UPDATE; the coordinator's
    ACQUIRE_LOCKS arbitration redirects a worker whose engine step was
    already committed. Whatever verdict the run reports (today the redirected
    execution fails closed as ``failure_kind="nondeterministic"``), an
    unbounded search must not claim ``exhausted=True``.
    """
    target = "tests.xproc_demo_pg_lock:locked_increment"
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(target, (pg_accounts,)),
            "w1": frontrun.Subprocess(target, (pg_accounts,)),
        },
        setup=_pg_reset,
        invariant=lambda _s: _pg_read_balance() == 200,
        preemption_bound=None,
        deadlock_timeout=15.0,
    )
    assert result.exhausted is False, (
        "Postgres-grade row-lock contention hit the redirect seam (issue #250); "
        f"coverage must not be certified (ok={result.ok}, failure_kind={result.failure_kind})"
    )
    # The redirect must only affect the coverage claim / redirect fail-close;
    # it must never be reported as a bogus invariant violation.
    assert result.failure_kind != "invariant", result.failure
