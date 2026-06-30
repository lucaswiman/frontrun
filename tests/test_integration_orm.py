"""Integration tests: SQLAlchemy ORM races and deadlocks against real Postgres.

Covers two scenarios:

1. **Lost update** — two threads read-compute-write the same row without
   locking, causing one increment to be silently overwritten.

2. **Row-lock deadlock** — two threads acquire ``SELECT FOR UPDATE`` row
   locks in opposite orders, creating a lock-ordering cycle that Frontrun
   detects instantly and reports as an invariant violation
   (``result.property_holds = False``).

Requires a running Postgres with a ``frontrun_test`` database::

    createdb frontrun_test   # or: sudo -u postgres createdb frontrun_test

Running::

    make test-integration-3.14t
    # or:
    frontrun .venv-3.14t/bin/pytest tests/test_integration_orm.py -v
"""

from __future__ import annotations

import os

import pytest

try:
    from sqlalchemy import String, create_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
except ImportError:
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

import frontrun

pytestmark = pytest.mark.integration
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    login_count: Mapped[int] = mapped_column(default=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """Create the engine and verify Postgres is reachable."""
    eng = create_engine(_DB_URL)
    try:
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_URL}")
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    with Session(eng) as session:
        session.add(User(id=1, name="Alice", login_count=0))
        session.commit()
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture(scope="module")
def _pg_available(engine):  # type: ignore[no-untyped-def]
    """Marker fixture — just ensures the engine fixture ran."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_row(engine) -> None:  # type: ignore[no-untyped-def]
    with Session(engine) as session:
        user = session.get(User, 1)
        assert user is not None
        user.login_count = 0
        session.commit()


def _read_count(engine) -> int:  # type: ignore[no-untyped-def]
    with Session(engine) as session:
        user = session.get(User, 1)
        assert user is not None
        return user.login_count


# ---------------------------------------------------------------------------
# 1. Trace markers — deterministic reproduction
# ---------------------------------------------------------------------------


class TestOrmTraceMarkers:
    """Reproduce the lost update deterministically with TraceExecutor."""

    def test_lost_update_via_trace_markers(self, engine) -> None:  # type: ignore[no-untyped-def]
        _reset_row(engine)

        errors: dict[str, str] = {}

        def handler_a() -> None:
            try:
                with Session(engine) as session:
                    user = session.get(User, 1)  # frontrun: orm_read
                    assert user is not None
                    user.login_count = user.login_count + 1
                    session.commit()  # frontrun: orm_write
            except Exception as exc:
                errors["a"] = str(exc).strip()

        def handler_b() -> None:
            try:
                with Session(engine) as session:
                    user = session.get(User, 1)  # frontrun: orm_read
                    assert user is not None
                    user.login_count = user.login_count + 1
                    session.commit()  # frontrun: orm_write
            except Exception as exc:
                errors["b"] = str(exc).strip()

        schedule = Schedule(
            [
                Step("a", "orm_read"),
                Step("b", "orm_read"),
                Step("a", "orm_write"),
                Step("b", "orm_write"),
            ]
        )

        executor = TraceExecutor(schedule)
        executor.run({"a": handler_a, "b": handler_b}, timeout=10.0)

        assert not errors, errors
        assert _read_count(engine) == 1, "Expected lost update (login_count=1, not 2)"


# ---------------------------------------------------------------------------
# 2. Bytecode exploration — automatic detection
# ---------------------------------------------------------------------------


class TestOrmBytecodeExploration:
    """Find the lost update with random bytecode schedules."""

    def test_bytecode_finds_lost_update(self, engine) -> None:  # type: ignore[no-untyped-def]
        class _State:
            def __init__(self) -> None:
                _reset_row(engine)

        def _thread_fn(_state: _State) -> None:
            with Session(engine) as session:
                user = session.get(User, 1)
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()

        def _invariant(_state: _State) -> bool:
            return _read_count(engine) == 2

        result = frontrun.explore_random(
            setup=_State,
            threads=[_thread_fn, _thread_fn],
            invariant=_invariant,
            max_attempts=50,
            seed=42,
            detect_io=False,
            deadlock_timeout=15.0,
            timeout_per_run=60.0,
            reproduce_on_failure=5,
        )
        assert not result.property_holds, "Bytecode exploration should find the lost update"


# ---------------------------------------------------------------------------
# 3. DPOR — systematic exploration via I/O interdiction
# ---------------------------------------------------------------------------


class TestOrmDpor:
    """Systematically explore interleavings with DPOR + LD_PRELOAD."""

    def test_dpor_finds_lost_update(self, engine) -> None:  # type: ignore[no-untyped-def]
        class _State:
            def __init__(self) -> None:
                # Dispose the pool so new connections go through the patched
                # psycopg2.connect (which injects the TracedCursor factory).
                # explore() calls patch_sql() before setup(), but the
                # engine may have pooled connections from before patching.
                engine.dispose()
                _reset_row(engine)

        def _thread_fn(_state: _State) -> None:
            with Session(engine) as session:
                user = session.get(User, 1)
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()

        def _invariant(_state: _State) -> bool:
            return _read_count(engine) == 2

        result = frontrun.explore(
            setup=_State,
            workers=[_thread_fn, _thread_fn],
            invariant=_invariant,
            detect_io=True,
            deadlock_timeout=15.0,
        )
        assert not result.property_holds, "DPOR should find the lost update"


# ---------------------------------------------------------------------------
# 4. Naive threading — intermittent failure rate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Row-lock deadlock — detected as invariant violation
# ---------------------------------------------------------------------------


class TestOrmRowLockDeadlock:
    """DPOR detects a SELECT FOR UPDATE deadlock as an invariant violation.

    Thread A acquires a row lock on user 1, then tries to lock user 2.
    Thread B acquires a row lock on user 2, then tries to lock user 1.
    In the interleaving where each thread holds one lock and waits for the
    other, the WaitForGraph detects the cycle and sets property_holds=False.
    """

    @pytest.fixture(scope="class")
    def deadlock_engine(self, engine):  # type: ignore[no-untyped-def]
        """Ensure a second user row exists for the two-resource deadlock."""
        with Session(engine) as session:
            existing = session.get(User, 2)
            if existing is None:
                session.add(User(id=2, name="Bob", login_count=0))
                session.commit()
        return engine

    def test_row_lock_deadlock_is_invariant_violation(self, deadlock_engine) -> None:  # type: ignore[no-untyped-def]
        """explore() reports property_holds=False when a SELECT FOR UPDATE deadlock is found."""
        engine = deadlock_engine

        class _State:
            def __init__(self) -> None:
                engine.dispose()

        def thread_a(_state: _State) -> None:
            """Lock user 1, then user 2."""
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT id FROM users WHERE id = %s FOR UPDATE", (1,))
                conn.exec_driver_sql("SELECT id FROM users WHERE id = %s FOR UPDATE", (2,))
                conn.exec_driver_sql("COMMIT")

        def thread_b(_state: _State) -> None:
            """Lock user 2, then user 1 — opposite order from thread_a."""
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT id FROM users WHERE id = %s FOR UPDATE", (2,))
                conn.exec_driver_sql("SELECT id FROM users WHERE id = %s FOR UPDATE", (1,))
                conn.exec_driver_sql("COMMIT")

        def _invariant(_state: _State) -> bool:
            return True  # deadlock fires before invariant check

        result = frontrun.explore(
            setup=_State,
            workers=[thread_a, thread_b],
            invariant=_invariant,
            detect_io=True,
            deadlock_timeout=15.0,
        )

        assert not result.property_holds, "DPOR should detect the row-lock deadlock as an invariant violation"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower(), result.explanation

    def test_autocommit_no_false_deadlock(self, deadlock_engine) -> None:  # type: ignore[no-untyped-def]
        """SELECT FOR UPDATE with autocommit=True should NOT report a deadlock.

        In true autocommit mode each statement is its own transaction, so
        row locks are released immediately after the statement.  There is
        no window for a cross-thread deadlock cycle.
        """
        import psycopg2  # type: ignore[import-untyped]

        engine = deadlock_engine

        class _State:
            def __init__(self) -> None:
                engine.dispose()

        def thread_a(_state: _State) -> None:
            conn = psycopg2.connect(str(engine.url))
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (1,))
            cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (2,))
            cur.close()
            conn.close()

        def thread_b(_state: _State) -> None:
            conn = psycopg2.connect(str(engine.url))
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (2,))
            cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (1,))
            cur.close()
            conn.close()

        def _invariant(_state: _State) -> bool:
            return True

        result = frontrun.explore(
            setup=_State,
            workers=[thread_a, thread_b],
            invariant=_invariant,
            detect_io=True,
            deadlock_timeout=15.0,
        )

        assert result.property_holds, (
            f"Autocommit SELECT FOR UPDATE should NOT deadlock — got false deadlock: {result.explanation}"
        )
