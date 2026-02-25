"""Integration test: SQLAlchemy ORM lost-update race against real Postgres.

SQLAlchemy's ORM computes new column values in Python.  When two
concurrent sessions read the same row, compute a new value from the
stale read, and write it back, the second commit silently overwrites
the first — a classic **lost update**.

Requires a running Postgres with a ``frontrun_test`` database::

    createdb frontrun_test   # or: sudo -u postgres createdb frontrun_test

Running::

    make test-integration-3.14t
    # or:
    frontrun .venv-3.14t/bin/pytest tests/test_integration_orm.py -v
"""

from __future__ import annotations

import os
import random
import threading
import time

import pytest

try:
    from sqlalchemy import String, create_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
except ImportError:
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from frontrun.bytecode import explore_interleavings
from frontrun.common import Schedule, Step
from frontrun.dpor import explore_dpor
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
        executor.run("a", handler_a)
        executor.run("b", handler_b)
        executor.wait(timeout=10.0)

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

        result = explore_interleavings(
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
                _reset_row(engine)

        def _thread_fn(_state: _State) -> None:
            with Session(engine) as session:
                user = session.get(User, 1)
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()

        def _invariant(_state: _State) -> bool:
            return _read_count(engine) == 2

        result = explore_dpor(
            setup=_State,
            threads=[_thread_fn, _thread_fn],
            invariant=_invariant,
            detect_io=True,
            deadlock_timeout=15.0,
        )
        assert not result.property_holds, "DPOR should find the lost update"


# ---------------------------------------------------------------------------
# 4. Naive threading — intermittent failure rate
# ---------------------------------------------------------------------------


class TestOrmNaiveThreading:
    """Show the intermittent nature of the race with plain threads."""

    def test_naive_threading_race_rate(self, engine) -> None:  # type: ignore[no-untyped-def]
        trials = 500
        failures = 0
        rng = random.Random(42)

        for _ in range(trials):
            _reset_row(engine)

            def handler() -> None:
                with Session(engine) as session:
                    user = session.get(User, 1)
                    assert user is not None
                    user.login_count = user.login_count + 1
                    session.commit()

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            time.sleep(rng.uniform(0, 0.015))
            t2.start()
            t1.join()
            t2.join()

            if _read_count(engine) != 2:
                failures += 1

        assert failures > 0, f"Race never triggered in {trials} trials"
