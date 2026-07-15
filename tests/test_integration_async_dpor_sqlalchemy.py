"""Integration tests: async DPOR with SQLAlchemy async against real Postgres.

Tests that ``explore(strategy="dpor")`` can detect SQL-level race conditions
using SQLAlchemy's async engine, with await_point() as the scheduling
granularity.

Requires a running Postgres with a ``frontrun_test`` database::

    createdb frontrun_test
"""

from __future__ import annotations

import asyncio
import os

import pytest

try:
    from sqlalchemy import String, text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
except ImportError:
    pytest.skip("sqlalchemy[asyncio] not installed", allow_module_level=True)

pytestmark = pytest.mark.integration

try:
    import asyncpg  # noqa: F401
except ImportError:
    pytest.skip("asyncpg not installed (needed for async postgres driver)", allow_module_level=True)

import frontrun
from frontrun.async_dpor import await_point
from frontrun.contrib.sqlalchemy import async_sqlalchemy_dpor, get_async_connection

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_SYNC_DSN = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")
# asyncpg needs the +asyncpg scheme; derive from the sync DSN
_ASYNC_DB_URL = (
    _SYNC_DSN.replace("postgresql://", "postgresql+asyncpg://", 1) if "asyncpg" not in _SYNC_DSN else _SYNC_DSN
)


# ---------------------------------------------------------------------------
# SQLAlchemy async model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class AsyncCounter(Base):
    __tablename__ = "async_sa_counter"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[int] = mapped_column(default=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _pg_available():
    """Create test table using sync psycopg2 (avoids event loop issues)."""
    try:
        import psycopg2

        conn = psycopg2.connect(_SYNC_DSN)
    except Exception:
        pytest.skip(f"Postgres not available at {_SYNC_DSN}")

    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS async_sa_counter")
        cur.execute("""
            CREATE TABLE async_sa_counter (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                value INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("INSERT INTO async_sa_counter (id, name, value) VALUES (1, 'test', 0)")
    conn.close()
    yield

    try:
        conn = psycopg2.connect(_SYNC_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS async_sa_counter")
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncDporSQLAlchemy:
    """Async DPOR integration tests with SQLAlchemy async."""

    def test_finds_lost_update_core(self, _pg_available) -> None:
        """Async DPOR should detect a lost-update race via SQLAlchemy Core."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:
                # Reset
                async with engine.begin() as conn:
                    await conn.execute(text("UPDATE async_sa_counter SET value = 0 WHERE id = 1"))

                async def increment(_state: object) -> None:
                    async with engine.connect() as conn:
                        result = await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))
                        row = result.fetchone()
                        assert row is not None
                        current = row[0]
                        await await_point()
                        await conn.execute(
                            text("UPDATE async_sa_counter SET value = :val WHERE id = 1"),
                            {"val": current + 1},
                        )
                        await conn.commit()

                return await frontrun.explore(
                    setup=object,
                    workers=[increment, increment],
                    invariant=lambda s: True,
                    detect_io=True,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert result.num_explored >= 1

    def test_finds_lost_update_orm(self, _pg_available) -> None:
        """Async DPOR should detect a lost-update race via SQLAlchemy async ORM."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:
                # Reset
                async with AsyncSession(engine) as session:
                    counter = await session.get(AsyncCounter, 1)
                    if counter is not None:
                        counter.value = 0
                        await session.commit()

                async def increment(_state: object) -> None:
                    async with AsyncSession(engine) as session:
                        counter = await session.get(AsyncCounter, 1)
                        assert counter is not None
                        current = counter.value
                        await await_point()
                        counter.value = current + 1
                        await session.commit()

                return await frontrun.explore(
                    setup=object,
                    workers=[increment, increment],
                    invariant=lambda s: True,
                    detect_io=True,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert result.num_explored >= 1

    def test_exploration_completes(self, _pg_available) -> None:
        """Verify the exploration completes without deadlock."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:

                async def read_only(_state: object) -> None:
                    async with engine.connect() as conn:
                        await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))
                        await await_point()

                return await frontrun.explore(
                    setup=object,
                    workers=[read_only, read_only],
                    invariant=lambda s: True,
                    detect_io=True,
                    deadlock_timeout=10.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert result.property_holds
        assert result.num_explored >= 1


def _pg_value() -> int:
    """Read the counter value with a plain sync connection (invariant helper)."""
    import psycopg2

    conn = psycopg2.connect(_SYNC_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM async_sa_counter WHERE id = 1")
            row = cur.fetchone()
            assert row is not None
            return row[0]
    finally:
        conn.close()


def _pg_reset() -> object:
    """Reset the counter with a plain sync connection (setup helper)."""
    import psycopg2

    conn = psycopg2.connect(_SYNC_DSN)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("UPDATE async_sa_counter SET value = 0 WHERE id = 1")
    finally:
        conn.close()
    return object()


class TestAsyncpgPreparedStatementReporting:
    """Regressions for the asyncpg prepared-statement blind spot.

    SQLAlchemy's asyncpg dialect executes every statement through
    ``Connection.prepare()`` → ``PreparedStatement.fetch()``.  Only asyncpg's
    textual ``BEGIN;``/``COMMIT;``/``ROLLBACK;`` (sent via the patched
    ``Connection.execute``) used to be reported, so DPOR saw no SQL
    dependencies: with a pre-warmed pool a genuine lost update was falsely
    certified, and race-free explorations were noisy and inconclusive.
    """

    def test_statements_reported_through_prepared_statements(self, _pg_available, monkeypatch) -> None:
        """SELECT/UPDATE through an AsyncEngine must reach _report_sql_access."""
        import frontrun._sql_cursor_async as sca

        seen: list[str] = []
        orig = sca._report_sql_access

        def spy(operation, parameters=None, **kwargs):  # type: ignore[no-untyped-def]
            if isinstance(operation, str):
                seen.append(operation)
            return orig(operation, parameters, **kwargs)

        monkeypatch.setattr(sca, "_report_sql_access", spy)

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:

                async def task(_state: object) -> None:
                    async with engine.connect() as conn:
                        await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))
                        await conn.execute(text("UPDATE async_sa_counter SET value = 0 WHERE id = 1"))
                        await conn.commit()

                return await frontrun.explore(
                    setup=object,
                    workers=[task],
                    invariant=lambda s: True,
                    detect_io=True,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert result.num_explored >= 1
        assert any("SELECT value FROM async_sa_counter" in op for op in seen), (
            f"SELECT never reached _report_sql_access; statements seen: {seen}"
        )
        assert any("UPDATE async_sa_counter" in op for op in seen), (
            f"UPDATE never reached _report_sql_access; statements seen: {seen}"
        )

    def test_prewarmed_pool_lost_update_detected(self, _pg_available) -> None:
        """A pool warmed before explore() must not falsely certify a lost update."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL, pool_size=2)
            try:
                # Pre-warm the pool so connections exist before exploration
                # patches drivers — like any long-lived application engine.
                c1 = await engine.connect()
                c2 = await engine.connect()
                await c1.close()
                await c2.close()

                async def increment(_state: object) -> None:
                    async with engine.connect() as conn:
                        row = (
                            await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))
                        ).fetchone()
                        assert row is not None
                        await await_point()
                        await conn.execute(
                            text("UPDATE async_sa_counter SET value = :v WHERE id = 1"),
                            {"v": row[0] + 1},
                        )
                        await conn.commit()

                return await frontrun.explore(
                    setup=_pg_reset,
                    workers=[increment, increment],
                    invariant=lambda s: _pg_value() == 2,
                    detect_io=True,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert not result.property_holds, (
            f"genuine lost update falsely certified clean after {result.num_explored} interleaving(s)"
        )

    def test_contrib_helper_finds_lost_update(self, _pg_available) -> None:
        """async_sqlalchemy_dpor must find the lost update with asyncpg."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:

                async def increment(_state: object) -> None:
                    conn = get_async_connection()
                    row = (await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))).fetchone()
                    assert row is not None
                    await conn.execute(
                        text("UPDATE async_sa_counter SET value = :v WHERE id = 1"),
                        {"v": row[0] + 1},
                    )
                    await conn.commit()

                return await async_sqlalchemy_dpor(
                    engine=engine,
                    setup=_pg_reset,
                    tasks=[increment, increment],
                    invariant=lambda s: _pg_value() == 2,
                    lock_timeout=2000,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert not result.property_holds, (
            f"contrib helper missed the lost update after {result.num_explored} interleaving(s)"
        )

    def test_contrib_helper_certifies_race_free(self, _pg_available) -> None:
        """Race-free read-only tasks must certify property_holds=True (no timeouts)."""

        async def run_test():
            engine = create_async_engine(_ASYNC_DB_URL)
            try:

                async def read_only(_state: object) -> None:
                    conn = get_async_connection()
                    (await conn.execute(text("SELECT value FROM async_sa_counter WHERE id = 1"))).fetchone()

                return await async_sqlalchemy_dpor(
                    engine=engine,
                    setup=object,
                    tasks=[read_only, read_only],
                    invariant=lambda s: True,
                    lock_timeout=2000,
                    deadlock_timeout=10.0,
                    timeout_per_run=15.0,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run_test())
        assert result.property_holds, (
            f"race-free exploration inconclusive after {result.num_explored} interleaving(s): {result.explanation}"
        )
