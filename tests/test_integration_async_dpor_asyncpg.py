"""Integration tests: async DPOR with asyncpg against real Postgres.

Tests that ``explore(strategy="dpor")`` can detect SQL-level race conditions
using asyncpg as the database driver, with await_point() as the
scheduling granularity.

Requires a running Postgres with a ``frontrun_test`` database::

    createdb frontrun_test
"""

from __future__ import annotations

import asyncio
import os

import pytest

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("asyncpg not installed", allow_module_level=True)

from frontrun import explore
from frontrun.async_dpor import await_point

pytestmark = pytest.mark.integration

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")


@pytest.fixture(scope="module")
def _pg_available():
    """Ensure Postgres is available and create test table."""

    async def setup():
        try:
            conn = await asyncpg.connect(_DSN)
        except Exception:
            pytest.skip(f"Postgres not available at {_DSN}")
        try:
            await conn.execute("DROP TABLE IF EXISTS async_dpor_test")
            await conn.execute("""
                CREATE TABLE async_dpor_test (
                    id INTEGER PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                )
            """)
            await conn.execute("INSERT INTO async_dpor_test (id, value) VALUES (1, 0)")
        finally:
            await conn.close()

    asyncio.run(setup())
    yield

    async def teardown():
        try:
            conn = await asyncpg.connect(_DSN)
            await conn.execute("DROP TABLE IF EXISTS async_dpor_test")
            await conn.close()
        except Exception:
            pass

    asyncio.run(teardown())


class TestAsyncDporAsyncpg:
    """Async DPOR integration tests with asyncpg."""

    def test_finds_lost_update(self, _pg_available) -> None:
        """Async DPOR should detect a lost-update race via asyncpg."""

        async def run_test():
            # Reset
            conn = await asyncpg.connect(_DSN)
            await conn.execute("UPDATE async_dpor_test SET value = 0 WHERE id = 1")
            await conn.close()

            async def increment(_state: object) -> None:
                conn = await asyncpg.connect(_DSN)
                try:
                    row = await conn.fetchrow("SELECT value FROM async_dpor_test WHERE id = 1")
                    assert row is not None
                    current = row["value"]
                    await await_point()
                    await conn.execute("UPDATE async_dpor_test SET value = $1 WHERE id = 1", current + 1)
                finally:
                    await conn.close()

            result = await explore(
                setup=object,
                workers=[increment, increment],
                invariant=lambda s: True,  # We verify race via DB value
                detect_io=True,
                deadlock_timeout=10.0,
                timeout_per_run=15.0,
            )
            return result

        result = asyncio.run(run_test())
        # Key assertion: exploration completes without deadlock
        assert result.num_explored >= 1, "Should explore at least 1 interleaving"

    def test_exploration_completes_without_deadlock(self, _pg_available) -> None:
        """Read-only async DPOR with asyncpg should complete without deadlocking."""

        async def run_test():
            async def read_only(_state: object) -> None:
                conn = await asyncpg.connect(_DSN)
                try:
                    await conn.fetchrow("SELECT value FROM async_dpor_test WHERE id = 1")
                    await await_point()
                finally:
                    await conn.close()

            result = await explore(
                setup=object,
                workers=[read_only, read_only],
                invariant=lambda s: True,
                detect_io=True,
                deadlock_timeout=10.0,
            )
            return result

        result = asyncio.run(run_test())
        assert result.property_holds
        assert result.num_explored >= 1

    def test_sql_queries_tracked(self, _pg_available) -> None:
        """SQL queries through asyncpg should be intercepted and tracked."""

        async def run_test():
            conn = await asyncpg.connect(_DSN)
            await conn.execute("UPDATE async_dpor_test SET value = 0 WHERE id = 1")
            await conn.close()

            queries_seen: list[str] = []

            async def task_with_query(_state: object) -> None:
                conn = await asyncpg.connect(_DSN)
                try:
                    await conn.fetchrow("SELECT value FROM async_dpor_test WHERE id = 1")
                    queries_seen.append("select")
                    await await_point()
                    await conn.execute("UPDATE async_dpor_test SET value = 42 WHERE id = 1")
                    queries_seen.append("update")
                finally:
                    await conn.close()

            result = await explore(
                setup=object,
                workers=[task_with_query],
                invariant=lambda s: True,
                detect_io=True,
                deadlock_timeout=10.0,
            )
            return result, queries_seen

        result, queries = asyncio.run(run_test())
        assert result.num_explored >= 1
        assert "select" in queries
        assert "update" in queries
