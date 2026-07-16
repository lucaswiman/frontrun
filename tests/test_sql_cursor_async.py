"""Tests for async DBAPI cursor monkey-patching.

Uses aiosqlite (async wrapper around sqlite3) to test the async patching
mechanism.  Tests mirror the structure of test_sql_cursor.py.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

import frontrun._sql_cursor_async as sql_cursor_async_mod
from frontrun._io_detection import _io_tls, set_io_reporter
from frontrun._sql_cursor_async import (
    _ASYNC_ORIGINAL_METHODS,
    _ASYNC_PATCHES,
    _intercept_execute_async,
    _report_sql_access,
    patch_sql_async,
    unpatch_sql_async,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class IOLog:
    """Collects IO events reported to the reporter callback."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, resource_id: str, kind: str) -> None:
        with self._lock:
            self.events.append((resource_id, kind))

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    @property
    def resource_ids(self) -> list[str]:
        with self._lock:
            return [r for r, _ in self.events]

    @property
    def kinds(self) -> list[str]:
        with self._lock:
            return [k for _, k in self.events]

    def events_for_table(self, table: str) -> list[tuple[str, str]]:
        prefix = f"sql:{table}"
        with self._lock:
            return [(r, k) for r, k in self.events if r == prefix or r.startswith(f"{prefix}:")]


async def _make_async_db() -> aiosqlite.Connection:
    """Create an in-memory aiosqlite database with test tables."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    await conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    await conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    await conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    await conn.execute("INSERT INTO orders VALUES (1, 1, 99.99)")
    await conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_async_sql_patch() -> Generator[None, None, None]:
    """Ensure async SQL patching is cleaned up between tests."""
    yield
    # aiosqlite 0.22 resolves close() after queuing its worker stop.  Join
    # already-closing non-daemon workers before the global leak check runs.
    for thread in threading.enumerate():
        if "_connection_worker_thread" in thread.name:
            thread.join(timeout=1.0)
    unpatch_sql_async()
    _ASYNC_ORIGINAL_METHODS.clear()
    _ASYNC_PATCHES.clear()
    sql_cursor_async_mod._sql_async_patched = False
    set_io_reporter(None)
    if hasattr(_io_tls, "_sql_suppress"):
        _io_tls._sql_suppress = False
    if hasattr(_io_tls, "_in_transaction"):
        _io_tls._in_transaction = False
    if hasattr(_io_tls, "_tx_buffer"):
        _io_tls._tx_buffer = []


# ---------------------------------------------------------------------------
# 1. Basic patching/unpatching
# ---------------------------------------------------------------------------


def test_patch_patches_aiosqlite_cursor() -> None:
    orig_execute = aiosqlite.Cursor.execute
    patch_sql_async()
    assert aiosqlite.Cursor.execute is not orig_execute


def test_patch_patches_aiosqlite_connection() -> None:
    orig_execute = aiosqlite.Connection.execute
    patch_sql_async()
    assert aiosqlite.Connection.execute is not orig_execute


@pytest.mark.asyncio
async def test_patch_wraps_aiosqlite_connection_transaction_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection-level commit/rollback/close must pass through async TX modeling."""
    seen: list[str] = []
    original_report = sql_cursor_async_mod._report_sql_access

    def spy(operation: Any, parameters: Any = None, **kwargs: Any) -> bool:
        if isinstance(operation, str):
            seen.append(operation)
        return original_report(operation, parameters, **kwargs)

    monkeypatch.setattr(sql_cursor_async_mod, "_report_sql_access", spy)
    patch_sql_async()
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE t (id INTEGER)")
        await conn.commit()
        await conn.rollback()
    finally:
        await conn.close()

    assert "COMMIT" in seen
    assert "ROLLBACK" in seen


def test_unpatch_restores_originals() -> None:
    orig_cursor_execute = aiosqlite.Cursor.execute
    orig_conn_execute = aiosqlite.Connection.execute
    patch_sql_async()
    unpatch_sql_async()
    assert aiosqlite.Cursor.execute is orig_cursor_execute
    assert aiosqlite.Connection.execute is orig_conn_execute


def test_double_patch_is_idempotent() -> None:
    patch_sql_async()
    execute_after_first = aiosqlite.Cursor.execute
    patch_sql_async()
    assert aiosqlite.Cursor.execute is execute_after_first


def test_double_unpatch_is_idempotent() -> None:
    patch_sql_async()
    unpatch_sql_async()
    unpatch_sql_async()  # should not raise


# ---------------------------------------------------------------------------
# 2. SQL interception via aiosqlite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_reports_read() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        log.clear()
        await conn.execute("SELECT * FROM users")

    assert any(r.startswith("sql:users") and k == "read" for r, k in log.events)


@pytest.mark.asyncio
async def test_insert_reports_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        log.clear()
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")

    assert any(r.startswith("sql:users") and k == "write" for r, k in log.events)


@pytest.mark.asyncio
async def test_update_reports_read_and_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        log.clear()
        await conn.execute("UPDATE users SET name = 'Bob' WHERE id = 1")

    read_events = [r for r, k in log.events if k == "read"]
    write_events = [r for r, k in log.events if k == "write"]
    assert any(r.startswith("sql:users") for r in read_events)
    assert any(r.startswith("sql:users") for r in write_events)


@pytest.mark.asyncio
async def test_delete_reports_read_and_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        log.clear()
        await conn.execute("DELETE FROM users WHERE id = 1")

    read_events = [r for r, k in log.events if k == "read"]
    write_events = [r for r, k in log.events if k == "write"]
    assert any(r.startswith("sql:users") for r in read_events)
    assert any(r.startswith("sql:users") for r in write_events)


@pytest.mark.asyncio
async def test_no_reporter_no_events() -> None:
    """Without a reporter set, interception still works but nothing is logged."""
    set_io_reporter(None)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE t (id INTEGER)")
        await conn.execute("SELECT * FROM t")
        # No reporter — should not raise


@pytest.mark.asyncio
async def test_multi_table_join() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER)")
        log.clear()
        await conn.execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id")

    assert any(r.startswith("sql:users") and k == "read" for r, k in log.events)
    assert any(r.startswith("sql:orders") and k == "read" for r, k in log.events)


@pytest.mark.asyncio
async def test_parameterized_query() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        log.clear()
        await conn.execute("SELECT * FROM users WHERE id = ?", (1,))

    # Should report row-level resource ID with predicate
    assert len(log.events) > 0
    assert any(r.startswith("sql:users") for r, _ in log.events)


@pytest.mark.asyncio
async def test_parameters_passed_as_keyword() -> None:
    """F7: ``conn.execute(sql, parameters=[...])`` must work under patching.

    The unpatched aiosqlite signature is ``execute(self, sql, parameters=None)``
    — ``parameters`` is an ordinary keyword.  The patched wrapper previously
    made it positional-only (``/``), so passing it by keyword raised a
    TypeError under exploration even though it is legal unpatched.
    """
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        log.clear()
        # Keyword 'parameters' is legal on the unpatched method.
        cursor = await conn.execute("SELECT * FROM users WHERE id = ?", parameters=(1,))
        rows = await cursor.fetchall()

    assert rows == [(1, "Alice")]
    # Row-level resolution should still see the parameter value.
    assert any("1" in r for r in log.resource_ids), log.resource_ids


@pytest.mark.asyncio
async def test_executemany() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        log.clear()
        await conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])

    assert any(r.startswith("sql:users") and k == "write" for r, k in log.events)


class TestExecutemanyPatching:
    def test_patched_executemany_uses_dpor_schedule_and_suppress(self) -> None:
        """_patched_executemany delegates DPOR scheduling to _dpor_schedule_and_suppress_async."""
        import ast
        import inspect

        source = inspect.getsource(sql_cursor_async_mod)
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_patched_executemany":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Name) and func.id == "_dpor_schedule_and_suppress_async":
                            found = True
                            break
        assert found, "_patched_executemany should delegate to _dpor_schedule_and_suppress_async"

    def test_dpor_schedule_and_suppress_acquires_and_releases_row_locks(self) -> None:
        """_dpor_schedule_and_suppress_async acquires row locks and releases on exception."""
        import inspect

        source = inspect.getsource(sql_cursor_async_mod._dpor_schedule_and_suppress_async)
        assert "_acquire_pending_row_locks" in source
        assert "_release_dpor_row_locks" in source


# ---------------------------------------------------------------------------
# 3. Cursor-level interception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_execute_reports() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE t (id INTEGER)")
        log.clear()
        cursor = await conn.execute("SELECT * FROM t")
        assert cursor is not None

    assert any(r.startswith("sql:t") and k == "read" for r, k in log.events)


# ---------------------------------------------------------------------------
# 4. Transaction grouping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transaction_begin_commit_groups_reports() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:", isolation_level=None) as conn:
        await conn.execute("CREATE TABLE accounts (id INTEGER, balance REAL)")
        await conn.execute("INSERT INTO accounts VALUES (1, 100.0)")
        log.clear()

        await conn.execute("BEGIN")
        assert len(log.events) == 0  # BEGIN itself doesn't report data access

        await conn.execute("UPDATE accounts SET balance = 200.0 WHERE id = 1")
        # Buffered during transaction — not reported yet
        assert len(log.events) == 0

        await conn.execute("COMMIT")
        # Now flushed
        assert len(log.events) > 0
        assert any(r.startswith("sql:accounts") for r, _ in log.events)


@pytest.mark.asyncio
async def test_transaction_rollback_discards_reports() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:", isolation_level=None) as conn:
        await conn.execute("CREATE TABLE accounts (id INTEGER, balance REAL)")
        log.clear()

        await conn.execute("BEGIN")
        await conn.execute("INSERT INTO accounts VALUES (1, 100.0)")
        await conn.execute("ROLLBACK")

        # Rollback should discard buffered events
        assert len(log.events) == 0


@pytest.mark.parametrize("operation", ["COMMIT", "ROLLBACK"])
@pytest.mark.asyncio
async def test_failed_textual_tx_end_preserves_modeled_transaction(operation: str) -> None:
    """An async driver failure must not finalize the modeled transaction."""
    set_io_reporter(IOLog())
    _io_tls._in_transaction = True
    _io_tls._is_autobegin = True
    _io_tls._tx_buffer = [("sql:accounts", "write")]
    _io_tls._tx_savepoints = {"before": 0}
    _io_tls._held_row_locks = {"sql:accounts:id=1"}

    async def fail(_cursor: object, _sql: str) -> None:
        raise RuntimeError(f"physical async {operation.lower()} failed")

    try:
        with pytest.raises(RuntimeError, match="physical async"):
            await _intercept_execute_async(fail, object(), operation)

        assert _io_tls._in_transaction is True
        assert _io_tls._is_autobegin is True
        assert _io_tls._tx_buffer == [("sql:accounts", "write")]
        assert _io_tls._tx_savepoints == {"before": 0}
        assert _io_tls._held_row_locks == {"sql:accounts:id=1"}
    finally:
        set_io_reporter(None)
        for attr in ("_in_transaction", "_is_autobegin", "_tx_buffer", "_tx_savepoints", "_held_row_locks"):
            if hasattr(_io_tls, attr):
                delattr(_io_tls, attr)


# ---------------------------------------------------------------------------
# 5. Row-level predicate extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_level_predicate_in_resource_id() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        log.clear()

        await conn.execute("SELECT * FROM users WHERE id = ?", (42,))

    # With parameter resolution, should produce a row-level resource ID
    ids = log.resource_ids
    assert len(ids) > 0
    assert any("42" in r for r in ids), f"Expected row-level ID with '42', got {ids}"


@pytest.mark.asyncio
async def test_different_row_predicates_produce_different_ids() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        log.clear()

        await conn.execute("SELECT * FROM users WHERE id = ?", (1,))
        await conn.execute("SELECT * FROM users WHERE id = ?", (2,))

    ids = log.resource_ids
    # Should have different resource IDs for different predicates
    assert len(set(ids)) >= 2, f"Expected different IDs for different rows, got {ids}"


# ---------------------------------------------------------------------------
# 6. _report_sql_access shared helper (unit tests)
# ---------------------------------------------------------------------------


def test_report_sql_access_returns_true_for_data_sql() -> None:
    log = IOLog()
    set_io_reporter(log)
    assert _report_sql_access("SELECT * FROM users") is True
    assert any(r.startswith("sql:users") and k == "read" for r, k in log.events)


def test_report_sql_access_returns_false_without_reporter() -> None:
    set_io_reporter(None)
    assert _report_sql_access("SELECT * FROM users") is False


def test_report_sql_access_reports_opaque_database_write_for_non_string() -> None:
    log = IOLog()
    set_io_reporter(log)
    assert _report_sql_access(12345) is True
    assert log.events == [("sql:__database__", "write")]


def test_report_sql_access_handles_tx_control() -> None:
    log = IOLog()
    set_io_reporter(log)
    assert _report_sql_access("BEGIN") is True
    assert len(log.events) == 0  # BEGIN doesn't produce data access events


# ---------------------------------------------------------------------------
# 7. _intercept_execute_async (unit tests with mock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intercept_async_calls_original() -> None:
    """The async interceptor should call and await the original method."""
    call_log: list[tuple[Any, ...]] = []

    async def fake_execute(self: Any, sql: Any, params: Any = None) -> str:
        call_log.append((sql, params))
        return "ok"

    log = IOLog()
    set_io_reporter(log)

    result = await _intercept_execute_async(fake_execute, None, "SELECT * FROM t", None, paramstyle="qmark")
    assert result == "ok"
    assert call_log == [("SELECT * FROM t", None)]
    assert any(r.startswith("sql:t") and k == "read" for r, k in log.events)


@pytest.mark.asyncio
async def test_intercept_async_passes_parameters() -> None:
    call_log: list[tuple[Any, ...]] = []

    async def fake_execute(self: Any, sql: Any, params: Any = None) -> str:
        call_log.append((sql, params))
        return "ok"

    log = IOLog()
    set_io_reporter(log)

    result = await _intercept_execute_async(
        fake_execute, None, "SELECT * FROM t WHERE id = ?", (1,), paramstyle="qmark"
    )
    assert result == "ok"
    assert call_log == [("SELECT * FROM t WHERE id = ?", (1,))]


@pytest.mark.asyncio
async def test_intercept_async_no_reporter() -> None:
    """Without a reporter, interceptor should still call the original."""
    call_log: list[str] = []

    async def fake_execute(self: Any, sql: Any) -> str:
        call_log.append(sql)
        return "done"

    set_io_reporter(None)
    result = await _intercept_execute_async(fake_execute, None, "SELECT 1")
    assert result == "done"
    assert call_log == ["SELECT 1"]


# ---------------------------------------------------------------------------
# 8. Functional: real queries through patched aiosqlite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_query_workflow() -> None:
    """Verify SQL actually executes and returns correct results through patching."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        await conn.execute("INSERT INTO users VALUES (2, 'Bob')")
        await conn.commit()

        log.clear()
        cursor = await conn.execute("SELECT name FROM users ORDER BY id")
        rows = await cursor.fetchall()

    assert rows == [("Alice",), ("Bob",)]
    assert any(r.startswith("sql:users") and k == "read" for r, k in log.events)


@pytest.mark.asyncio
async def test_insert_select_roundtrip() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
        log.clear()

        await conn.execute("INSERT INTO items VALUES (1, 'test')")
        cursor = await conn.execute("SELECT val FROM items WHERE id = ?", (1,))
        rows = await cursor.fetchall()

    assert rows == [("test",)]
    write_events = [e for e in log.events if e[1] == "write"]
    read_events = [e for e in log.events if e[1] == "read"]
    assert len(write_events) >= 1
    assert len(read_events) >= 1


# ---------------------------------------------------------------------------
# 9. SELECT FOR UPDATE lock intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_for_update_reports_write() -> None:
    """FOR UPDATE intent is detected at the parsing layer (SQLite doesn't support it)."""
    log = IOLog()
    set_io_reporter(log)

    # Use a mock to avoid SQLite syntax error (SQLite doesn't support FOR UPDATE)
    async def fake_execute(self: Any, sql: Any, params: Any = None) -> None:
        pass

    await _intercept_execute_async(
        fake_execute, None, "SELECT * FROM accounts WHERE id = 1 FOR UPDATE", paramstyle="qmark"
    )

    write_events = [(r, k) for r, k in log.events if k == "write"]
    assert len(write_events) >= 1


# ---------------------------------------------------------------------------
# 10. Multiple tables in single query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_into_select_reports_both() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE src (id INTEGER, val TEXT)")
        await conn.execute("CREATE TABLE dst (id INTEGER, val TEXT)")
        log.clear()
        await conn.execute("INSERT INTO dst SELECT * FROM src")

    tables_reported = {r.split(":")[1] for r, _ in log.events if r.startswith("sql:")}
    assert "src" in tables_reported
    assert "dst" in tables_reported


# ---------------------------------------------------------------------------
# 11. Non-SQL operations passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pragma_no_data_access() -> None:
    """Statements like PRAGMA conservatively report opaque database access."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:") as conn:
        log.clear()
        await conn.execute("PRAGMA journal_mode")

    assert log.events == [("sql:__database__", "write")]


# ---------------------------------------------------------------------------
# 12. Savepoint support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_savepoint_rollback_to_truncates_buffer() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()

    async with aiosqlite.connect(":memory:", isolation_level=None) as conn:
        await conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        log.clear()

        await conn.execute("BEGIN")
        await conn.execute("INSERT INTO t VALUES (1, 'a')")
        await conn.execute("SAVEPOINT sp1")
        await conn.execute("INSERT INTO t VALUES (2, 'b')")
        await conn.execute("ROLLBACK TO sp1")
        await conn.execute("COMMIT")

    # Only the first INSERT's events should be flushed (second was rolled back).
    # Each INSERT produces table-level + alias + sequence writes.
    write_events = [(r, k) for r, k in log.events if k == "write"]
    write_resources = {r for r, _ in write_events}
    # First INSERT's resources survive; second INSERT's do not
    assert "sql:t" in write_resources or any(r.startswith("sql:t:") for r in write_resources)
    # No events from the second INSERT (VALUES (2, 'b'))
    assert not any("ins1" in r for r, _ in write_events)


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_string_operation() -> None:
    """Empty SQL string should not crash."""
    log = IOLog()
    set_io_reporter(log)
    # Direct call to the interception function with empty string
    call_log: list[str] = []

    async def fake_execute(self: Any, sql: Any) -> None:
        call_log.append(sql)

    await _intercept_execute_async(fake_execute, None, "")
    assert call_log == [""]


@pytest.mark.asyncio
async def test_non_string_operation_passthrough() -> None:
    """Non-string operations pass through after an opaque database report."""
    call_log: list[Any] = []

    async def fake_execute(self: Any, op: Any) -> str:
        call_log.append(op)
        return "ok"

    log = IOLog()
    set_io_reporter(log)
    result = await _intercept_execute_async(fake_execute, None, 42)  # type: ignore[arg-type]
    assert result == "ok"
    assert log.events == [("sql:__database__", "write")]


class TestAsyncUpdateZeroRowRelease:
    """Async PostgreSQL zero-row handling mirrors transaction lock ownership."""

    @pytest.mark.asyncio
    async def test_zero_row_update_releases_only_current_statement_row_lock(self) -> None:
        from frontrun._io_detection import (
            set_dpor_scheduler_task,
            set_dpor_thread_id_task,
            set_tx_store_task,
        )

        class FakeConnection:
            autocommit = False

        FakeConnection.__module__ = "psycopg.connection"

        class FakeCursor:
            connection = FakeConnection()
            rowcount = -1

        class FakeScheduler:
            def __init__(self, prior: str) -> None:
                self.held = {prior}
                self.acquired: list[str] = []
                self.release_calls: list[list[str] | None] = []

            def report_and_wait(self, _frame: object, _task_id: int) -> bool:
                return True

            def acquire_row_locks(self, _task_id: int, resources: list[str]) -> list[str]:
                self.acquired.extend(resources)
                self.held.update(resources)
                return resources

            def release_row_locks(self, _task_id: int, resources: list[str] | None = None) -> None:
                self.release_calls.append(resources)
                if resources is None:
                    self.held.clear()
                else:
                    self.held.difference_update(resources)

        prior = "sql:accounts:(('id', '1'),)"
        scheduler = FakeScheduler(prior)
        store = set_tx_store_task()
        store._in_transaction = True
        store._is_autobegin = True
        store._held_row_locks = {prior}
        store._pending_row_locks = []
        set_dpor_scheduler_task(scheduler)
        set_dpor_thread_id_task(0)
        set_io_reporter(IOLog())

        async def execute(cursor: FakeCursor, _operation: object, _parameters: object) -> None:
            cursor.rowcount = 0

        try:
            await _intercept_execute_async(
                execute,
                FakeCursor(),
                "UPDATE accounts SET balance = %s WHERE id = %s",
                (100, 2),
                paramstyle="format",
            )

            assert len(scheduler.acquired) == 1
            assert scheduler.release_calls == [scheduler.acquired]
            assert scheduler.held == {prior}
            assert store._held_row_locks == {prior}
        finally:
            set_dpor_scheduler_task(None)
            set_dpor_thread_id_task(None)
            set_tx_store_task()
            set_io_reporter(None)

    def test_source_has_update_match(self) -> None:
        """_intercept_execute_async should extract _RE_UPDATE_TABLE match like the sync version."""
        import inspect

        source = inspect.getsource(_intercept_execute_async)
        assert "_RE_UPDATE_TABLE" in source or "update_match" in source, (
            "_intercept_execute_async should check for UPDATE statements "
            "to release row locks on 0-row matches (Defect #6 fix)"
        )

    @pytest.mark.parametrize(
        ("method_name", "result", "releases"),
        [("fetch", [], True), ("fetchrow", None, True), ("fetchval", None, False)],
    )
    async def test_asyncpg_result_shape_controls_statement_lock_release(
        self, method_name: str, result: Any, releases: bool
    ) -> None:
        from frontrun._io_detection import (
            set_dpor_scheduler_task,
            set_dpor_thread_id_task,
            set_tx_store_task,
        )
        from frontrun._sql_cursor_async import _intercept_asyncpg_execute

        class FakeScheduler:
            def __init__(self) -> None:
                self.acquired: list[str] = []
                self.release_calls: list[list[str] | None] = []

            def report_and_wait(self, _frame: object, _task_id: int) -> bool:
                return True

            def acquire_row_locks(self, _task_id: int, resources: list[str]) -> list[str]:
                self.acquired.extend(resources)
                return resources

            def release_row_locks(self, _task_id: int, resources: list[str] | None = None) -> None:
                self.release_calls.append(resources)

        class FakeConnection:
            pass

        FakeConnection.__module__ = "asyncpg.connection"
        scheduler = FakeScheduler()
        store = set_tx_store_task()
        store._in_transaction = True
        store._is_autobegin = True
        store._held_row_locks = set()
        store._pending_row_locks = []
        set_dpor_scheduler_task(scheduler)
        set_dpor_thread_id_task(0)
        set_io_reporter(IOLog())

        async def execute(_connection: object, _operation: object) -> Any:
            return result

        try:
            actual = await _intercept_asyncpg_execute(
                execute,
                FakeConnection(),
                "UPDATE accounts SET balance = 100 WHERE id = 2 RETURNING id",
                method_name=method_name,
            )
            assert actual == result
            assert len(scheduler.acquired) == 1
            if releases:
                assert scheduler.release_calls == [scheduler.acquired]
                assert store._held_row_locks == set()
            else:
                assert scheduler.release_calls == []
                assert store._held_row_locks == set(scheduler.acquired)
        finally:
            set_dpor_scheduler_task(None)
            set_dpor_thread_id_task(None)
            set_tx_store_task()
            set_io_reporter(None)


class TestAsyncpgExecutemanyDbObj:
    """Verify asyncpg _patched_executemany passes db_obj to _report_sql_access."""

    def test_executemany_passes_db_obj(self) -> None:
        """_patched_executemany should pass db_obj=self for correct database scoping."""
        import inspect

        source = inspect.getsource(sql_cursor_async_mod)
        # Find the _patched_executemany function definition
        idx = source.find("async def _patched_executemany")
        assert idx != -1, "_patched_executemany not found in source"
        # Extract the function body (next ~15 lines)
        func_body = source[idx : idx + 600]
        assert "db_obj=self" in func_body, "_patched_executemany should pass db_obj=self to _report_sql_access"

    def test_executemany_has_dpor_scheduling_point(self) -> None:
        """_patched_executemany should have a DPOR scheduling point via the shared helper."""
        import inspect

        # The scheduling point is now in _dpor_schedule_and_suppress_async
        source = inspect.getsource(sql_cursor_async_mod._dpor_schedule_and_suppress_async)
        assert "_get_dpor_context" in source, (
            "_dpor_schedule_and_suppress_async should call _get_dpor_context for DPOR scheduling"
        )


# ---------------------------------------------------------------------------
# Bug: async executemany INSERT does not call _record_uncaptured_insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executemany_insert_records_uncaptured() -> None:
    """Async executemany INSERT must record the table as uncaptured (finding 10e).

    The sync path calls _record_uncaptured_insert for executemany INSERTs
    so the nondeterminism guard catches uncaptured row IDs.  The async path
    at _sql_cursor_async.py only handles the single-row case and silently
    skips the executemany INSERT tracking.
    """
    from frontrun import _sql_insert_tracker

    _sql_insert_tracker.clear_insert_tracker()
    log = IOLog()
    set_io_reporter(log)
    patch_sql_async()
    try:
        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            log.clear()
            await conn.executemany("INSERT INTO users (name) VALUES (?)", [("a",), ("b",), ("c",)])
        assert "users" in _sql_insert_tracker.get_uncaptured_tables(), (
            "async executemany INSERT must record the table as uncaptured"
        )
    finally:
        _sql_insert_tracker.clear_insert_tracker()


# ---------------------------------------------------------------------------
# asyncpg PreparedStatement patching (SQLAlchemy asyncpg dialect blind spot)
# ---------------------------------------------------------------------------


class TestAsyncpgPreparedStatementPatching:
    """SQLAlchemy's asyncpg dialect executes every statement through
    ``Connection.prepare()`` → ``PreparedStatement.fetch()``.  Those methods
    (and the cursor factories) must be patched, or the SQL never reaches
    ``_report_sql_access`` and DPOR certifies racy code as clean.
    """

    def test_patch_wraps_prepared_statement_and_cursor_methods(self) -> None:
        asyncpg = pytest.importorskip("asyncpg")
        from asyncpg import prepared_stmt as ps_mod

        ps_cls = ps_mod.PreparedStatement
        method_names = [
            name
            for name in ("fetch", "fetchmany", "fetchrow", "fetchval", "executemany", "cursor")
            if hasattr(ps_cls, name)
        ]
        ps_originals = {name: getattr(ps_cls, name) for name in method_names}
        conn_cursor_orig = asyncpg.Connection.cursor

        patch_sql_async()
        try:
            for name, orig in ps_originals.items():
                assert getattr(ps_cls, name) is not orig, f"PreparedStatement.{name} must be patched"
            assert asyncpg.Connection.cursor is not conn_cursor_orig, "Connection.cursor must be patched"
        finally:
            unpatch_sql_async()

        for name, orig in ps_originals.items():
            assert getattr(ps_cls, name) is orig, f"PreparedStatement.{name} must be restored on unpatch"
        assert asyncpg.Connection.cursor is conn_cursor_orig, "Connection.cursor must be restored on unpatch"
