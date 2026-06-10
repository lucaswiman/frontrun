"""Tests for DBAPI cursor monkey-patching.

Uses sqlite3 (always available) to test the patching mechanism.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, call

import pytest

import frontrun._sql_cursor as sql_cursor_mod
from frontrun._io_detection import _io_tls, set_io_reporter
from frontrun._sql_cursor import (
    _ORIGINAL_METHODS,
    _PATCHES,
    _suppress_lock,
    _suppress_tids,
    is_tid_suppressed,
    patch_sql,
    unpatch_sql,
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


def _make_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with test tables (uses patched connect if patched)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    # Use raw execute to avoid polluting logs during setup
    orig_execute = sqlite3.Cursor.execute
    cur = conn.cursor()
    orig_execute(cur, "INSERT INTO users VALUES (1, 'Alice', 30)")
    orig_execute(cur, "INSERT INTO users VALUES (2, 'Bob', 25)")
    orig_execute(cur, "INSERT INTO orders VALUES (1, 1, 99.99)")
    conn.commit()
    return conn


def _make_fresh_db() -> sqlite3.Connection:
    """Create a fresh in-memory db, bypassing any patching to avoid noise during setup."""
    orig_connect = getattr(sql_cursor_mod, "_get_orig_sqlite3_connect", lambda: None)()
    if orig_connect is not None:
        conn = orig_connect(":memory:")
    else:
        # Not patched yet or already unpatched
        conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    conn.execute("INSERT INTO orders VALUES (1, 1, 99.99)")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_sql_patch() -> Generator[None, None, None]:
    """Ensure SQL patching is cleaned up between tests."""
    yield
    unpatch_sql()
    _ORIGINAL_METHODS.clear()
    _PATCHES.clear()
    _suppress_tids.clear()
    sql_cursor_mod._sql_patched = False
    set_io_reporter(None)
    if hasattr(_io_tls, "_sql_suppress"):
        _io_tls._sql_suppress = False
    _io_tls._in_transaction = False
    _io_tls._is_autobegin = False
    _io_tls._tx_buffer = []
    _io_tls._tx_savepoints = {}
    _io_tls._pending_row_locks = []


# ---------------------------------------------------------------------------
# 1. Basic patching/unpatching
# ---------------------------------------------------------------------------


def test_patch_patches_sqlite3_connect() -> None:
    orig_connect = sqlite3.connect
    patch_sql()
    assert sqlite3.connect is not orig_connect


def test_patch_produces_traced_connection() -> None:
    patch_sql()
    conn = sqlite3.connect(":memory:")
    assert "Traced" in type(conn).__name__
    conn.close()


def test_patch_produces_traced_cursor() -> None:
    patch_sql()
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    assert "Traced" in type(cur).__name__
    conn.close()


def test_unpatch_restores_original_connect() -> None:
    orig_connect = sqlite3.connect
    patch_sql()
    unpatch_sql()
    assert sqlite3.connect is orig_connect


def test_unpatch_restores_plain_connection() -> None:
    patch_sql()
    unpatch_sql()
    conn = sqlite3.connect(":memory:")
    assert type(conn) is sqlite3.Connection
    conn.close()


def test_double_patch_is_idempotent() -> None:
    patch_sql()
    connect_after_first = sqlite3.connect
    patch_sql()
    assert sqlite3.connect is connect_after_first


def test_double_unpatch_is_idempotent() -> None:
    patch_sql()
    unpatch_sql()
    unpatch_sql()  # should not raise


def test_patch_unpatch_cycle() -> None:
    orig_connect = sqlite3.connect
    patch_sql()
    assert sqlite3.connect is not orig_connect

    unpatch_sql()
    assert sqlite3.connect is orig_connect

    # Reset the patched flag so we can re-patch
    sql_cursor_mod._sql_patched = False
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()

    patch_sql()
    assert sqlite3.connect is not orig_connect


def test_patch_sets_sql_patched_flag() -> None:
    assert sql_cursor_mod._sql_patched is False
    patch_sql()
    assert sql_cursor_mod._sql_patched is True


def test_unpatch_clears_sql_patched_flag() -> None:
    patch_sql()
    unpatch_sql()
    assert sql_cursor_mod._sql_patched is False


def test_unpatch_without_patch_is_safe() -> None:
    assert sql_cursor_mod._sql_patched is False
    unpatch_sql()  # should not raise or change state
    assert sql_cursor_mod._sql_patched is False


def test_patches_list_populated_after_patch() -> None:
    patch_sql()
    # At minimum, sqlite3.connect should be in _PATCHES
    sqlite3_patches = [p for p in _PATCHES if p[0] is sqlite3 and p[1] == "connect"]
    assert len(sqlite3_patches) == 1


def test_patches_cleared_after_unpatch() -> None:
    patch_sql()
    unpatch_sql()
    assert len(_PATCHES) == 0


def test_original_methods_populated_after_patch() -> None:
    patch_sql()
    # At minimum sqlite3.Cursor execute/executemany are tracked
    assert (sqlite3.Cursor, "execute") in _ORIGINAL_METHODS
    assert (sqlite3.Cursor, "executemany") in _ORIGINAL_METHODS


def test_original_methods_cleared_after_unpatch() -> None:
    patch_sql()
    unpatch_sql()
    assert len(_ORIGINAL_METHODS) == 0


# ---------------------------------------------------------------------------
# 2. SQL interception with reporter
# ---------------------------------------------------------------------------


def test_select_reports_read() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    conn.execute("SELECT * FROM users")

    events = log.events_for_table("users")
    assert len(events) >= 1
    assert any(k == "read" for _, k in events)
    conn.close()


def test_insert_reports_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    log.clear()
    conn.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")

    events = log.events_for_table("users")
    assert len(events) >= 1
    assert any(k == "write" for _, k in events)
    conn.close()


def test_update_reports_read_and_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    log.clear()
    conn.execute("UPDATE users SET age = 31 WHERE id = 1")

    events = log.events_for_table("users")
    kinds = [k for _, k in events]
    assert "read" in kinds
    assert "write" in kinds
    conn.close()


def test_delete_reports_read_and_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    log.clear()
    conn.execute("DELETE FROM users WHERE id = 1")

    events = log.events_for_table("users")
    kinds = [k for _, k in events]
    assert "read" in kinds
    assert "write" in kinds
    conn.close()


def test_join_reports_multiple_tables() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, total REAL)")
    log.clear()
    conn.execute("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")

    user_events = log.events_for_table("users")
    order_events = log.events_for_table("orders")
    assert len(user_events) >= 1
    assert len(order_events) >= 1
    assert all(k == "read" for _, k in user_events)
    assert all(k == "read" for _, k in order_events)
    conn.close()


def test_no_reporter_still_executes() -> None:
    patch_sql()
    set_io_reporter(None)  # ensure no reporter

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    assert len(rows) == 2
    conn.close()


def test_reporter_called_with_correct_resource_id_format() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE mytable (x INTEGER)")
    log.clear()
    conn.execute("SELECT x FROM mytable")

    assert all(r == "sql:mytable" or r.startswith("sql:mytable:") for r, _ in log.events)
    assert any((r == "sql:mytable" or r.startswith("sql:mytable:")) and k == "read" for r, k in log.events)
    conn.close()


def test_select_where_clause_still_reports_table() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    log.clear()
    conn.execute("SELECT name FROM users WHERE age > 20")

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    conn.close()


def test_reporter_called_once_per_table_per_execute() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    conn.execute("SELECT * FROM users")
    # Table-level read + :seq read for phantom detection
    user_reads = [(r, k) for r, k in log.events if (r == "sql:users" or r.startswith("sql:users:")) and k == "read"]
    assert len(user_reads) == 2
    conn.close()


def test_multiple_executes_each_reported() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    conn.execute("SELECT * FROM users")
    conn.execute("SELECT * FROM users")

    # Each SELECT produces a table-level read + :seq read for phantom detection (2 per execute)
    user_reads = [(r, k) for r, k in log.events if (r == "sql:users" or r.startswith("sql:users:")) and k == "read"]
    assert len(user_reads) == 4
    conn.close()


def test_different_tables_reported_independently() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER, total REAL)")
    log.clear()
    conn.execute("SELECT * FROM users")
    conn.execute("SELECT * FROM orders")

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    assert any((r == "sql:orders" or r.startswith("sql:orders:")) and k == "read" for r, k in log.events)
    conn.close()


# ---------------------------------------------------------------------------
# 3. Suppression infrastructure
# ---------------------------------------------------------------------------


def test_sql_suppress_flag_set_during_original_execute() -> None:
    """_io_tls._sql_suppress is True while the original execute runs."""
    suppress_seen: list[bool] = []

    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    # Wrap the stored original to spy on it.
    # Since TracedCursor looks up _ORIGINAL_METHODS at call time, replacing
    # the stored value here will affect subsequent execute calls.
    sqlite3_cursor_key = (sqlite3.Cursor, "execute")
    old_original = _ORIGINAL_METHODS[sqlite3_cursor_key]

    def spy_original(self: Any, operation: Any, parameters: Any = None) -> Any:
        suppress_seen.append(getattr(_io_tls, "_sql_suppress", False))
        if parameters is not None:
            return old_original(self, operation, parameters)
        return old_original(self, operation)

    _ORIGINAL_METHODS[sqlite3_cursor_key] = spy_original

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INT)")
    suppress_seen.clear()  # clear the CREATE TABLE event
    conn.execute("SELECT * FROM t")
    conn.close()

    assert any(suppress_seen), f"suppress flag should be True during original execute, got: {suppress_seen}"


def test_sql_suppress_flag_cleared_after_execute() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("SELECT * FROM users")

    assert getattr(_io_tls, "_sql_suppress", False) is False
    conn.close()


def test_suppress_tid_added_during_execute() -> None:
    tids_during: list[set[int]] = []

    patch_sql()

    sqlite3_cursor_key = (sqlite3.Cursor, "execute")
    old_original = _ORIGINAL_METHODS[sqlite3_cursor_key]

    def spy_original(self: Any, operation: Any, parameters: Any = None) -> Any:
        with _suppress_lock:
            tids_during.append(set(_suppress_tids))
        if parameters is not None:
            return old_original(self, operation, parameters)
        return old_original(self, operation)

    _ORIGINAL_METHODS[sqlite3_cursor_key] = spy_original

    log = IOLog()
    set_io_reporter(log)

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INT)")
    tids_during.clear()  # clear the CREATE TABLE event
    conn.execute("SELECT * FROM t")
    conn.close()

    current_tid = threading.get_native_id()
    assert any(current_tid in snap for snap in tids_during), (
        f"Expected tid {current_tid} in _suppress_tids during execute, got: {tids_during}"
    )


def test_suppress_tid_removed_after_execute() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("SELECT * FROM users")
    conn.close()

    current_tid = threading.get_native_id()
    with _suppress_lock:
        assert current_tid not in _suppress_tids


def test_is_tid_suppressed_false_for_unknown_tid() -> None:
    assert is_tid_suppressed(99999999) is False


def test_is_tid_suppressed_true_when_tid_in_set() -> None:
    fake_tid = 12345678
    with _suppress_lock:
        _suppress_tids.add(fake_tid)
    try:
        assert is_tid_suppressed(fake_tid) is True
    finally:
        with _suppress_lock:
            _suppress_tids.discard(fake_tid)


def test_is_tid_suppressed_thread_safe() -> None:
    results: list[bool] = []

    def worker() -> None:
        results.append(is_tid_suppressed(threading.get_native_id()))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results == [False]


def test_suppress_cleaned_on_exception() -> None:
    patch_sql()

    sqlite3_cursor_key = (sqlite3.Cursor, "execute")
    old_original = _ORIGINAL_METHODS[sqlite3_cursor_key]

    def raising_original(self: Any, operation: Any, parameters: Any = None) -> Any:
        raise RuntimeError("simulated DB error")

    _ORIGINAL_METHODS[sqlite3_cursor_key] = raising_original

    log = IOLog()
    set_io_reporter(log)

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    # Use a parseable SQL so the reporter fires and suppression is activated,
    # then the raising_original raises, triggering cleanup in the finally block.
    with pytest.raises(RuntimeError, match="simulated DB error"):
        cur.execute("SELECT * FROM some_table")

    current_tid = threading.get_native_id()
    with _suppress_lock:
        assert current_tid not in _suppress_tids
    assert getattr(_io_tls, "_sql_suppress", False) is False
    conn.close()

    # Restore so unpatch works cleanly
    _ORIGINAL_METHODS[sqlite3_cursor_key] = old_original


def test_suppress_not_set_when_no_tables_parsed() -> None:
    """Suppression should not be activated when SQL has no parseable tables."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")

    assert getattr(_io_tls, "_sql_suppress", False) is False
    assert len(log.events) == 0
    conn.close()


# ---------------------------------------------------------------------------
# 4. Actual SQL execution
# ---------------------------------------------------------------------------


def test_select_actually_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    cur = conn.cursor()
    cur.execute("SELECT name, age FROM users WHERE id = 1")
    row = cur.fetchone()

    assert row == ("Alice", 30)
    conn.close()


def test_select_returns_all_rows() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("INSERT INTO users VALUES (1)")
    conn.execute("INSERT INTO users VALUES (2)")
    cur = conn.cursor()
    cur.execute("SELECT id FROM users ORDER BY id")
    rows = cur.fetchall()

    assert rows == [(1,), (2,)]
    conn.close()


def test_insert_actually_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    conn.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]  # type: ignore[index]
    assert count == 2
    conn.close()


def test_update_actually_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 30)")
    conn.execute("UPDATE users SET age = 99 WHERE id = 1")
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT age FROM users WHERE id = 1")
    age = cur.fetchone()[0]  # type: ignore[index]
    assert age == 99
    conn.close()


def test_delete_actually_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("INSERT INTO users VALUES (1)")
    conn.execute("INSERT INTO users VALUES (2)")
    conn.execute("DELETE FROM users WHERE id = 2")
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]  # type: ignore[index]
    assert count == 1
    conn.close()


def test_parameterized_query_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE id = ?", (1,))
    row = cur.fetchone()
    assert row == ("Alice",)
    conn.close()


def test_parameterized_insert_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (?, ?, ?)", (4, "Dave", 40))
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE id = ?", (4,))
    row = cur.fetchone()
    assert row == ("Dave",)
    conn.close()


def test_parameterized_query_reports_correctly() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    conn.execute("SELECT name FROM users WHERE id = ?", (1,))

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    conn.close()


def test_executemany_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    data = [(10, "Eve", 22), (11, "Frank", 28)]
    conn.executemany("INSERT INTO users VALUES (?, ?, ?)", data)
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]  # type: ignore[index]
    assert count == 2
    conn.close()


def test_executemany_reports_write() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    log.clear()
    data = [(10, "Eve", 22), (11, "Frank", 28)]
    conn.executemany("INSERT INTO users VALUES (?, ?, ?)", data)

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "write" for r, k in log.events)
    conn.close()


def test_execute_returns_cursor() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("INSERT INTO users VALUES (1)")
    conn.execute("INSERT INTO users VALUES (2)")
    cur = conn.execute("SELECT * FROM users")
    assert cur is not None
    rows = cur.fetchall()
    assert len(rows) == 2
    conn.close()


def test_cursor_execute_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("INSERT INTO users VALUES (1)")
    conn.execute("INSERT INTO users VALUES (2)")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    assert len(rows) == 2
    conn.close()


def test_executemany_via_cursor_works() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, val TEXT)")
    cur = conn.cursor()
    cur.executemany("INSERT INTO users VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]  # type: ignore[index]
    assert count == 3
    conn.close()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


def test_non_string_operation_skips_parsing() -> None:
    log = IOLog()
    set_io_reporter(log)

    from frontrun._sql_cursor import _intercept_execute

    fake_original = MagicMock(return_value=None)

    class FakeCursor:
        pass

    # Call _intercept_execute with bytes — should skip parsing and call original
    _intercept_execute(fake_original, FakeCursor(), b"SELECT 1")
    fake_original.assert_called_once()
    assert len(log.events) == 0


def test_unparseable_sql_falls_through() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    with pytest.raises(Exception):  # noqa: B017
        cur.execute("XYZZY this is not valid SQL at all blorp")

    assert len(log.events) == 0
    assert getattr(_io_tls, "_sql_suppress", False) is False
    with _suppress_lock:
        assert threading.get_native_id() not in _suppress_tids
    conn.close()


def test_empty_sql_string_falls_through() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # sqlite3 may or may not raise on empty string; either way no tables are parsed
    try:
        cur.execute("")
    except Exception:  # noqa: BLE001
        pass

    # No table events should be reported for empty SQL
    assert len(log.events) == 0
    assert getattr(_io_tls, "_sql_suppress", False) is False
    conn.close()


def test_pragma_not_reported() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")

    assert len(log.events) == 0
    conn.close()


def test_create_table_reported() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE foo (x INTEGER)")

    # DDL write + :seq write for phantom detection
    assert len(log.events) == 2
    resource_id, kind = log.events[0]
    assert kind == "write"
    assert resource_id.startswith("sql:foo")
    conn.close()


def test_no_reporter_select_does_not_crash() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.execute("INSERT INTO users VALUES (1)")
    conn.execute("INSERT INTO users VALUES (2)")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    assert len(rows) == 2
    conn.close()


def test_no_reporter_insert_does_not_crash() -> None:
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (5, 'Grace')")
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]  # type: ignore[index]
    assert count == 1
    conn.close()


def test_concurrent_patching_safe() -> None:
    """Multiple threads can use patched execute simultaneously."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    errors: list[Exception] = []
    results: list[list[Any]] = []
    lock = threading.Lock()

    def worker(thread_id: int) -> None:
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
            conn.execute("INSERT INTO t VALUES (?, ?)", (thread_id, f"thread{thread_id}"))
            cur = conn.cursor()
            cur.execute("SELECT id FROM t WHERE id = ?", (thread_id,))
            rows = cur.fetchall()
            with lock:
                results.append(rows)
            conn.close()
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent execution: {errors}"
    assert len(results) == 10
    # Each thread selected its own id
    for i, rows in enumerate(results):
        assert len(rows) == 1


def test_concurrent_suppression_cleanup() -> None:
    """Suppression TIDs are properly cleaned up even with concurrent threads."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    barrier = threading.Barrier(5)
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE t (x INT)")
            barrier.wait()
            conn.execute("SELECT * FROM t")
            conn.close()
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors: {errors}"

    with _suppress_lock:
        assert len(_suppress_tids) == 0


def test_reporter_tls_isolation() -> None:
    """Each thread sees only its own reporter."""
    patch_sql()

    main_log = IOLog()
    set_io_reporter(main_log)

    thread_events: list[tuple[str, str]] = []
    thread_main_events: list[tuple[str, str]] = []

    def worker() -> None:
        thread_log = IOLog()
        set_io_reporter(thread_log)
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE orders (id INTEGER)")
        conn.execute("SELECT * FROM orders")
        conn.close()
        with thread_log._lock:
            thread_events.extend(thread_log.events)
        with main_log._lock:
            thread_main_events.extend(main_log.events)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # Main thread reporter should not have received thread's events
    assert len(thread_main_events) == 0
    # Thread reporter should have received orders access
    assert any((r == "sql:orders" or r.startswith("sql:orders:")) and k == "read" for r, k in thread_events)


def test_sql_suppress_tls_isolation() -> None:
    """_sql_suppress flag is per-thread via TLS."""
    patch_sql()
    log = IOLog()
    set_io_reporter(log)

    suppress_in_thread: list[bool] = []

    def worker() -> None:
        suppress_in_thread.append(getattr(_io_tls, "_sql_suppress", False))

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert suppress_in_thread == [False]


# ---------------------------------------------------------------------------
# 6. _intercept_execute unit tests (white-box)
# ---------------------------------------------------------------------------


def test_intercept_execute_calls_original() -> None:
    original = MagicMock(return_value="result")
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    result = _intercept_execute(original, fake_self, "SELECT 1")
    original.assert_called_once_with(fake_self, "SELECT 1")
    assert result == "result"


def test_intercept_execute_passes_parameters() -> None:
    original = MagicMock(return_value=None)
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    _intercept_execute(original, fake_self, "SELECT * FROM t WHERE id = ?", (1,))
    original.assert_called_once_with(fake_self, "SELECT * FROM t WHERE id = ?", (1,))


def test_intercept_execute_no_parameters_omits_param_arg() -> None:
    original = MagicMock(return_value=None)
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    _intercept_execute(original, fake_self, "SELECT 1")
    assert original.call_args == call(fake_self, "SELECT 1")


def test_intercept_execute_reports_to_reporter() -> None:
    log = IOLog()
    set_io_reporter(log)

    original = MagicMock(return_value=None)
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    _intercept_execute(original, fake_self, "SELECT * FROM mytable")

    assert any((r == "sql:mytable" or r.startswith("sql:mytable:")) and k == "read" for r, k in log.events)


def test_intercept_execute_no_reporter_no_report() -> None:
    set_io_reporter(None)

    original = MagicMock(return_value=None)
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    _intercept_execute(original, fake_self, "SELECT * FROM mytable")

    original.assert_called_once()


def test_intercept_execute_exception_cleanup() -> None:
    """Exception from original execute cleans up suppression state."""
    log = IOLog()
    set_io_reporter(log)

    def raising_original(self: Any, operation: Any, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("DB exploded")

    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    with pytest.raises(ValueError, match="DB exploded"):
        _intercept_execute(raising_original, fake_self, "SELECT * FROM sometable")

    assert getattr(_io_tls, "_sql_suppress", False) is False
    with _suppress_lock:
        assert threading.get_native_id() not in _suppress_tids


def test_intercept_execute_bytes_skips_parsing() -> None:
    log = IOLog()
    set_io_reporter(log)

    original = MagicMock(return_value=None)
    fake_self = MagicMock()

    from frontrun._sql_cursor import _intercept_execute

    _intercept_execute(original, fake_self, b"SELECT * FROM t")

    original.assert_called_once()
    assert len(log.events) == 0


# ---------------------------------------------------------------------------
# 7. Integration: patching + real sqlite3 queries with reporter
# ---------------------------------------------------------------------------


def test_full_workflow_select_insert_update_delete() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    log.clear()

    conn.execute("SELECT * FROM users")
    conn.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
    conn.execute("UPDATE users SET age = 36 WHERE id = 3")
    conn.execute("DELETE FROM users WHERE id = 3")
    conn.commit()

    user_events = log.events_for_table("users")
    kinds = [k for _, k in user_events]
    assert kinds.count("read") >= 3  # SELECT + UPDATE + DELETE all read
    assert kinds.count("write") >= 3  # INSERT + UPDATE + DELETE all write
    conn.close()


def test_schema_qualified_table_stripped() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    # sqlite3 uses schema.table notation with "main" schema
    conn.execute("SELECT * FROM main.users")

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    conn.close()


def test_quoted_table_name_stripped() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute('CREATE TABLE "my_table" (x INTEGER)')
    log.clear()
    conn.execute('INSERT INTO "my_table" VALUES (1)')

    write_events = [(r, k) for r, k in log.events if k == "write"]
    assert any("my_table" in r for r, _ in write_events)
    conn.close()


def test_case_insensitive_sql_keywords() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    log.clear()
    conn.execute("select name from users")

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    conn.close()


def test_multiline_sql_works() -> None:
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    log.clear()
    conn.execute("""
        SELECT
            name,
            age
        FROM
            users
        WHERE
            age > 20
    """)

    assert any((r == "sql:users" or r.startswith("sql:users:")) and k == "read" for r, k in log.events)
    conn.close()


def test_existing_connection_not_traced() -> None:
    """Connections created before patching are NOT traced (by design)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INT)")

    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    # This connection was opened BEFORE patching, so it won't be traced
    conn.execute("SELECT * FROM t")
    # No events expected for pre-patch connections
    # (The reporter is set, but the connection uses the original cursor)
    assert not any((r == "sql:t" or r.startswith("sql:t:")) and k == "read" for r, k in log.events)
    conn.close()


def test_new_connection_after_patch_is_traced() -> None:
    """Connections created after patching ARE traced."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INT)")
    log.clear()
    conn.execute("SELECT * FROM t")

    assert any((r == "sql:t" or r.startswith("sql:t:")) and k == "read" for r, k in log.events)
    conn.close()


# ---------------------------------------------------------------------------
# Row-lock capture tests
# ---------------------------------------------------------------------------


def test_report_or_buffer_captures_pending_row_locks() -> None:
    """_report_or_buffer with force_immediate=True inside a tx captures resource IDs."""
    from frontrun._sql_cursor import _report_or_buffer

    log = IOLog()
    set_io_reporter(log)
    _io_tls._in_transaction = True
    _io_tls._pending_row_locks = []

    try:
        _report_or_buffer(log, "sql:users:(('id', 42))", "write", force_immediate=True)
        pending = getattr(_io_tls, "_pending_row_locks", [])
        assert "sql:users:(('id', 42))" in pending
        # Should also have reported immediately
        assert ("sql:users:(('id', 42))", "write") in log.events
    finally:
        _io_tls._in_transaction = False
        _io_tls._pending_row_locks = []


def test_report_or_buffer_no_capture_outside_tx() -> None:
    """_report_or_buffer with force_immediate=True does NOT track row locks outside a tx.

    When ``_in_transaction`` is False the DB releases locks immediately after the
    statement, so there's no blocking risk.  Autobegin detection (which sets
    ``_in_transaction=True``) happens earlier in ``_intercept_execute``, before
    ``_report_or_buffer`` is called.
    """
    from frontrun._sql_cursor import _report_or_buffer

    log = IOLog()
    set_io_reporter(log)
    _io_tls._in_transaction = False
    _io_tls._pending_row_locks = []

    try:
        _report_or_buffer(log, "sql:users:(('id', 42))", "write", force_immediate=True)
        pending = getattr(_io_tls, "_pending_row_locks", [])
        assert len(pending) == 0
    finally:
        _io_tls._pending_row_locks = []


def test_report_or_buffer_captures_writes_in_tx() -> None:
    """_report_or_buffer captures write-kind accesses in transactions for row lock arbitration.

    After the defect #6 fix, all writes inside transactions (not just SELECT FOR UPDATE)
    are tracked in _pending_row_locks to prevent cooperative scheduler deadlocks with PG row locks.
    """
    from frontrun._sql_cursor import _report_or_buffer

    log = IOLog()
    set_io_reporter(log)
    _io_tls._in_transaction = True
    _io_tls._tx_buffer = []
    _io_tls._pending_row_locks = []

    try:
        _report_or_buffer(log, "sql:users:(('id', 42))", "write", force_immediate=False)
        pending = getattr(_io_tls, "_pending_row_locks", [])
        assert len(pending) == 1
        assert pending[0] == "sql:users:(('id', 42))"
    finally:
        _io_tls._in_transaction = False
        _io_tls._tx_buffer = []
        _io_tls._pending_row_locks = []


def test_release_dpor_row_locks_no_scheduler() -> None:
    """_release_dpor_row_locks is a no-op when no scheduler is set."""
    from frontrun._sql_cursor import _release_dpor_row_locks

    # Should not raise
    _release_dpor_row_locks()


# ---------------------------------------------------------------------------
# Finding 3: connection.commit() / rollback() interception
# ---------------------------------------------------------------------------


def test_traced_cursor_execute_recovers_keyword_params() -> None:
    """TracedCursor must not silently drop keyword params (finding 10c).

    psycopg2 accepts ``execute(sql, vars=params)`` and pymysql
    ``execute(sql, args=params)``.  Those land in **kwargs; if dropped, the
    placeholders are never substituted for predicate extraction and the real
    driver call loses its parameters.
    """
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    cur = conn.cursor()
    log.clear()
    # sqlite3's TracedCursor.execute accepts parameters as a keyword via kwargs.
    cur.execute("SELECT * FROM users WHERE id = ?", parameters=(1,))
    rows = cur.fetchall()
    assert rows == [(1, "Alice")], "keyword params must reach the driver"
    conn.close()


def test_executemany_insert_records_uncaptured(monkeypatch: pytest.MonkeyPatch) -> None:
    """executemany INSERT must engage the uncaptured-insert determinism guard.

    executemany assigns one ID per row but lastrowid exposes only the final
    one, so per-row aliases can't be captured.  Previously this path skipped
    record_insert entirely, silently disabling the determinism guard (finding
    10e).  The table must now be recorded as uncaptured.
    """
    from frontrun import _sql_insert_tracker

    _sql_insert_tracker.clear_insert_tracker()
    log = IOLog()
    set_io_reporter(log)
    patch_sql()
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cur = conn.cursor()
        cur.executemany("INSERT INTO users (name) VALUES (?)", [("a",), ("b",), ("c",)])
        assert "users" in _sql_insert_tracker.get_uncaptured_tables(), (
            "executemany INSERT must record the table as uncaptured"
        )
        conn.close()
    finally:
        _sql_insert_tracker.clear_insert_tracker()


def test_wrap_connection_cursor_traces_explicit_factory() -> None:
    """conn.cursor(cursor_factory=...) must be wrapped into a traced subclass.

    Before the fix, an explicit per-cursor cursor_factory (e.g. psycopg2's
    RealDictCursor) bypassed tracing entirely — those queries were invisible at
    every level (finding 5).  The wrapped conn.cursor must dynamically subclass
    the requested factory with the traced mixin, preserving the row format.
    """
    from frontrun._sql_cursor import _wrap_connection_cursor

    class RealDictCursor:
        """Stand-in for a user-supplied cursor_factory (preserves row format)."""

        def __init__(self, factory: type) -> None:
            self.factory = factory

    captured: dict[str, Any] = {}

    class FakeConn:
        def cursor(self, cursor_factory: type | None = None) -> Any:
            captured["factory"] = cursor_factory
            # Mimic the driver building a cursor instance from the factory.
            return cursor_factory(cursor_factory) if cursor_factory else None

    conn = FakeConn()
    _wrap_connection_cursor(conn, paramstyle="pyformat")
    conn.cursor(cursor_factory=RealDictCursor)

    traced_factory = captured["factory"]
    assert traced_factory is not RealDictCursor, "explicit cursor_factory was not wrapped"
    assert issubclass(traced_factory, RealDictCursor), "traced factory must subclass the user's factory"
    assert getattr(traced_factory, "_frontrun_traced_cursor", False), "wrapped factory must be marked traced"
    # The traced subclass intercepts execute (the whole point of tracing).
    assert "execute" in traced_factory.__dict__


def test_wrap_connection_cursor_idempotent() -> None:
    """Double-wrapping conn.cursor must not double-wrap the factory."""
    from frontrun._sql_cursor import _wrap_connection_cursor

    class BaseFactory:
        def __init__(self, *a: Any) -> None: ...

    captured: dict[str, Any] = {}

    class FakeConn:
        def cursor(self, cursor_factory: type | None = None) -> Any:
            captured["factory"] = cursor_factory
            return None

    conn = FakeConn()
    _wrap_connection_cursor(conn, paramstyle="pyformat")
    _wrap_connection_cursor(conn, paramstyle="pyformat")  # second call is a no-op
    conn.cursor(cursor_factory=BaseFactory)
    traced = captured["factory"]
    # Wrapping a factory twice should not subclass a traced subclass again.
    assert issubclass(traced, BaseFactory)
    assert sum(1 for c in traced.__mro__ if getattr(c, "_frontrun_traced_cursor", False)) == 1


def test_for_update_primary_colset_emits_read_bridge() -> None:
    """SELECT ... FOR UPDATE on the primary colset must emit a READ bridge.

    A primary-colset FOR UPDATE must conflict with a non-primary-colset access
    to the same physical row.  Before the fix, lock elevation flipped the kind
    to "write" before the bridge logic, so the FOR UPDATE emitted neither a
    READ nor WRITE bridge and could not conflict (finding 6).  This mirrors the
    plain-UPDATE read-phase behaviour: primary colset → READ bridge.
    """
    from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id
    from frontrun._sql_cursor import _report_sql_access, clear_sql_metadata

    log = IOLog()
    set_io_reporter(log)
    clear_sql_metadata()  # fresh primary-colset registry
    set_dpor_scheduler(object())  # any non-None scheduler enables the bridge
    set_dpor_thread_id(0)
    _io_tls._in_transaction = False
    try:
        # First access establishes "id" as the primary colset for users.
        _report_sql_access("SELECT * FROM users WHERE id = 1 FOR UPDATE")
        bridge_kinds = {k for r, k in log.events if r == "sql:users"}
        assert "read" in bridge_kinds, (
            "FOR UPDATE on the primary colset must emit a READ bridge resource "
            f"(got events: {log.events})"
        )

        # A non-primary-colset access emits a WRITE bridge; READ vs WRITE on
        # the same sql:users resource is a conflict, as required.
        log.clear()
        _report_sql_access("SELECT * FROM users WHERE username = 'alice'")
        assert ("sql:users", "write") in log.events, (
            "non-primary colset must emit a WRITE bridge so it conflicts with "
            f"the FOR UPDATE READ bridge (got: {log.events})"
        )
    finally:
        set_dpor_scheduler(None)
        set_dpor_thread_id(None)
        clear_sql_metadata()


def test_connection_commit_flushes_buffer() -> None:
    """conn.commit() must drive the tx state machine: flush buffer, end tx.

    With ``cur.execute("BEGIN"); ...; conn.commit()`` the textual COMMIT never
    reaches _handle_tx_op, so without intercepting conn.commit() the buffered
    accesses are silently discarded and _in_transaction stays True (finding 3).
    """
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur = conn.cursor()
    cur.execute("BEGIN")
    log.clear()
    cur.execute("INSERT INTO users VALUES (5, 'Eve')")
    # Inside the transaction, accesses are buffered (not yet reported).
    assert ("sql:users", "write") not in log.events
    conn.commit()

    # After commit, the buffered write must be flushed and tx state cleared.
    assert _io_tls._in_transaction is False
    assert any(k == "write" for r, k in log.events if r.startswith("sql:users"))
    conn.close()


def test_connection_rollback_clears_state() -> None:
    """conn.rollback() must end the transaction and discard the buffer."""
    log = IOLog()
    set_io_reporter(log)
    patch_sql()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur = conn.cursor()
    cur.execute("BEGIN")
    log.clear()
    cur.execute("INSERT INTO users VALUES (6, 'Frank')")
    conn.rollback()

    assert _io_tls._in_transaction is False
    # Rolled-back writes must NOT be flushed.
    assert not any(k == "write" for r, k in log.events if r.startswith("sql:users"))
    conn.close()


# ---------------------------------------------------------------------------
# Finding 4: pymysql autobegin detection (callable autocommit)
# ---------------------------------------------------------------------------


def test_detect_autobegin_callable_autocommit_method() -> None:
    """pymysql exposes autocommit as a *method*; the flag is get_autocommit().

    A truthy bound method must not be read as 'autocommit on'.  With autocommit
    OFF, _detect_autobegin must set _in_transaction (finding 4).
    """
    from frontrun._sql_cursor import _detect_autobegin

    class FakePyMySQLConn:
        def autocommit(self, value: bool | None = None) -> None:  # method, like pymysql
            pass

        def get_autocommit(self) -> bool:
            return False  # autocommit OFF → implicit transaction active

    class FakeCursor:
        connection = FakePyMySQLConn()

    _io_tls._in_transaction = False
    _io_tls._is_autobegin = False
    try:
        _detect_autobegin(FakeCursor())
        assert _io_tls._in_transaction is True
        assert _io_tls._is_autobegin is True
    finally:
        _io_tls._in_transaction = False
        _io_tls._is_autobegin = False


def test_detect_autobegin_callable_autocommit_on() -> None:
    """When pymysql autocommit is ON (get_autocommit()=True), no autobegin."""
    from frontrun._sql_cursor import _detect_autobegin

    class FakePyMySQLConn:
        def autocommit(self, value: bool | None = None) -> None:
            pass

        def get_autocommit(self) -> bool:
            return True

    class FakeCursor:
        connection = FakePyMySQLConn()

    _io_tls._in_transaction = False
    _io_tls._is_autobegin = False
    try:
        _detect_autobegin(FakeCursor())
        assert _io_tls._in_transaction is False
    finally:
        _io_tls._in_transaction = False
        _io_tls._is_autobegin = False
