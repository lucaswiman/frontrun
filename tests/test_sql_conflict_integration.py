"""Integration tests for SQL-level conflict detection with real databases.

These tests verify that the SQL cursor patching correctly detects table-level
(and row-level) conflicts when used with real database drivers.

Uses sqlite3 (in-memory) — no external dependencies required.

The tests operate at the _intercept_execute level by:
1. Setting up sqlite3 with patch_sql()
2. Creating threads that execute SQL against real in-memory databases
3. Checking that the I/O reporter receives the correct resource IDs

This verifies that the SQL conflict detection correctly identifies:
- Threads touching different tables as independent (no conflict)
- Threads writing to the same table as conflicting
- Threads reading the same table as independent (reads don't conflict)
- A thread reading and another writing the same table as conflicting
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

import frontrun
from frontrun._io_detection import set_io_reporter
from frontrun._sql_cursor import (
    _ORIGINAL_METHODS,
    _PATCHES,
    _sql_resource_id,
    _sql_sequence_resource_id,
    _suppress_tids,
    patch_sql,
    unpatch_sql,
)
from tests.io_test_helpers import IOLog
from tests.sql_test_helpers import execute_with_retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSqlConflictIsolation:
    """Verify that SQL detection correctly isolates threads touching different tables."""

    def test_same_table_name_in_different_databases_reports_distinct_resources(self) -> None:
        """Independent databases should not collapse to the same sql:<table> resource."""
        patch_sql()

        db_uri_a = "file:memdb_identity_a?mode=memory&cache=shared"
        db_uri_b = "file:memdb_identity_b?mode=memory&cache=shared"

        conn_a = sqlite3.connect(db_uri_a, timeout=30, uri=True, check_same_thread=False)
        conn_b = sqlite3.connect(db_uri_b, timeout=30, uri=True, check_same_thread=False)

        execute_with_retry(conn_a, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        execute_with_retry(conn_b, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        execute_with_retry(conn_a, "INSERT INTO users VALUES (1, 'alice')")
        execute_with_retry(conn_b, "INSERT INTO users VALUES (1, 'bob')")
        conn_a.commit()
        conn_b.commit()

        log_a = IOLog()
        log_b = IOLog()

        set_io_reporter(log_a)
        execute_with_retry(conn_a, "SELECT * FROM users")

        set_io_reporter(log_b)
        execute_with_retry(conn_b, "SELECT * FROM users")

        res_a = {r for r, _ in log_a.events}
        res_b = {r for r, _ in log_b.events}

        assert res_a
        assert res_b
        assert res_a.isdisjoint(res_b), (res_a, res_b)

        conn_a.close()
        conn_b.close()

    def test_different_tables_independent_report_distinct_resources(self) -> None:
        """Two threads touching different tables → distinct resource IDs."""
        patch_sql()

        # Each thread uses its own connection to avoid cross-talk, but they share
        # the same in-memory DB via URI.
        db_uri = "file:memdb_diff?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE table_a (id INTEGER PRIMARY KEY, val TEXT)")
        execute_with_retry(conn, "CREATE TABLE table_b (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "INSERT INTO table_a VALUES (1, 'from thread a')")
            with lock:
                results["a"].extend(thread_log.events)
            c.close()

        def thread_b() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "INSERT INTO table_b VALUES (1, 'from thread b')")
            with lock:
                results["b"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both reported something
        assert len(results["a"]) >= 1
        assert len(results["b"]) >= 1

        # Resource IDs are different
        res_a = {r for r, _ in results["a"]}
        res_b = {r for r, _ in results["b"]}

        assert any(r == "sql:table_a" or r.startswith("sql:table_a:") for r in res_a)
        assert any(r == "sql:table_b" or r.startswith("sql:table_b:") for r in res_b)

        # Parsed statements also share the conservative database-intent
        # resource. Their concrete table resources must remain independent.
        assert any(r.startswith("sql:__database__") for r in res_a)
        assert any(r.startswith("sql:__database__") for r in res_b)
        tables_a = {
            r.split(":", 1)[1].split(":")[0]
            for r in res_a
            if r.startswith("sql:") and not r.startswith("sql:__database__")
        }
        tables_b = {
            r.split(":", 1)[1].split(":")[0]
            for r in res_b
            if r.startswith("sql:") and not r.startswith("sql:__database__")
        }
        assert not (tables_a & tables_b)

    def test_multi_thread_independent_accesses(self) -> None:
        """Many threads touching many tables → all independent."""
        patch_sql()

        db_uri = "file:memdb_multi?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        for i in range(10):
            execute_with_retry(conn, f"CREATE TABLE table_{i} (id INTEGER PRIMARY KEY)")
        conn.commit()

        all_events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def thread_fn(i: int) -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, f"INSERT INTO table_{i} VALUES (1)")
            with lock:
                all_events.extend(thread_log.events)
            c.close()

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 distinct tables reported
        assert any(r.startswith("sql:__database__") and kind == "read" for r, kind in all_events)
        tables = {
            r.split(":", 1)[1].split(":")[0]
            for r, _ in all_events
            if r.startswith("sql:") and not r.startswith("sql:__database__")
        }
        assert len(tables) == 10
        assert all(f"table_{i}" in tables for i in range(10))


class TestSameTableWriteWriteConflict:
    """Verify that multiple writes to the same table produce the same resource ID."""

    def test_two_threads_insert_same_table_report_same_resource(self) -> None:
        """Both threads INSERT to the same table → same resource ID sql:items."""
        patch_sql()

        db_uri = "file:memdb_insert?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "INSERT INTO items VALUES (1, 'item 1')")
            with lock:
                results["a"].extend(thread_log.events)
            c.close()

        def thread_b() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "INSERT INTO items VALUES (2, 'item 2')")
            with lock:
                results["b"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        writes_a = [
            (r, k) for r, k in results["a"] if (r == "sql:items" or r.startswith("sql:items:")) and k == "write"
        ]
        writes_b = [
            (r, k) for r, k in results["b"] if (r == "sql:items" or r.startswith("sql:items:")) and k == "write"
        ]

        assert len(writes_a) >= 1, "Thread A should report a write to sql:items"
        assert len(writes_b) >= 1, "Thread B should report a write to sql:items"

    def test_two_threads_update_same_table_report_shared_resource(self) -> None:
        """Both threads UPDATE the same table → shared resource ID sql:counters."""
        patch_sql()

        # Use shared memory URI to allow multiple connections to same DB
        db_uri = "file:memdb_ww?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE counters (id INTEGER PRIMARY KEY, val INTEGER)")
        execute_with_retry(conn, "INSERT INTO counters VALUES (1, 0)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "UPDATE counters SET val = val + 1 WHERE id = 1")
            with lock:
                results["a"].extend(thread_log.events)
            c.close()

        def thread_b() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "UPDATE counters SET val = val + 10 WHERE id = 1")
            with lock:
                results["b"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        resources_a = {r for r, _ in results["a"]}
        resources_b = {r for r, _ in results["b"]}

        # Both threads touched the same table → same resource IDs
        assert any(r == "sql:counters" or r.startswith("sql:counters:") for r in resources_a), (
            f"Thread A resources: {resources_a}"
        )
        assert any(r == "sql:counters" or r.startswith("sql:counters:") for r in resources_b), (
            f"Thread B resources: {resources_b}"
        )

        # UPDATE reports both read (WHERE clause) and write (SET clause)
        assert any(k == "write" for r, k in results["a"] if r == "sql:counters" or r.startswith("sql:counters:")), (
            "Thread A UPDATE should report a write to sql:counters"
        )
        assert any(k == "write" for r, k in results["b"] if r == "sql:counters" or r.startswith("sql:counters:")), (
            "Thread B UPDATE should report a write to sql:counters"
        )

    def test_write_write_conflict_detected_same_resource_id(self) -> None:
        """Two writes to the same table produce identical resource IDs — a conflict."""
        patch_sql()

        db_uri = "file:memdb_ww_conflict?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE events (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()

        all_writes: list[str] = []
        lock = threading.Lock()

        def thread_fn(row_id: int) -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                execute_with_retry(c, "INSERT INTO events VALUES (?, 'payload')", (row_id,))
            finally:
                # Collect events even if sqlite3 raises (e.g. OperationalError
                # from concurrent access) — the write is reported before the
                # actual SQL executes.
                with lock:
                    all_writes.extend(r for r, k in thread_log.events if k == "write")
            c.close()

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(1, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes are to the same resource — conflicting
        assert all(r == "sql:events" or r.startswith("sql:events:") for r in all_writes), (
            f"Expected all writes to be sql:events, got: {set(all_writes)}"
        )

    def test_parameterized_writes_report_same_resource(self) -> None:
        """Parameterized INSERTs to same table report same resource ID."""
        patch_sql()

        db_uri = "file:memdb_param_write?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE stats (key TEXT PRIMARY KEY, val INTEGER)")
        conn.commit()

        all_writes: list[str] = []
        lock = threading.Lock()

        def thread_fn(key: str) -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "INSERT INTO stats VALUES (?, 1)", (key,))
            with lock:
                all_writes.extend(r for r, k in thread_log.events if k == "write")
            c.close()

        threads = [threading.Thread(target=thread_fn, args=(f"k{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == "sql:stats" or r.startswith("sql:stats:") for r in all_writes), (
            f"Expected all writes to be sql:stats, got: {set(all_writes)}"
        )

    def test_delete_same_table_both_threads_report_write(self) -> None:
        """DELETE on same table → both threads report write on shared resource."""
        patch_sql()

        db_uri = "file:memdb_delete?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE logs (id INTEGER PRIMARY KEY, msg TEXT)")
        execute_with_retry(conn, "INSERT INTO logs VALUES (1, 'msg1')")
        execute_with_retry(conn, "INSERT INTO logs VALUES (2, 'msg2')")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "DELETE FROM logs WHERE id = 1")
            with lock:
                results["a"].extend(thread_log.events)
            c.close()

        def thread_b() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "DELETE FROM logs WHERE id = 2")
            with lock:
                results["b"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        writes_a = [r for r, k in results["a"] if k == "write" and (r == "sql:logs" or r.startswith("sql:logs:"))]
        writes_b = [r for r, k in results["b"] if k == "write" and (r == "sql:logs" or r.startswith("sql:logs:"))]

        assert len(writes_a) >= 1, "Thread A DELETE should report write to sql:logs"
        assert len(writes_b) >= 1, "Thread B DELETE should report write to sql:logs"


class TestSameTableReadReadIndependent:
    """Verify that multiple reads (SELECT) to the same table are independent."""

    def test_two_selects_same_table_both_report_read(self) -> None:
        """Two SELECTs on the same table → both report read, independent."""
        patch_sql()

        db_uri = "file:memdb_rr?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE products (id INTEGER PRIMARY KEY, price REAL)")
        execute_with_retry(conn, "INSERT INTO products VALUES (1, 19.99)")
        execute_with_retry(conn, "INSERT INTO products VALUES (2, 29.99)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "SELECT * FROM products WHERE id = 1")
            with lock:
                results["a"].extend(thread_log.events)
            c.close()

        def thread_b() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "SELECT * FROM products WHERE id = 2")
            with lock:
                results["b"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        reads_a = [
            (r, k) for r, k in results["a"] if (r == "sql:products" or r.startswith("sql:products:")) and k == "read"
        ]
        reads_b = [
            (r, k) for r, k in results["b"] if (r == "sql:products" or r.startswith("sql:products:")) and k == "read"
        ]

        assert len(reads_a) >= 1, f"Thread A should report a read from sql:products, got {results['a']}"
        assert len(reads_b) >= 1, f"Thread B should report a read from sql:products, got {results['b']}"

    def test_multiple_threads_select_same_table_all_report_read(self) -> None:
        """5 threads SELECTing the same table → all report read, all independent."""
        patch_sql()

        db_uri = "file:memdb_rr_multi?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE stats (id INTEGER PRIMARY KEY, count INTEGER)")
        execute_with_retry(conn, "INSERT INTO stats VALUES (1, 100)")
        conn.commit()

        all_events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def thread_fn() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "SELECT count FROM stats")
            with lock:
                all_events.extend(thread_log.events)
            c.close()

        threads = [threading.Thread(target=thread_fn) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats_events = [(r, k) for r, k in all_events if (r == "sql:stats" or r.startswith("sql:stats:"))]
        # Each thread reports 2 reads: table-level + :seq for phantom detection
        assert len(stats_events) == 10, f"Expected 10 read events (2 per thread), got {len(stats_events)}"
        assert all(k == "read" for _, k in stats_events)


class TestReadWriteConflict:
    """Verify that a read (SELECT) and a write (INSERT/UPDATE/DELETE) conflict."""

    def test_read_and_write_same_table_both_report_resource(self) -> None:
        """One thread SELECTs, another UPDATES same table → conflict."""
        patch_sql()

        db_uri = "file:memdb_rw?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE balances (id INTEGER PRIMARY KEY, amount REAL)")
        execute_with_retry(conn, "INSERT INTO balances VALUES (1, 1000.0)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"reader": [], "writer": []}
        lock = threading.Lock()

        def reader() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "SELECT amount FROM balances WHERE id = 1")
            with lock:
                results["reader"].extend(thread_log.events)
            c.close()

        def writer() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            execute_with_retry(c, "UPDATE balances SET amount = amount + 100 WHERE id = 1")
            with lock:
                results["writer"].extend(thread_log.events)
            c.close()

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        reader_reads = [
            (r, k)
            for r, k in results["reader"]
            if (r == "sql:balances" or r.startswith("sql:balances:")) and k == "read"
        ]
        assert len(reader_reads) >= 1, f"Reader should report read from sql:balances, got {results['reader']}"

        writer_writes = [
            (r, k)
            for r, k in results["writer"]
            if (r == "sql:balances" or r.startswith("sql:balances:")) and k == "write"
        ]
        assert len(writer_writes) >= 1, f"Writer should report write to sql:balances, got {results['writer']}"

    def test_read_write_conflict_shared_resource_id(self) -> None:
        """Verify that reader and writer produce the same base resource ID (sql:table)."""
        patch_sql()

        db_uri = "file:memdb_rw_shared?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE queue (id INTEGER PRIMARY KEY, msg TEXT)")
        conn.commit()

        all_events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def thread_fn(is_writer: bool) -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            if is_writer:
                execute_with_retry(c, "INSERT INTO queue VALUES (1, 'msg')")
            else:
                execute_with_retry(c, "SELECT * FROM queue")
            with lock:
                all_events.extend(thread_log.events)
            c.close()

        threads = [
            threading.Thread(target=thread_fn, args=(False,)),
            threading.Thread(target=thread_fn, args=(True,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        queue_events = [(r, k) for r, k in all_events if (r == "sql:queue" or r.startswith("sql:queue:"))]
        resource_ids = {r for r, k in queue_events}
        assert any(r == "sql:queue" or r.startswith("sql:queue:") for r in resource_ids), (
            f"Expected sql:queue in resources, got {resource_ids}"
        )


class TestSqlMultiStatementHandling:
    """Verify that multiple statements on the same connection are correctly traced."""

    def test_two_queries_on_same_connection(self) -> None:
        """Two statements on the same cursor → both reported correctly."""
        patch_sql()

        db_uri = "file:memdb_multi_stmt?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"select": [], "insert": []}
        lock = threading.Lock()

        def thread_fn() -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)

            # 1. SELECT
            execute_with_retry(c, "SELECT * FROM messages")
            with lock:
                results["select"].extend(thread_log.events)
            thread_log.clear()

            # 2. INSERT
            execute_with_retry(c, "INSERT INTO messages VALUES (1, 'hello')")
            with lock:
                results["insert"].extend(thread_log.events)

            c.close()

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()

        select_reads = [
            k for r, k in results["select"] if (r == "sql:messages" or r.startswith("sql:messages:")) and k == "read"
        ]
        assert len(select_reads) >= 1

        insert_writes = [
            k for r, k in results["insert"] if (r == "sql:messages" or r.startswith("sql:messages:")) and k == "write"
        ]
        assert len(insert_writes) >= 1


class TestSqlPreloadIntegration:
    """Verify that SQL-level reporting correctly suppresses endpoint-level LD_PRELOAD reporting."""

    def test_insert_while_select_running_reports_conflict_resources(self) -> None:
        """Concurrent INSERT and SELECT on same table → conflicting resource IDs."""
        patch_sql()

        db_uri = "file:memdb_preload?mode=memory&cache=shared"
        conn = sqlite3.connect(db_uri, timeout=30, uri=True, check_same_thread=False)
        execute_with_retry(conn, "CREATE TABLE data (id INTEGER PRIMARY KEY)")
        conn.commit()

        all_resources: set[str] = set()
        lock = threading.Lock()

        def thread_fn(is_writer: bool) -> None:
            c = sqlite3.connect(db_uri, timeout=30, uri=True)
            thread_log = IOLog()
            set_io_reporter(thread_log)
            if is_writer:
                execute_with_retry(c, "INSERT INTO data VALUES (1)")
            else:
                execute_with_retry(c, "SELECT * FROM data")
            with lock:
                for r, _ in thread_log.events:
                    all_resources.add(r)
            c.close()

        threads = [
            threading.Thread(target=thread_fn, args=(False,)),
            threading.Thread(target=thread_fn, args=(True,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only see sql-level resources, not socket-level ones (because of suppression)
        assert any(r == "sql:data" or r.startswith("sql:data:") for r in all_resources)
        # Verify no socket-level resources are present
        assert not any(r.startswith("socket:") for r in all_resources)


class TestTableNameCaseFolding:
    """Unquoted table identifiers are case-insensitive under PostgreSQL and
    SQLite (and MySQL with ``lower_case_table_names``), so two spellings of one
    table denote the same physical rows.  The conflict-graph identity must
    therefore case-fold table names: keeping ``Accounts`` and ``accounts``
    distinct is an *under-merge* that hides a genuine same-row race, and
    under-merging is forbidden (CLAUDE.md soundness principles).
    """

    def test_resource_ids_fold_table_case(self) -> None:
        assert _sql_resource_id("accounts", []) == _sql_resource_id("Accounts", [])
        assert _sql_resource_id("accounts", []) == _sql_resource_id("ACCOUNTS", [])
        assert _sql_sequence_resource_id("Orders") == _sql_sequence_resource_id("orders")

    def test_primary_colset_folds_table_case(self) -> None:
        # The first spelling seen establishes the primary column set for the
        # table; a differently-cased spelling must resolve to that SAME entry,
        # not create a fresh (and independently-tracked) one.
        from frontrun._sql_db_scope import _get_primary_colset, _table_primary_colset

        _table_primary_colset.clear()
        try:
            first = _get_primary_colset("Widget", ("id",))
            second = _get_primary_colset("widget", ("name",))
            assert first == second == ("id",)
        finally:
            _table_primary_colset.clear()

    def test_insert_alias_folds_table_case(self) -> None:
        # A captured INSERT id recorded under one spelling must resolve when a
        # later statement references the row via a differently-cased name.
        from frontrun._sql_insert_tracker import clear_insert_tracker, record_insert, resolve_alias

        clear_insert_tracker()
        try:
            alias = record_insert("Accounts", 42)
            assert resolve_alias("accounts", 42) == alias
        finally:
            clear_insert_tracker()

    def test_case_variant_lost_update_is_found(self, tmp_path: Path) -> None:
        """DPOR must find a lost update between workers spelling the table
        differently.  Distinct resource IDs make the two writes look
        independent, so the race is pruned and ``explore()`` certifies a
        program that is genuinely broken — a false pass.
        """
        db = str(tmp_path / "accounts.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, bal INTEGER)")
        conn.execute("INSERT INTO accounts VALUES (1, 100)")
        conn.commit()
        conn.close()

        def make_worker(table: str) -> Callable[[str], None]:
            def worker(path: str) -> None:
                c = sqlite3.connect(path, timeout=5)
                bal = c.execute(f"SELECT bal FROM {table} WHERE id = 1").fetchone()[0]
                c.execute(f"UPDATE {table} SET bal = ? WHERE id = 1", (bal + 10,))
                c.commit()
                c.close()

            return worker

        def invariant(path: str) -> bool:
            c = sqlite3.connect(path)
            bal = c.execute("SELECT bal FROM accounts WHERE id = 1").fetchone()[0]
            c.close()
            return bal == 120

        result = frontrun.explore(
            setup=lambda: db,
            workers=[make_worker("accounts"), make_worker("Accounts")],
            invariant=invariant,
            detect_io=True,
            max_executions=200,
        )
        assert not result.property_holds, "DPOR missed the lost update across case-variant table names"
