"""Tests for indexical INSERT ID tracking."""

from __future__ import annotations

import sqlite3
import threading
from unittest.mock import patch

import pytest

from frontrun._sql_cursor import patch_sql, unpatch_sql
from frontrun._sql_insert_tracker import (
    clear_insert_tracker,
    ensure_no_uncaptured_inserts,
    get_records,
    get_uncaptured_tables,
    record_insert,
    resolve_alias,
)
from frontrun.common import NondeterministicSQLError


class TestRecordInsert:
    """Unit tests for record_insert and alias generation."""

    def setup_method(self) -> None:
        clear_insert_tracker()

    def teardown_method(self) -> None:
        clear_insert_tracker()

    def test_basic_alias(self) -> None:
        """Outside scheduler context, thread_id is None → 'setup' prefix."""
        alias = record_insert("users", 42)
        assert alias == "sql:users:setup_ins0"

    def test_sequential_aliases_same_table(self) -> None:
        alias0 = record_insert("users", 1)
        alias1 = record_insert("users", 2)
        assert alias0 == "sql:users:setup_ins0"
        assert alias1 == "sql:users:setup_ins1"

    def test_different_tables_independent_counters(self) -> None:
        a0 = record_insert("users", 1)
        b0 = record_insert("orders", 10)
        a1 = record_insert("users", 2)
        assert a0 == "sql:users:setup_ins0"
        assert b0 == "sql:orders:setup_ins0"
        assert a1 == "sql:users:setup_ins1"

    def test_with_thread_context(self) -> None:
        """When scheduler context is active, thread_id is used."""
        with patch("frontrun._sql_insert_tracker.get_context", return_value=(None, 0)):
            alias = record_insert("users", 1)
        assert alias == "sql:users:t0_ins0"

        with patch("frontrun._sql_insert_tracker.get_context", return_value=(None, 1)):
            alias = record_insert("users", 1)
        assert alias == "sql:users:t1_ins0"

    def test_uncaptured_insert(self) -> None:
        """lastrowid=None marks the table as uncaptured."""
        record_insert("users", None)
        assert "users" in get_uncaptured_tables()

    def test_captured_insert_not_uncaptured(self) -> None:
        record_insert("users", 42)
        assert "users" not in get_uncaptured_tables()


class TestResolveAlias:
    """Unit tests for resolve_alias lookups."""

    def setup_method(self) -> None:
        clear_insert_tracker()

    def teardown_method(self) -> None:
        clear_insert_tracker()

    def test_resolve_captured_id(self) -> None:
        record_insert("users", 42)
        assert resolve_alias("users", 42) == "sql:users:setup_ins0"
        assert resolve_alias("users", "42") == "sql:users:setup_ins0"

    def test_resolve_alias_is_scoped_by_database_identity(self) -> None:
        record_insert("users", 42, db_scope="db_a")
        record_insert("users", 42, db_scope="db_b")

        assert resolve_alias("users", 42, db_scope="db_a") == "sql:users:db=db_a:setup_ins0"
        assert resolve_alias("users", 42, db_scope="db_b") == "sql:users:db=db_b:setup_ins0"
        assert resolve_alias("users", 42) is None

    def test_resolve_unknown_id(self) -> None:
        record_insert("users", 42)
        assert resolve_alias("users", 99) is None

    def test_resolve_wrong_table(self) -> None:
        record_insert("users", 42)
        assert resolve_alias("orders", 42) is None

    def test_resolve_uncaptured_returns_none(self) -> None:
        record_insert("users", None)
        assert resolve_alias("users", 1) is None


class TestClearAndRecords:
    """Tests for clear_insert_tracker and get_records."""

    def setup_method(self) -> None:
        clear_insert_tracker()

    def teardown_method(self) -> None:
        clear_insert_tracker()

    def test_clear_resets_everything(self) -> None:
        record_insert("users", 1)
        clear_insert_tracker()
        assert get_records() == []
        assert resolve_alias("users", 1) is None
        assert get_uncaptured_tables() == set()

    def test_get_records_returns_copies(self) -> None:
        record_insert("users", 42)
        records = get_records()
        assert len(records) == 1
        assert records[0].table == "users"
        assert records[0].concrete_id == 42
        assert records[0].captured is True

    def test_counters_reset_after_clear(self) -> None:
        record_insert("users", 1)
        clear_insert_tracker()
        alias = record_insert("users", 2)
        assert alias == "sql:users:setup_ins0"  # Counter reset to 0


class TestCheckUncapturedInserts:
    """Tests for the fallback warning."""

    def setup_method(self) -> None:
        clear_insert_tracker()

    def teardown_method(self) -> None:
        clear_insert_tracker()

    def test_no_inserts_no_error(self) -> None:
        ensure_no_uncaptured_inserts()  # Should not raise

    def test_captured_inserts_no_error(self) -> None:
        record_insert("users", 42)
        ensure_no_uncaptured_inserts()  # Should not raise

    def test_uncaptured_inserts_raises(self) -> None:
        record_insert("users", None)
        with pytest.raises(NondeterministicSQLError, match="users"):
            ensure_no_uncaptured_inserts()


class TestThreadSafety:
    """Concurrent access to the tracker."""

    def setup_method(self) -> None:
        clear_insert_tracker()

    def teardown_method(self) -> None:
        clear_insert_tracker()

    def test_concurrent_inserts(self) -> None:
        barrier = threading.Barrier(4)
        _tls = threading.local()

        def mock_get_context() -> tuple[None, int] | None:
            tid = getattr(_tls, "tid", None)
            if tid is not None:
                return (None, tid)
            return None

        def insert_from_thread(tid: int) -> None:
            _tls.tid = tid
            barrier.wait()
            record_insert("users", tid * 100 + 1)
            record_insert("users", tid * 100 + 2)

        with patch("frontrun._sql_insert_tracker.get_context", side_effect=mock_get_context):
            threads = [threading.Thread(target=insert_from_thread, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        records = get_records()
        assert len(records) == 8
        # Each thread should have seq 0 and 1 for "users"
        aliases = {r.logical_alias for r in records}
        for tid in range(4):
            assert f"sql:users:t{tid}_ins0" in aliases
            assert f"sql:users:t{tid}_ins1" in aliases


class TestInsertCaptureViaCursor:
    """Integration test: INSERT through a patched SQLite cursor captures lastrowid."""

    def setup_method(self) -> None:
        clear_insert_tracker()
        patch_sql()

    def teardown_method(self) -> None:
        unpatch_sql()
        clear_insert_tracker()

    def test_insert_captures_lastrowid(self) -> None:
        from frontrun._io_detection import set_io_reporter

        reported: list[tuple[str, str]] = []

        def reporter(res_id: str, kind: str) -> None:
            reported.append((res_id, kind))

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (name) VALUES ('alice')")
            conn.close()
        finally:
            set_io_reporter(None)

        records = get_records()
        assert len(records) == 1
        assert records[0].table == "users"
        assert records[0].concrete_id == 1  # SQLite autoincrement starts at 1
        assert records[0].captured is True
        assert records[0].logical_alias.startswith("sql:users:")
        assert records[0].logical_alias.endswith("setup_ins0")

        # Should have reported the alias and sequence resource
        alias_writes = [(r, k) for r, k in reported if r.startswith("sql:users:") and "setup_ins0" in r]
        seq_writes = [(r, k) for r, k in reported if r.startswith("sql:users:") and r.endswith(":seq")]
        assert len(alias_writes) >= 1
        assert len(seq_writes) >= 1

    def test_downstream_select_resolves_alias(self) -> None:
        """A SELECT using the autoincrement ID resolves to the logical alias."""
        from frontrun._io_detection import set_io_reporter

        reported: list[tuple[str, str]] = []

        def reporter(res_id: str, kind: str) -> None:
            reported.append((res_id, kind))

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (name) VALUES ('alice')")
            # Now SELECT using the autoincrement ID (1)
            conn.execute("SELECT * FROM users WHERE id = 1")
            conn.close()
        finally:
            set_io_reporter(None)

        # The SELECT should have resolved id=1 to the alias
        select_reports = [
            (r, k) for r, k in reported if r.startswith("sql:users:") and "setup_ins0" in r and k == "read"
        ]
        assert len(select_reports) >= 1

    def test_select_not_tracked_as_insert(self) -> None:
        """A SELECT does not create INSERT records."""
        from frontrun._io_detection import set_io_reporter

        def reporter(res_id: str, kind: str) -> None:
            pass

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            clear_insert_tracker()
            conn.execute("SELECT * FROM users")
            conn.close()
        finally:
            set_io_reporter(None)

        assert get_records() == []

    def test_update_not_tracked_as_insert(self) -> None:
        """An UPDATE does not create INSERT records."""
        from frontrun._io_detection import set_io_reporter

        def reporter(res_id: str, kind: str) -> None:
            pass

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            clear_insert_tracker()
            conn.execute("UPDATE users SET name = 'bob' WHERE id = 1")
            conn.close()
        finally:
            set_io_reporter(None)

        assert get_records() == []


class TestNondeterministicSQLError:
    """Test the exception itself."""

    def test_exception_is_importable(self) -> None:
        from frontrun.common import NondeterministicSQLError as E

        assert issubclass(E, Exception)

    def test_exception_message(self) -> None:
        err = NondeterministicSQLError("lastrowid capture failed on: users")
        assert "users" in str(err)
