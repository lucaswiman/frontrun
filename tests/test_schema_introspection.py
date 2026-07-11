"""Tests for automatic FK schema introspection."""

from __future__ import annotations

import sqlite3

import pytest

from frontrun._schema import (
    ForeignKey,
    Schema,
    _detect_driver,
    _introspect_sqlite,
    get_schema,
    introspect_fks,
    register_schema,
)


@pytest.fixture(autouse=True)
def _clean_schema():
    """Reset global schema between tests."""
    yield
    register_schema(Schema())


# ---------------------------------------------------------------------------
# Driver detection
# ---------------------------------------------------------------------------


class TestDetectDriver:
    def test_sqlite3_connection(self):
        conn = sqlite3.connect(":memory:")
        try:
            assert _detect_driver(conn) == "sqlite"
        finally:
            conn.close()

    def test_unknown_driver_raises(self):
        with pytest.raises(ValueError, match="Cannot detect database driver"):
            _detect_driver(object())

    def test_psycopg2_connection_detected(self):
        """Verify detection logic for psycopg2 module prefix."""

        class FakeConn:
            pass

        FakeConn.__module__ = "psycopg2.extensions"
        FakeConn.__qualname__ = "connection"
        assert _detect_driver(FakeConn()) == "postgresql"

    def test_psycopg3_connection_detected(self):
        class FakeConn:
            pass

        FakeConn.__module__ = "psycopg.connection"
        FakeConn.__qualname__ = "Connection"
        assert _detect_driver(FakeConn()) == "postgresql"

    def test_traced_subclass_uses_driver_base_module(self):
        class DriverConnection:
            pass

        DriverConnection.__module__ = "psycopg2.extensions"

        class TracedConnection(DriverConnection):
            pass

        TracedConnection.__module__ = "frontrun._sql_cursor"
        assert _detect_driver(TracedConnection()) == "postgresql"

    def test_pymysql_connection_detected(self):
        class FakeConn:
            pass

        FakeConn.__module__ = "pymysql.connections"
        FakeConn.__qualname__ = "Connection"
        assert _detect_driver(FakeConn()) == "mysql"

    def test_asyncpg_connection_detected(self):
        class FakeConn:
            pass

        FakeConn.__module__ = "asyncpg.connection"
        FakeConn.__qualname__ = "Connection"
        assert _detect_driver(FakeConn()) == "postgresql"

    def test_mysqldb_connection_detected(self):
        class FakeConn:
            pass

        FakeConn.__module__ = "MySQLdb.connections"
        FakeConn.__qualname__ = "Connection"
        assert _detect_driver(FakeConn()) == "mysql"


# ---------------------------------------------------------------------------
# SQLite introspection (uses real sqlite3)
# ---------------------------------------------------------------------------


class TestSQLiteIntrospection:
    def _make_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def test_no_tables(self):
        conn = self._make_db()
        try:
            schema = _introspect_sqlite(conn)
            assert schema.get_fks("anything") == []
        finally:
            conn.close()

    def test_single_fk(self):
        conn = self._make_db()
        try:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))")
            schema = _introspect_sqlite(conn)
            fks = schema.get_fks("orders")
            assert len(fks) == 1
            assert fks[0] == ForeignKey(table="orders", column="user_id", ref_table="users", ref_column="id")
        finally:
            conn.close()

    def test_multiple_fks_same_table(self):
        conn = self._make_db()
        try:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY)")
            conn.execute(
                "CREATE TABLE reviews ("
                "  id INTEGER PRIMARY KEY,"
                "  user_id INTEGER REFERENCES users(id),"
                "  product_id INTEGER REFERENCES products(id))"
            )
            schema = _introspect_sqlite(conn)
            fks = schema.get_fks("reviews")
            assert len(fks) == 2
            ref_tables = {fk.ref_table for fk in fks}
            assert ref_tables == {"users", "products"}
        finally:
            conn.close()

    def test_chain_fks(self):
        """orders -> users -> departments."""
        conn = self._make_db()
        try:
            conn.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, dept_id INTEGER REFERENCES departments(id))")
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))")
            schema = _introspect_sqlite(conn)

            assert schema.get_referenced_tables("orders") == {"users"}
            assert schema.get_referenced_tables("users") == {"departments"}
            assert schema.get_referenced_tables("departments") == set()
        finally:
            conn.close()

    def test_self_referential_fk(self):
        conn = self._make_db()
        try:
            conn.execute(
                "CREATE TABLE categories (  id INTEGER PRIMARY KEY,  parent_id INTEGER REFERENCES categories(id))"
            )
            schema = _introspect_sqlite(conn)
            fks = schema.get_fks("categories")
            assert len(fks) == 1
            assert fks[0].ref_table == "categories"
            assert fks[0].column == "parent_id"
            assert fks[0].ref_column == "id"
        finally:
            conn.close()

    def test_no_sqlite_internal_tables(self):
        """sqlite_* tables should be skipped."""
        conn = self._make_db()
        try:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            schema = _introspect_sqlite(conn)
            # Should not error or include sqlite_master etc.
            assert schema.get_fks("sqlite_master") == []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# introspect_fks() integration (SQLite end-to-end)
# ---------------------------------------------------------------------------


class TestIntrospectFks:
    def test_sqlite_registers_globally(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))")

            schema = introspect_fks(conn)
            assert schema is get_schema()
            assert len(schema.get_fks("orders")) == 1
        finally:
            conn.close()

    def test_register_false(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))")

            old_schema = get_schema()
            schema = introspect_fks(conn, register=False)
            assert schema is not get_schema()
            assert get_schema() is old_schema
        finally:
            conn.close()

    def test_unknown_driver(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            introspect_fks(object())

    def test_empty_database(self):
        conn = sqlite3.connect(":memory:")
        try:
            schema = introspect_fks(conn)
            assert schema.get_fks("anything") == []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# PostgreSQL / MySQL introspection (mocked, no live server needed)
# ---------------------------------------------------------------------------


class TestPostgreSQLIntrospection:
    """Test PG introspection using a mock cursor that returns canned rows."""

    def test_postgresql_fks(self):
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_conn.__class__.__module__ = "psycopg2.extensions"
        mock_conn.__class__.__qualname__ = "connection"

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("orders", "user_id", "users", "id"),
            ("shipments", "order_id", "orders", "id"),
        ]

        schema = introspect_fks(mock_conn, register=False)

        assert len(schema.get_fks("orders")) == 1
        assert schema.get_fks("orders")[0] == ForeignKey("orders", "user_id", "users", "id")
        assert len(schema.get_fks("shipments")) == 1
        assert schema.get_fks("shipments")[0] == ForeignKey("shipments", "order_id", "orders", "id")
        mock_cursor.close.assert_called_once()

    def test_pg_schema_parameter(self):
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_conn.__class__.__module__ = "psycopg2.extensions"
        mock_conn.__class__.__qualname__ = "connection"

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        introspect_fks(mock_conn, pg_schema="myschema", register=False)
        # Verify the schema parameter was passed
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == ("myschema",)


class TestMySQLIntrospection:
    def test_mysql_fks(self):
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_conn.__class__.__module__ = "pymysql.connections"
        mock_conn.__class__.__qualname__ = "Connection"

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("orders", "user_id", "users", "id"),
        ]

        schema = introspect_fks(mock_conn, register=False)

        assert len(schema.get_fks("orders")) == 1
        assert schema.get_fks("orders")[0] == ForeignKey("orders", "user_id", "users", "id")
        mock_cursor.close.assert_called_once()
