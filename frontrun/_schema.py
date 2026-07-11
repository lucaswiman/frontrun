"""Schema information for SQL conflict detection.

Provides structures to store and query database schema metadata, primarily
Foreign Key relationships, which are necessary for correct conflict detection
in multi-table workloads (Phase 6).

If a table T1 has a foreign key to T2, any write to T1 implicitly reads T2
(to validate the constraint).  Without this, operations on T1 and T2 might
appear independent when they are not.

Manual registration::

    from frontrun._schema import Schema, ForeignKey, register_schema

    schema = Schema()
    schema.add_foreign_key(ForeignKey("orders", "user_id", "users", "id"))
    register_schema(schema)

Automatic introspection from a live database connection::

    from frontrun._schema import introspect_fks

    schema = introspect_fks(connection)  # PEP 249 connection
    # Also registers as the global schema automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForeignKey:
    """A foreign key constraint."""

    table: str  # The child table (referencing)
    column: str  # The child column
    ref_table: str  # The parent table (referenced)
    ref_column: str  # The parent column


@dataclass
class Schema:
    """Registry of database schema information."""

    # Map from child_table -> list[ForeignKey]
    _fks: dict[str, list[ForeignKey]] = field(default_factory=dict)

    def add_foreign_key(self, fk: ForeignKey) -> None:
        """Register a foreign key constraint."""
        self._fks.setdefault(fk.table, []).append(fk)

    def get_fks(self, table: str) -> list[ForeignKey]:
        """Get all foreign keys where the given table is the child."""
        return list(self._fks.get(table, []))

    def get_referenced_tables(self, table: str) -> set[str]:
        """Get set of tables referenced by this table via FKs."""
        return {fk.ref_table for fk in self.get_fks(table)}


# Global singleton for the application schema
_global_schema: Schema = Schema()


def register_schema(schema: Schema) -> None:
    """Set the global schema instance."""
    global _global_schema  # noqa: PLW0603
    _global_schema = schema


def get_schema() -> Schema:
    """Get the current global schema."""
    return _global_schema


def _detect_driver(conn: Any) -> str:
    """Detect database driver from a PEP 249 connection object.

    Returns one of: ``"sqlite"``, ``"postgresql"``, ``"mysql"``.

    Raises ``ValueError`` if the driver cannot be identified.
    """
    conn_type = type(conn)
    mod = conn_type.__module__
    cls = conn_type.__qualname__
    modules = tuple(base.__module__ for base in conn_type.__mro__)

    # SQLite: stdlib sqlite3 or aiosqlite
    if any(module.startswith(("sqlite3", "aiosqlite")) for module in modules) or (
        cls.startswith("Connection") and "sqlite" in mod
    ):
        return "sqlite"

    # PostgreSQL: psycopg2, psycopg (v3), asyncpg
    if any(module.startswith(("psycopg2", "psycopg.", "psycopg_", "asyncpg")) for module in modules):
        return "postgresql"

    # MySQL: pymysql, mysqlclient (MySQLdb), aiomysql, mysql.connector
    if any(module.startswith(("pymysql", "MySQLdb", "_mysql", "aiomysql", "mysql")) for module in modules):
        return "mysql"

    raise ValueError(f"Cannot detect database driver for connection type {mod}.{cls}")


def _introspect_sqlite(conn: Any) -> Schema:
    """Introspect FK constraints from a SQLite connection."""
    schema = Schema()
    cur = conn.cursor()
    try:
        # Get all user tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables: list[str] = [row[0] for row in cur.fetchall()]

        for table in tables:
            cur.execute(f"PRAGMA foreign_key_list({table})")  # noqa: S608
            for row in cur.fetchall():
                # PRAGMA foreign_key_list columns: id, seq, table, from, to, ...
                ref_table: str = row[2]
                from_col: str = row[3]
                to_col: str = row[4]
                schema.add_foreign_key(ForeignKey(table=table, column=from_col, ref_table=ref_table, ref_column=to_col))
    finally:
        cur.close()
    return schema


_PG_FK_QUERY = """\
SELECT
    kcu.table_name,
    kcu.column_name,
    ccu.table_name  AS ref_table,
    ccu.column_name AS ref_column
FROM information_schema.key_column_usage kcu
JOIN information_schema.referential_constraints rc
    ON  kcu.constraint_name   = rc.constraint_name
    AND kcu.constraint_schema = rc.constraint_schema
JOIN information_schema.constraint_column_usage ccu
    ON  rc.unique_constraint_name   = ccu.constraint_name
    AND rc.unique_constraint_schema = ccu.constraint_schema
WHERE kcu.table_schema = %s
"""

_MYSQL_FK_QUERY = """\
SELECT
    kcu.TABLE_NAME,
    kcu.COLUMN_NAME,
    kcu.REFERENCED_TABLE_NAME,
    kcu.REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE kcu
WHERE kcu.REFERENCED_TABLE_NAME IS NOT NULL
    AND kcu.TABLE_SCHEMA = DATABASE()
"""


def _introspect_postgresql(conn: Any, pg_schema: str = "public") -> Schema:
    """Introspect FK constraints from a PostgreSQL connection."""
    schema = Schema()
    cur = conn.cursor()
    try:
        cur.execute(_PG_FK_QUERY, (pg_schema,))
        for row in cur.fetchall():
            schema.add_foreign_key(ForeignKey(table=row[0], column=row[1], ref_table=row[2], ref_column=row[3]))
    finally:
        cur.close()
    return schema


def _introspect_mysql(conn: Any) -> Schema:
    """Introspect FK constraints from a MySQL connection."""
    schema = Schema()
    cur = conn.cursor()
    try:
        cur.execute(_MYSQL_FK_QUERY)
        for row in cur.fetchall():
            schema.add_foreign_key(ForeignKey(table=row[0], column=row[1], ref_table=row[2], ref_column=row[3]))
    finally:
        cur.close()
    return schema


def introspect_fks(conn: Any, *, pg_schema: str = "public", register: bool = True) -> Schema:
    """Introspect Foreign Key constraints from a live database connection.

    Detects the database driver (SQLite, PostgreSQL, MySQL) from the
    connection object and queries ``information_schema`` (or ``PRAGMA``
    for SQLite) to discover all FK relationships.

    Args:
        conn: A PEP 249 (DBAPI 2.0) connection object.
        pg_schema: PostgreSQL schema to introspect (default ``"public"``).
        register: If ``True`` (default), also register the resulting
            :class:`Schema` as the global schema via :func:`register_schema`.

    Returns:
        A :class:`Schema` populated with all discovered FK constraints.

    Raises:
        ValueError: If the database driver cannot be identified.
    """
    driver = _detect_driver(conn)

    if driver == "sqlite":
        schema = _introspect_sqlite(conn)
    elif driver == "postgresql":
        schema = _introspect_postgresql(conn, pg_schema=pg_schema)
    else:
        schema = _introspect_mysql(conn)

    if register:
        register_schema(schema)

    return schema
