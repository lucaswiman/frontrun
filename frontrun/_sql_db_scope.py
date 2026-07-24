"""Stable database-scope identity tracking for SQL interception.

Given a DBAPI connection (or a cursor wrapping one), produce a short
deterministic token (the "db scope") that uniquely identifies the
underlying database. Resource IDs reported to the I/O reporter include
this token so that conflicts on tables that happen to share a name
across different databases stay distinct.

This module is the metadata layer underneath ``_sql_cursor`` — it has
no dependency on cursor patching or interception. Other parts of the
package (``_sql_cursor.py`` itself, async cursor helpers, etc.) import
from here.

The module-level ``_CONNECTION_DB_SCOPES`` and ``_table_primary_colset``
dicts are process-global state by design; ``clear_sql_metadata`` in
``_sql_cursor`` mutates them between DPOR exploration sessions.
"""

from __future__ import annotations

import hashlib
import os
import weakref
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

__all__ = [
    "_CONNECTION_DB_SCOPE_OWNERS",
    "_CONNECTION_DB_SCOPES",
    "_DB_SCOPE_ATTR",
    "_get_connection_db_scope",
    "_get_primary_colset",
    "_normalize_db_identity",
    "_normalize_table_identity",
    "_register_connection_db_scope",
    "_stable_db_scope",
    "_table_primary_colset",
    "_unregister_connection_db_scope",
]


def _normalize_table_identity(table: str) -> str:
    """Case-fold a table name for conflict-graph identity.

    Unquoted table identifiers are case-insensitive under PostgreSQL and SQLite
    (and MySQL with ``lower_case_table_names``), so ``Accounts`` and
    ``accounts`` denote the same physical table.  The parse layer preserves the
    written case, so fold it here: distinct resource IDs for one table would be
    an *under-merge* that hides a genuine same-row race.  Folding is a
    conservative over-merge — at worst it forces a spurious conflict between
    two quoted identifiers differing only in case, which stays sound.
    """
    return table.lower()


# ---------------------------------------------------------------------------
# DB scope identity tracking
# ---------------------------------------------------------------------------

_DB_SCOPE_ATTR = "_frontrun_db_scope"
_CONNECTION_DB_SCOPES: dict[int, str] = {}
# Some DBAPI connection types (notably raw sqlite3.Connection on supported
# Python versions) are neither weakref-able nor attribute-settable.  Keep a
# strong owner for those id-keyed entries until clear_sql_metadata(): this
# bounds retention to one exploration and makes address reuse impossible.
_CONNECTION_DB_SCOPE_OWNERS: dict[int, Any] = {}


# Global to track primary column set per (db_scope, table) for cross-column
# conflict detection.  Keyed by (db_scope, table) rather than just table to
# avoid cross-database contamination when the same table name exists in
# multiple databases with different schemas/access patterns.
_table_primary_colset: dict[tuple[str | None, str], tuple[str, ...]] = {}


def _get_primary_colset(table: str, colset: tuple[str, ...], *, db_scope: str | None = None) -> tuple[str, ...]:
    """Return the primary column set for a table, initializing it if necessary."""
    return _table_primary_colset.setdefault((db_scope, _normalize_table_identity(table)), colset)


def _stable_db_scope(identity: str) -> str:
    """Return a short deterministic token for a database identity string."""
    return hashlib.sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _register_connection_db_scope(connection: Any, identity: str) -> str:
    """Associate a stable database scope with a connection object."""
    scope = _stable_db_scope(identity)
    key = id(connection)
    _CONNECTION_DB_SCOPES[key] = scope
    # Evict the entry when the connection is collected: id() values are
    # reusable, and a stale mapping would hand this scope to whatever
    # unrelated object next occupies the address. Weakref callbacks fire
    # during dealloc, before the address can be reused. Some connection
    # types are not weakref-able; their entries stay (as before).
    try:
        weakref.finalize(connection, _CONNECTION_DB_SCOPES.pop, key, None)
    except TypeError:
        _CONNECTION_DB_SCOPE_OWNERS[key] = connection
    try:
        setattr(connection, _DB_SCOPE_ATTR, scope)
    except AttributeError:
        pass
    return scope


def _unregister_connection_db_scope(connection: Any) -> None:
    """Release id-keyed metadata after a physical connection closes."""
    key = id(connection)
    owner = _CONNECTION_DB_SCOPE_OWNERS.get(key)
    if owner is None or owner is connection:
        _CONNECTION_DB_SCOPES.pop(key, None)
        _CONNECTION_DB_SCOPE_OWNERS.pop(key, None)
    try:
        delattr(connection, _DB_SCOPE_ATTR)
    except AttributeError:
        pass


def _normalize_db_identity(kind: str, *args: Any, **kwargs: Any) -> str | None:
    """Build a canonical database identity string, dispatching on ``kind``.

    * ``"sqlite"``     — from ``sqlite3.connect`` positional/keyword args.
    * ``"mapping"``    — from ``(driver, mapping_dict)``.
    * ``"connection"`` — inferred from a live DBAPI connection.
    """
    if kind == "mapping":
        driver, mapping = args
        driver_name = str(driver).lower()
        if driver_name in {"postgres", "postgresql", "psycopg", "psycopg2", "asyncpg"}:
            driver_name = "postgresql"
        elif driver_name in {"mysql", "pymysql", "mysqldb", "aiomysql", "mysql.connector"}:
            driver_name = "mysql"
        # Host spelling is not a stable database identity: DNS aliases and
        # Unix/TCP endpoints can reach one server.  Omitting it conservatively
        # merges same-driver/port/database resources across distinct servers,
        # which may add paths but cannot hide a real dependency.
        items = [(k, v) for k, v in sorted(mapping.items()) if k != "host" and v not in (None, "")]
        return f"{driver_name}:{repr(items)}" if items else None
    if kind == "sqlite":
        database = kwargs.get("database") or (args[0] if args else None)
        if database is None:
            return None
        raw = os.fspath(database)
        s = raw.decode("utf-8", errors="surrogateescape") if isinstance(raw, bytes) else raw
        use_uri = bool(kwargs.get("uri"))
        if s == ":memory:" and not use_uri:
            return None
        if use_uri or s.startswith("file:"):
            parsed = urlsplit(s)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if query.get("mode", "").lower() == "memory" or parsed.path == ":memory:":
                # Named shared-memory databases are identified by their URI;
                # they are not filesystem paths and must not be realpathed.
                canonical_query = "&".join(f"{key}={query[key]}" for key in sorted(query))
                suffix = f"?{canonical_query}" if canonical_query else ""
                return f"sqlite-memory-uri:{parsed.path}{suffix}"
            uri_path = unquote(parsed.path)
            if parsed.netloc and parsed.netloc not in ("", "localhost"):
                uri_path = f"//{parsed.netloc}{uri_path}"
            return f"sqlite-path:{os.path.realpath(os.path.abspath(uri_path))}"
        return f"sqlite-path:{os.path.realpath(os.path.abspath(s))}"
    if kind == "connection":
        (conn,) = args
        info = getattr(conn, "info", None)
        dsn_params = getattr(info, "dsn_parameters", None)
        if isinstance(dsn_params, dict):
            relevant = {k: dsn_params.get(k) for k in ("host", "port", "dbname") if dsn_params.get(k)}
            if (identity := _normalize_db_identity("mapping", "postgres", relevant)) is not None:
                return identity
        dsn = getattr(conn, "dsn", None)
        relevant = {
            "host": getattr(conn, "host", None),
            "port": getattr(conn, "port", None),
            "database": getattr(conn, "database", None),
            "db": getattr(conn, "db", None),
            "dbname": getattr(info, "dbname", None),
        }
        if (identity := _normalize_db_identity("mapping", "dbapi", relevant)) is not None:
            return identity
        if isinstance(dsn, str) and dsn:
            return f"dsn:{dsn}"
        path = getattr(conn, "filename", None)
        if isinstance(path, str) and path:
            return f"sqlite-path:{os.path.realpath(os.path.abspath(path))}"
        return None
    raise ValueError(f"unknown db identity kind: {kind!r}")


def _get_connection_db_scope(db_obj: Any) -> str | None:
    """Resolve the stable database scope for a cursor/connection-like object."""
    if db_obj is None:
        return None
    if type(db_obj).__module__.startswith("unittest.mock"):
        return None

    seen: set[int] = set()
    pending = [db_obj]
    while pending:
        candidate = pending.pop(0)
        if type(candidate).__module__.startswith("unittest.mock"):
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        scope = getattr(candidate, _DB_SCOPE_ATTR, None)
        if isinstance(scope, str):
            return scope

        mapped_scope = _CONNECTION_DB_SCOPES.get(candidate_id)
        if mapped_scope is not None:
            return mapped_scope

        for attr in ("connection", "_conn", "_connection"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                pending.append(nested)

    connection = getattr(db_obj, "connection", None)
    if connection is None:
        connection = getattr(db_obj, "_conn", None)
    if connection is None:
        connection = db_obj

    identity = _normalize_db_identity("connection", connection)
    if identity is None:
        return None
    return _register_connection_db_scope(connection, identity)
