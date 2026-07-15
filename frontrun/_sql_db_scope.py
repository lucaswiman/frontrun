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

__all__ = [
    "_CONNECTION_DB_SCOPES",
    "_DB_SCOPE_ATTR",
    "_get_connection_db_scope",
    "_get_primary_colset",
    "_normalize_db_identity",
    "_register_connection_db_scope",
    "_stable_db_scope",
    "_table_primary_colset",
]


# ---------------------------------------------------------------------------
# DB scope identity tracking
# ---------------------------------------------------------------------------

_DB_SCOPE_ATTR = "_frontrun_db_scope"
_CONNECTION_DB_SCOPES: dict[int, str] = {}


# Global to track primary column set per (db_scope, table) for cross-column
# conflict detection.  Keyed by (db_scope, table) rather than just table to
# avoid cross-database contamination when the same table name exists in
# multiple databases with different schemas/access patterns.
_table_primary_colset: dict[tuple[str | None, str], tuple[str, ...]] = {}


def _get_primary_colset(table: str, colset: tuple[str, ...], *, db_scope: str | None = None) -> tuple[str, ...]:
    """Return the primary column set for a table, initializing it if necessary."""
    return _table_primary_colset.setdefault((db_scope, table), colset)


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
        pass
    try:
        setattr(connection, _DB_SCOPE_ATTR, scope)
    except AttributeError:
        pass
    return scope


def _normalize_db_identity(kind: str, *args: Any, **kwargs: Any) -> str | None:
    """Build a canonical database identity string, dispatching on ``kind``.

    * ``"sqlite"``     — from ``sqlite3.connect`` positional/keyword args.
    * ``"mapping"``    — from ``(driver, mapping_dict)``.
    * ``"connection"`` — inferred from a live DBAPI connection.
    """
    if kind == "mapping":
        driver, mapping = args
        items = [(k, v) for k, v in sorted(mapping.items()) if v not in (None, "")]
        return f"{driver}:{repr(items)}" if items else None
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
            return f"sqlite-uri:{s}"
        return f"sqlite-path:{os.path.abspath(s)}"
    if kind == "connection":
        (conn,) = args
        info = getattr(conn, "info", None)
        dsn_params = getattr(info, "dsn_parameters", None)
        if isinstance(dsn_params, dict):
            relevant = {k: dsn_params.get(k) for k in ("host", "port", "dbname") if dsn_params.get(k)}
            if (identity := _normalize_db_identity("mapping", "postgres", relevant)) is not None:
                return identity
        dsn = getattr(conn, "dsn", None)
        if isinstance(dsn, str) and dsn:
            return f"dsn:{dsn}"
        relevant = {
            "host": getattr(conn, "host", None),
            "port": getattr(conn, "port", None),
            "database": getattr(conn, "database", None),
            "db": getattr(conn, "db", None),
            "dbname": getattr(info, "dbname", None),
        }
        if (identity := _normalize_db_identity("mapping", "dbapi", relevant)) is not None:
            return identity
        path = getattr(conn, "filename", None)
        if isinstance(path, str) and path:
            return f"sqlite-path:{os.path.abspath(path)}"
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
