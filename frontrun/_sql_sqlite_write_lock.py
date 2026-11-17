"""Arbitrate SQLite's writer before entering the driver's busy handler."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from typing import Any

from frontrun._io_detection import get_dpor_context, tx_store
from frontrun._sql_db_scope import _get_connection_db_scope
from frontrun._sql_row_locks import _release_dpor_row_locks

_RESOURCE_PREFIX = "sqlite:write-lock:"
# Only the leading keywords matter for transaction control. SQLite skips
# whitespace, empty statements and comments before the first keyword.
_KEYWORD = re.compile(r"(?:[\s;\ufeff]|--[^\n]*|/\*.*?(?:\*/|$))*+([A-Z_][\w$]*)", re.I | re.S)


def _sqlite_statement_needs_write_lock(operation: str) -> bool:
    first = _KEYWORD.match(operation)
    if first is None:
        return True
    keyword = first[1].upper()
    if keyword == "BEGIN":
        mode = _KEYWORD.match(operation, first.end())
        return mode is not None and mode[1].upper() in {"IMMEDIATE", "EXCLUSIVE"}
    if keyword == "WITH":
        # A CTE can end in SELECT or DML. Resolve it only when the optional
        # parser positively identifies a query.
        from frontrun._sql_parsing import parse_sql_access

        access = parse_sql_access(operation)
        if access.ast is not None:
            from sqlglot import exp  # type: ignore[import-untyped]

            return not isinstance(access.ast, exp.Query) or bool(access.write_tables)
    return keyword not in {"SELECT", "VALUES", "EXPLAIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"}


def is_sqlite_write_lock(resource_id: str) -> bool:
    return resource_id.startswith(_RESOURCE_PREFIX)


def sqlite_connection(db_obj: Any) -> sqlite3.Connection | None:
    if isinstance(db_obj, sqlite3.Connection):
        return db_obj
    if isinstance(db_obj, sqlite3.Cursor):
        return db_obj.connection
    # aiosqlite keeps the native connection here until close().
    native = getattr(db_obj, "_connection", None)
    return native if isinstance(native, sqlite3.Connection) else None


def sqlite_write_lock_for(db_obj: Any, operation: Any) -> str | None:
    if get_dpor_context() is None or sqlite_connection(db_obj) is None or not isinstance(operation, str):
        return None
    if not _sqlite_statement_needs_write_lock(operation):
        return None
    return f"{_RESOURCE_PREFIX}{_get_connection_db_scope(db_obj) or 'unknown'}"


def queue_sqlite_write_lock(resource_id: str) -> None:
    # SQLite grants the database writer before any row-level arbitration.
    # Taking a row first can deadlock with the owner of BEGIN IMMEDIATE.
    store = tx_store()
    store._pending_row_locks = [resource_id, *getattr(store, "_pending_row_locks", ())]


def settle_sqlite_write_lock(
    connection: sqlite3.Connection | None, acquired: Iterable[str] = (), *, transaction_ended: bool = False
) -> None:
    if connection is None:
        return
    store = tx_store()
    held = getattr(store, "_sqlite_write_locks", None)
    for resource in acquired:
        if is_sqlite_write_lock(resource):
            if held is None:
                held = store._sqlite_write_locks = {}
            held[resource] = connection
    if held and (transaction_ended or not connection.in_transaction):
        released = [resource for resource, owner in held.items() if owner is connection]
        for resource in released:
            del held[resource]
        if released:
            _release_dpor_row_locks(released)
