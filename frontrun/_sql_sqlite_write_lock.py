"""SQLite's database write lock, modeled as a DPOR row lock.

SQLite allows one writer per database file. A connection takes the write lock
at ``BEGIN IMMEDIATE`` / ``BEGIN EXCLUSIVE`` or at its first write statement
and holds it until its transaction ends; an autocommit write holds it for the
statement. A second writer waits in SQLite's busy handler, inside the C call,
where the cooperative scheduler cannot run the holder.

The lock is therefore arbitrated by the scheduler as a row lock on one
resource per database (``sqlite:write-lock:<db scope>``): it is queued before
any statement that may write and acquired with the statement's other row
locks, inside the SQL scheduling turn. It is released from the connection's
own state (``Connection.in_transaction``) after each statement and after
``commit()`` / ``rollback()`` / ``close()``, so explicit, deferred, legacy
implicit and autocommit transactions are handled alike, and a statement that
fails inside an open transaction keeps the lock SQLite still holds. Generic
"release every row lock" paths leave these resources alone
(:func:`frontrun._sql_row_locks._release_dpor_row_locks`).

Classification over-approximates: only statements positively identified as
read-only queries or as transaction control that takes no lock skip it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from frontrun._io_detection import tx_store
from frontrun._sql_db_scope import _get_connection_db_scope
from frontrun._sql_parsing import TxOp, parse_sql_access
from frontrun._sql_row_locks import _release_dpor_row_locks

__all__ = [
    "is_sqlite_write_lock",
    "queue_sqlite_write_lock",
    "settle_sqlite_write_lock",
    "sqlite_write_lock_for",
]

_RESOURCE_PREFIX = "sqlite:write-lock:"
_LOCKING_BEGIN_MODES = frozenset({"IMMEDIATE", "EXCLUSIVE"})


def _sqlite_statement_needs_write_lock(operation: str) -> bool:
    """Whether executing *operation* may take SQLite's database write lock."""
    from sqlglot import exp  # type: ignore[import-untyped]

    access = parse_sql_access(operation)
    if access.tx_op is TxOp.BEGIN:
        if access.ast is None:
            return True
        mode = access.ast.args.get("this")
        return mode is not None and str(mode).upper() in _LOCKING_BEGIN_MODES
    if access.tx_op is not None:
        return False
    return not (isinstance(access.ast, exp.Query) and not access.write_tables)


def is_sqlite_write_lock(resource_id: str) -> bool:
    return resource_id.startswith(_RESOURCE_PREFIX)


def _sqlite_connection(db_obj: Any) -> sqlite3.Connection | None:
    """The ``sqlite3.Connection`` behind *db_obj*: itself, a cursor's, or the
    one an async wrapper (aiosqlite) runs on its worker thread. Looks at most
    two attribute hops away (an aiosqlite cursor's connection's connection)."""
    level = [db_obj]
    for _ in range(3):
        following: list[Any] = []
        for candidate in level:
            if isinstance(candidate, sqlite3.Connection):
                return candidate
            if candidate is None or type(candidate).__module__.startswith("unittest.mock"):
                continue
            for attr in ("connection", "_conn", "_connection"):
                try:
                    following.append(getattr(candidate, attr, None))
                except Exception:  # e.g. aiosqlite's _conn on a closed connection
                    continue
        level = following
    return None


def sqlite_write_lock_for(db_obj: Any, operation: Any) -> str | None:
    """The write-lock resource *operation* needs on *db_obj*'s database, if any."""
    connection = _sqlite_connection(db_obj)
    if connection is None or not isinstance(operation, str) or not _sqlite_statement_needs_write_lock(operation):
        return None
    # An unknown scope shares one resource: over-serializing is safe.
    return f"{_RESOURCE_PREFIX}{_get_connection_db_scope(connection) or 'unknown'}"


def queue_sqlite_write_lock(resource_id: str) -> None:
    """Queue *resource_id* for the next row-lock acquisition on this worker."""
    store = tx_store()
    pending = getattr(store, "_pending_row_locks", None)
    if pending is None:
        pending = []
        store._pending_row_locks = pending
    pending.append(resource_id)


def _in_transaction(connection: sqlite3.Connection) -> bool:
    try:
        return connection.in_transaction
    except sqlite3.ProgrammingError:  # closed: SQLite released its locks
        return False


def settle_sqlite_write_lock(db_obj: Any, acquired: Iterable[str] = ()) -> None:
    """Record write locks in *acquired* for *db_obj*'s connection, then release
    every write lock this worker holds whose connection's transaction ended."""
    store = tx_store()
    held: dict[str, sqlite3.Connection] | None = getattr(store, "_sqlite_write_locks", None)
    if held is None:
        held = {}
        store._sqlite_write_locks = held
    connection = _sqlite_connection(db_obj)
    if connection is not None:
        for resource_id in acquired:
            if is_sqlite_write_lock(resource_id):
                held.setdefault(resource_id, connection)
    released = [resource_id for resource_id, owner in held.items() if not _in_transaction(owner)]
    for resource_id in released:
        del held[resource_id]
    if released:
        _release_dpor_row_locks(released)
