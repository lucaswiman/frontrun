"""Async DBAPI cursor monkey-patching for SQL-level conflict detection.

Async counterpart to ``_sql_cursor.py``.  Intercepts ``cursor.execute()``
and ``cursor.executemany()`` on async database drivers to extract
table-level read/write sets from SQL statements.

Supported drivers:

* **aiosqlite** — async wrapper around sqlite3
* **psycopg.AsyncCursor** — psycopg3 async mode
* **aiomysql** — async MySQL driver
* **asyncpg** — PostgreSQL async driver (connection-level methods)

The SQL parsing, resource reporting, and transaction grouping logic is
shared with the sync module via ``_report_sql_access``.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from typing import Any

from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._io_detection import tx_store
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
from frontrun._sql_cursor import (
    _RE_INSERT_TABLE,
    _RE_UPDATE_TABLE,
    _capture_insert_id,
    _connection_for_db_object,
    _detect_autobegin,
    _is_postgresql_db_object,
    _record_uncaptured_insert,
    _release_dpor_row_locks,
    _report_sql_access,
    _suppress_endpoint_io,
)
from frontrun._sql_transactions import _apply_tx_op_after_success

# ---------------------------------------------------------------------------
# Shared async DPOR scheduling + endpoint suppression
# ---------------------------------------------------------------------------


async def _acquire_pending_row_locks_async() -> list[str]:
    """Drain and asynchronously acquire the current task's modeled row locks."""
    store = tx_store()
    lock_resources = getattr(store, "_pending_row_locks", None)
    if not lock_resources:
        return []
    store._pending_row_locks = []
    lock_resources = list(dict.fromkeys(lock_resources))
    ctx = _get_dpor_context()
    if ctx is None:
        return []
    scheduler, task_id = ctx
    acquire_async = getattr(scheduler, "acquire_row_locks_async", None)
    if acquire_async is None:
        acquired = scheduler.acquire_row_locks(task_id, lock_resources)
    else:
        acquired = await acquire_async(task_id, lock_resources)
    if acquired is None:
        acquired = lock_resources
    held = getattr(store, "_held_row_locks", None)
    if held is None:
        held = set()
        store._held_row_locks = held
    held.update(acquired)
    return list(acquired)


async def _dpor_schedule_and_suppress_async(
    reported: bool,
    execute: Callable[[], Awaitable[Any]],
    acquired_row_locks: list[str] | None = None,
    *,
    release_locks_on_error: bool = True,
) -> Any:
    """DPOR scheduling point + endpoint I/O suppression for async SQL execution.

    Shared core used by all async SQL interception paths:
    acquires pending row locks, forces a DPOR scheduling point if *reported*,
    suppresses endpoint-level I/O during the actual driver call, and releases
    row locks on exception.

    Divergence from the sync path (``_sql_cursor._dpor_schedule_and_suppress_sync``):
    the sync path grants the exclusive scheduler turn *before* execute() and
    acquires row locks inside that turn, so a thread cannot become the modeled
    row-lock holder while still waiting to be scheduled.  This async path keeps
    the older ordering — ``_acquire_pending_row_locks()`` runs *before*
    ``report_and_wait`` and no turn is held across the driver call — and that is
    sound here because async tasks are cooperatively single-threaded: only one
    task runs at a time and there is no preemption window between becoming the
    modeled holder and being scheduled, so the sync path's serialization guard
    is unnecessary.

    Args:
        reported: Whether ``_report_sql_access`` recorded table accesses.
        execute: A zero-argument async callable that performs the actual
            driver method call.
    """
    acquired = await _acquire_pending_row_locks_async()
    if acquired_row_locks is not None:
        acquired_row_locks.extend(acquired)
    if reported:
        _dpor_ctx = _get_dpor_context()
        if _dpor_ctx is not None:
            _dpor_ctx[0].report_and_wait(None, _dpor_ctx[1])
    try:
        if reported:
            with _suppress_endpoint_io():
                return await execute()
        return await execute()
    except Exception:
        if release_locks_on_error:
            _release_dpor_row_locks(acquired)
        raise


async def _execute_and_finalize_tx_end(
    execute: Callable[[], Awaitable[Any]], tx_op: Any | None, connection: Any = None
) -> Any:
    """Finalize modeled transaction state only after physical async I/O succeeds."""
    result = await execute()
    if tx_op is not None:
        _apply_tx_op_after_success(tx_op, _connection_for_db_object(connection))
    return result


# ---------------------------------------------------------------------------
# Async interception
# ---------------------------------------------------------------------------


async def _intercept_execute_async(
    original_method: Any,
    self: Any,
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> Any:
    """Async version of ``_intercept_execute``.

    Parses *operation*, reports table accesses via the shared
    ``_report_sql_access`` helper, then ``await``s the original async method.
    """
    insert_match = _RE_INSERT_TABLE.match(operation) if isinstance(operation, str) else None
    update_match = _RE_UPDATE_TABLE.match(operation) if isinstance(operation, str) else None
    _detect_autobegin(self)
    deferred_tx_end: list[Any] = []
    reported = _report_sql_access(
        operation,
        parameters,
        db_obj=self,
        is_executemany=is_executemany,
        paramstyle=paramstyle,
        defer_tx_lock_release=True,
        deferred_tx_end=deferred_tx_end,
    )
    deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None

    async def _execute() -> Any:
        if parameters is not None:
            return await _execute_and_finalize_tx_end(
                lambda: original_method(self, operation, parameters), deferred_tx_op, self
            )
        return await _execute_and_finalize_tx_end(lambda: original_method(self, operation), deferred_tx_op, self)

    statement_row_locks: list[str] = []
    result = await _dpor_schedule_and_suppress_async(
        reported,
        _execute,
        statement_row_locks,
        release_locks_on_error=deferred_tx_op is None,
    )

    # Defect #6 fix: release only this statement's speculative lock for a
    # PostgreSQL 0-row UPDATE. Other transaction locks remain held until end.
    if update_match is not None and reported and statement_row_locks and _is_postgresql_db_object(self):
        rowcount = getattr(self, "rowcount", -1)
        if rowcount == 0:
            _release_dpor_row_locks(statement_row_locks)

    # Post-INSERT: capture lastrowid and record indexical alias
    if insert_match is not None and reported:
        if not is_executemany:
            _capture_insert_id(self, insert_match.group(1))
        else:
            _record_uncaptured_insert(self, insert_match.group(1))

    return result


async def _intercept_asyncpg_execute(
    original_method: Any,
    self: Any,
    operation: Any,
    *args: Any,
    method_name: str = "execute",
    **kwargs: Any,
) -> Any:
    """Intercept asyncpg connection methods (execute, fetch, fetchrow, fetchval).

    asyncpg uses ``$1``-style positional parameters passed as ``*args``, not
    a single parameters collection.  We report at table level (no parameter
    resolution for asyncpg's binary protocol parameters).
    """
    update_match = _RE_UPDATE_TABLE.match(operation) if isinstance(operation, str) else None
    deferred_tx_end: list[Any] = []
    reported = _report_sql_access(
        operation,
        None,
        db_obj=self,
        is_executemany=False,
        paramstyle="dollar",
        defer_tx_lock_release=True,
        deferred_tx_end=deferred_tx_end,
    )
    deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None
    statement_row_locks: list[str] = []
    result = await _dpor_schedule_and_suppress_async(
        reported,
        lambda: _execute_and_finalize_tx_end(
            lambda: original_method(self, operation, *args, **kwargs), deferred_tx_op, self
        ),
        statement_row_locks,
        release_locks_on_error=deferred_tx_op is None,
    )
    zero_rows = (
        (method_name == "execute" and result == "UPDATE 0")
        or (method_name == "fetch" and result == [])
        or (method_name == "fetchrow" and result is None)
    )
    # fetchval() returns None both for no row and for a matched row whose first
    # RETURNING value is SQL NULL, so retaining the lock is the only sound choice.
    if update_match is not None and reported and statement_row_locks and zero_rows:
        _release_dpor_row_locks(statement_row_locks)
    return result


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_sql_async_patched = False
_ASYNC_PATCHES: list[tuple[Any, str, Any]] = []
_ASYNC_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


def _patch_async_methods(
    target_cls: Any,
    method_names: tuple[str, ...],
    make_wrapper: Callable[[Any, str], Any],
) -> None:
    """Patch several async methods on one target class."""
    for method_name in method_names:
        patch_method(
            target_cls,
            method_name,
            originals=_ASYNC_ORIGINAL_METHODS,
            patches=_ASYNC_PATCHES,
            make_wrapper=lambda orig, _method_name=method_name: make_wrapper(orig, _method_name),
        )


# ---------------------------------------------------------------------------
# aiosqlite patching
# ---------------------------------------------------------------------------


def _patch_aiosqlite() -> None:
    """Patch aiosqlite.Cursor and aiosqlite.Connection execute/executemany."""
    try:
        import aiosqlite  # type: ignore[import-untyped]
    except ImportError:
        return

    # Patch both Cursor and Connection — users commonly use conn.execute()
    for target_cls in (aiosqlite.Cursor, aiosqlite.Connection):

        def _make_patched(orig: Any, method_name: str) -> Any:
            # ``parameters`` matches the unpatched aiosqlite signature
            # (execute(self, sql, parameters=None)) and is NOT positional-only,
            # so callers may pass it by keyword (F7).
            async def _patched(self: Any, sql: Any, parameters: Any = None) -> Any:
                return await _intercept_execute_async(
                    orig, self, sql, parameters, is_executemany=method_name == "executemany", paramstyle="qmark"
                )

            return wrap_method_metadata(_patched, orig, name=method_name)

        _patch_async_methods(target_cls, ("execute", "executemany"), _make_patched)


# ---------------------------------------------------------------------------
# psycopg AsyncCursor patching
# ---------------------------------------------------------------------------


def _patch_psycopg_async() -> None:
    """Patch psycopg.AsyncCursor.execute and executemany."""
    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        return

    cursor_cls = getattr(psycopg, "AsyncCursor", None)
    if cursor_cls is None:
        return

    def _make_patched(orig: Any, method_name: str) -> Any:
        # ``params`` matches the unpatched psycopg AsyncCursor signature and is
        # NOT positional-only, so ``execute(query, params=[...])`` resolves it
        # here (rather than swallowing it into **kwargs and losing row-level
        # resolution) — F7.
        async def _patched(self: Any, query: Any, params: Any = None, **kwargs: Any) -> Any:
            update_match = _RE_UPDATE_TABLE.match(query) if isinstance(query, str) else None
            deferred_tx_end: list[Any] = []
            reported = _report_sql_access(
                query,
                params,
                db_obj=self,
                is_executemany=method_name == "executemany",
                paramstyle="format",
                defer_tx_lock_release=True,
                deferred_tx_end=deferred_tx_end,
            )
            deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None
            statement_row_locks: list[str] = []
            result = await _dpor_schedule_and_suppress_async(
                reported,
                lambda: _execute_and_finalize_tx_end(lambda: orig(self, query, params, **kwargs), deferred_tx_op, self),
                statement_row_locks,
                release_locks_on_error=deferred_tx_op is None,
            )
            if update_match is not None and reported and statement_row_locks and getattr(self, "rowcount", -1) == 0:
                _release_dpor_row_locks(statement_row_locks)
            return result

        return wrap_method_metadata(_patched, orig, name=method_name)

    _patch_async_methods(cursor_cls, ("execute", "executemany"), _make_patched)


# ---------------------------------------------------------------------------
# aiomysql patching
# ---------------------------------------------------------------------------


def _patch_aiomysql() -> None:
    """Patch aiomysql.Cursor.execute and executemany."""
    try:
        mod = importlib.import_module("aiomysql.cursors")
    except ImportError:
        return

    cursor_cls = getattr(mod, "Cursor", None)
    if cursor_cls is None:
        return

    def _make_patched(orig: Any, method_name: str) -> Any:
        async def _patched(self: Any, query: Any, args: Any = None, *extra: Any, **kwargs: Any) -> Any:
            deferred_tx_end: list[Any] = []
            reported = _report_sql_access(
                query,
                args,
                db_obj=self,
                is_executemany=method_name == "executemany",
                paramstyle="pyformat",
                defer_tx_lock_release=True,
                deferred_tx_end=deferred_tx_end,
            )
            deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None
            return await _dpor_schedule_and_suppress_async(
                reported,
                lambda: _execute_and_finalize_tx_end(
                    lambda: orig(self, query, args, *extra, **kwargs), deferred_tx_op, self
                ),
                release_locks_on_error=deferred_tx_op is None,
            )

        return wrap_method_metadata(_patched, orig, name=method_name)

    _patch_async_methods(cursor_cls, ("execute", "executemany"), _make_patched)


# ---------------------------------------------------------------------------
# asyncpg patching
# ---------------------------------------------------------------------------


def _patch_asyncpg() -> None:
    """Patch asyncpg.Connection query methods (execute, fetch, fetchrow, fetchval, executemany)."""
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        return

    conn_cls = asyncpg.Connection

    # asyncpg methods all take (query, *args) — no separate params arg.
    # execute returns command tag, fetch/fetchrow/fetchval return results.
    def _make_patched(orig: Any, method_name: str) -> Any:
        async def _patched(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            return await _intercept_asyncpg_execute(orig, self, query, *args, method_name=method_name, **kwargs)

        return wrap_method_metadata(_patched, orig, name=method_name)

    _patch_async_methods(conn_cls, ("execute", "fetch", "fetchrow", "fetchval"), _make_patched)

    # executemany on asyncpg takes (command, args) where args is a list of tuples
    orig_em = getattr(conn_cls, "executemany", None)
    if orig_em is not None:

        async def _patched_executemany(self: Any, command: Any, args: Any, **kwargs: Any) -> Any:
            deferred_tx_end: list[Any] = []
            reported = _report_sql_access(
                command,
                None,
                db_obj=self,
                is_executemany=True,
                paramstyle="dollar",
                defer_tx_lock_release=True,
                deferred_tx_end=deferred_tx_end,
            )
            deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None
            return await _dpor_schedule_and_suppress_async(
                reported,
                lambda: _execute_and_finalize_tx_end(
                    lambda: orig_em(self, command, args, **kwargs), deferred_tx_op, self
                ),
                release_locks_on_error=deferred_tx_op is None,
            )

        patch_method(
            conn_cls,
            "executemany",
            originals=_ASYNC_ORIGINAL_METHODS,
            patches=_ASYNC_PATCHES,
            make_wrapper=lambda original: wrap_method_metadata(_patched_executemany, original, name="executemany"),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_sql_async() -> None:
    """Monkey-patch async DBAPI cursor.execute() for known drivers."""
    global _sql_async_patched  # noqa: PLW0603
    if _sql_async_patched:
        return

    _patch_aiosqlite()
    _patch_psycopg_async()
    _patch_aiomysql()
    _patch_asyncpg()

    _sql_async_patched = True


def unpatch_sql_async() -> None:
    """Restore original async DBAPI cursor methods."""
    global _sql_async_patched  # noqa: PLW0603
    if not _sql_async_patched:
        return
    restore_patches(_ASYNC_PATCHES)
    _ASYNC_PATCHES.clear()
    _ASYNC_ORIGINAL_METHODS.clear()
    _sql_async_patched = False
