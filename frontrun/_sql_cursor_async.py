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
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
from frontrun._sql_cursor import (
    _RE_INSERT_TABLE,
    _RE_UPDATE_TABLE,
    _acquire_pending_row_locks,
    _capture_insert_id,
    _detect_autobegin,
    _record_uncaptured_insert,
    _release_dpor_row_locks,
    _report_sql_access,
    _suppress_endpoint_io,
)

# ---------------------------------------------------------------------------
# Shared async DPOR scheduling + endpoint suppression
# ---------------------------------------------------------------------------


async def _dpor_schedule_and_suppress_async(
    reported: bool,
    execute: Callable[[], Awaitable[Any]],
) -> Any:
    """DPOR scheduling point + endpoint I/O suppression for async SQL execution.

    Shared core used by all async SQL interception paths:
    acquires pending row locks, forces a DPOR scheduling point if *reported*,
    suppresses endpoint-level I/O during the actual driver call, and releases
    row locks on exception.

    Args:
        reported: Whether ``_report_sql_access`` recorded table accesses.
        execute: A zero-argument async callable that performs the actual
            driver method call.
    """
    _acquire_pending_row_locks()
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
        _release_dpor_row_locks()
        raise


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
    reported = _report_sql_access(
        operation, parameters, db_obj=self, is_executemany=is_executemany, paramstyle=paramstyle
    )

    async def _execute() -> Any:
        if parameters is not None:
            return await original_method(self, operation, parameters)
        return await original_method(self, operation)

    result = await _dpor_schedule_and_suppress_async(reported, _execute)

    # Defect #6 fix: release row locks for 0-row UPDATEs.
    # An UPDATE that matches 0 rows acquires no real database row locks,
    # but frontrun's row-lock arbitration may have acquired a scheduler-level
    # lock based on the WHERE-clause resource ID.  Releasing it prevents
    # over-serialization that blocks DPOR from exploring interleavings where
    # both 0-row UPDATEs execute before either INSERT.
    if update_match is not None and reported:
        rowcount = getattr(self, "rowcount", -1)
        if rowcount == 0:
            _release_dpor_row_locks()

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
    **kwargs: Any,
) -> Any:
    """Intercept asyncpg connection methods (execute, fetch, fetchrow, fetchval).

    asyncpg uses ``$1``-style positional parameters passed as ``*args``, not
    a single parameters collection.  We report at table level (no parameter
    resolution for asyncpg's binary protocol parameters).
    """
    reported = _report_sql_access(operation, None, db_obj=self, is_executemany=False, paramstyle="dollar")
    return await _dpor_schedule_and_suppress_async(reported, lambda: original_method(self, operation, *args, **kwargs))


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
            reported = _report_sql_access(
                query, params, db_obj=self, is_executemany=method_name == "executemany", paramstyle="format"
            )
            return await _dpor_schedule_and_suppress_async(reported, lambda: orig(self, query, params, **kwargs))

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
            reported = _report_sql_access(
                query, args, db_obj=self, is_executemany=method_name == "executemany", paramstyle="pyformat"
            )
            return await _dpor_schedule_and_suppress_async(reported, lambda: orig(self, query, args, *extra, **kwargs))

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
            return await _intercept_asyncpg_execute(orig, self, query, *args, **kwargs)

        return wrap_method_metadata(_patched, orig, name=method_name)

    _patch_async_methods(conn_cls, ("execute", "fetch", "fetchrow", "fetchval"), _make_patched)

    # executemany on asyncpg takes (command, args) where args is a list of tuples
    orig_em = getattr(conn_cls, "executemany", None)
    if orig_em is not None:

        async def _patched_executemany(self: Any, command: Any, args: Any, **kwargs: Any) -> Any:
            reported = _report_sql_access(command, None, db_obj=self, is_executemany=True, paramstyle="dollar")
            return await _dpor_schedule_and_suppress_async(reported, lambda: orig_em(self, command, args, **kwargs))

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
