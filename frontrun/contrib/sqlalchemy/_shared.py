"""Shared helpers for SQLAlchemy contrib adapters."""

from __future__ import annotations

import contextvars
import sys
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")


def make_current_connection_var(name: str) -> contextvars.ContextVar[Any]:
    """Create a context variable for exposing the active SQLAlchemy connection."""
    return contextvars.ContextVar(name)


def _lock_timeout_statement(lock_timeout: int | None) -> str | None:
    """Build the SQL statement used to set a per-connection lock timeout."""
    if lock_timeout is None:
        return None
    return f"SET lock_timeout = '{int(lock_timeout)}ms'"


@contextmanager
def _current_connection_scope(
    current_connection: contextvars.ContextVar[Any],
    conn: Any,
):
    """Expose the active connection in a context variable."""
    token = current_connection.set(conn)
    try:
        yield
    finally:
        current_connection.reset(token)


def wrap_sync_setup(engine: Any, setup: Callable[[], T]) -> Callable[[], T]:
    """Return a setup wrapper that disposes the engine before setup runs."""

    def wrapped_setup() -> T:
        from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

        suppress_sync_reporting()
        try:
            engine.dispose()
            return setup()
        finally:
            unsuppress_sync_reporting()

    return wrapped_setup


def wrap_sync_thread(
    engine: Any,
    current_connection: contextvars.ContextVar[Any],
    lock_timeout: int | None,
    fn: Callable[[T], Any],
) -> Callable[[T], Any]:
    """Return a thread wrapper that manages a per-thread SQLAlchemy connection."""

    def wrapper(state: T) -> Any:
        from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

        # Suppress cooperative lock sync events during connection setup
        # and teardown.  Internal SQLAlchemy/psycopg2 locks are implementation
        # details that shouldn't create DPOR sync events.
        suppress_sync_reporting()
        try:
            conn_ctx = engine.connect()
            conn = conn_ctx.__enter__()
        except BaseException:
            unsuppress_sync_reporting()
            raise
        # conn_ctx is now entered — guarantee __exit__ via outer finally.
        try:
            lock_timeout_sql = _lock_timeout_statement(lock_timeout)
            if lock_timeout_sql is not None:
                from frontrun._sql_endpoint_suppression import suppress_sql_write

                suppress_sql_write("BEGIN")
                suppress_sql_write(lock_timeout_sql)
                conn.exec_driver_sql(lock_timeout_sql)
        except BaseException:
            unsuppress_sync_reporting()
            suppress_sync_reporting()
            try:
                conn_ctx.__exit__(*sys.exc_info())
            finally:
                unsuppress_sync_reporting()
            raise
        unsuppress_sync_reporting()
        # Wrap SA Connection methods that trigger internal locks.
        # conn.execute() acquires statement compilation locks.
        # conn.commit()/rollback() acquires transaction state locks.
        _orig_execute = conn.execute
        _orig_exec_driver_sql = conn.exec_driver_sql
        _orig_commit = conn.commit
        _orig_rollback = conn.rollback

        def _wrap_sa_method(method: Any, sql_write: str | None = None) -> Any:
            def wrapped(*args: Any, **kw: Any) -> Any:
                suppress_sync_reporting()
                try:
                    if sql_write is not None:
                        from frontrun._sql_endpoint_suppression import suppress_sql_write

                        suppress_sql_write(sql_write)
                    return method(*args, **kw)
                finally:
                    unsuppress_sync_reporting()

            return wrapped

        conn.execute = _wrap_sa_method(_orig_execute)  # type: ignore[method-assign]
        conn.exec_driver_sql = _wrap_sa_method(_orig_exec_driver_sql)  # type: ignore[method-assign]
        conn.commit = _wrap_sa_method(_orig_commit, "COMMIT")  # type: ignore[method-assign]
        conn.rollback = _wrap_sa_method(_orig_rollback, "ROLLBACK")  # type: ignore[method-assign]
        exc_info: tuple[type[BaseException], BaseException, object] | tuple[None, None, None] = (None, None, None)
        with _current_connection_scope(current_connection, conn):
            try:
                return fn(state)
            except BaseException:
                exc_info = sys.exc_info()  # type: ignore[assignment]
                raise
            finally:
                suppress_sync_reporting()
                try:
                    conn_ctx.__exit__(*exc_info)
                finally:
                    unsuppress_sync_reporting()

    return wrapper


def wrap_async_setup(engine: Any, setup: Callable[[], T]) -> Callable[[], T]:
    """Return a setup wrapper that disposes the async engine before setup runs."""

    def wrapped_setup() -> T:
        from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

        suppress_sync_reporting()
        try:
            engine.sync_engine.dispose()
            return setup()
        finally:
            unsuppress_sync_reporting()

    return wrapped_setup


def wrap_async_task(
    engine: Any,
    current_connection: contextvars.ContextVar[Any],
    lock_timeout: int | None,
    fn: Callable[[T], Coroutine[Any, Any, None]],
) -> Callable[[T], Coroutine[Any, Any, None]]:
    """Return a task wrapper that manages a per-task async SQLAlchemy connection."""

    async def wrapper(state: T) -> None:
        async with engine.connect() as conn:
            lock_timeout_sql = _lock_timeout_statement(lock_timeout)
            if lock_timeout_sql is not None:
                await conn.exec_driver_sql(lock_timeout_sql)
            with _current_connection_scope(current_connection, conn):
                await fn(state)

    return wrapper
