"""Shared helpers for SQLAlchemy contrib adapters."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Coroutine
from contextlib import ExitStack, contextmanager
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


@contextmanager
def _suppress_sync_reporting():
    from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

    suppress_sync_reporting()
    try:
        yield
    finally:
        unsuppress_sync_reporting()


@contextmanager
def _sync_connection(engine: Any):
    """Keep internal locks quiet and close even when method wrapping fails."""
    with ExitStack() as stack:
        with _suppress_sync_reporting():
            context = engine.connect()
            conn = context.__enter__()

        def close(*exc: Any) -> None:
            with _suppress_sync_reporting():
                context.__exit__(*exc)

        stack.push(close)
        yield conn


def wrap_sync_setup(engine: Any, setup: Callable[[], T]) -> Callable[[], T]:
    """Return a setup wrapper that disposes the engine before setup runs."""

    def wrapped_setup() -> T:
        with _suppress_sync_reporting():
            engine.dispose()
            return setup()

    return wrapped_setup


def wrap_sync_thread(
    engine: Any,
    current_connection: contextvars.ContextVar[Any],
    lock_timeout: int | None,
    fn: Callable[[T], Any],
) -> Callable[[T], Any]:
    """Return a thread wrapper that manages a per-thread SQLAlchemy connection."""

    def wrapper(state: T) -> Any:
        with ExitStack() as scope, _sync_connection(engine) as conn:
            with _suppress_sync_reporting():
                lock_timeout_sql = _lock_timeout_statement(lock_timeout)
                if lock_timeout_sql is not None:
                    from frontrun._sql_endpoint_suppression import suppress_sql_write

                    suppress_sql_write("BEGIN")
                    suppress_sql_write(lock_timeout_sql)
                    conn.exec_driver_sql(lock_timeout_sql)

            def _wrap_sa_method(method: Any, sql_write: str | None) -> Any:
                def wrapped(*args: Any, **kw: Any) -> Any:
                    with _suppress_sync_reporting():
                        if sql_write is not None:
                            from frontrun._sql_endpoint_suppression import suppress_sql_write

                            suppress_sql_write(sql_write)
                        return method(*args, **kw)

                return wrapped

            # Compilation and transaction-state locks are SQLAlchemy internals.
            for name, sql_write in (
                ("execute", None),
                ("exec_driver_sql", None),
                ("commit", "COMMIT"),
                ("rollback", "ROLLBACK"),
            ):
                setattr(conn, name, _wrap_sa_method(getattr(conn, name), sql_write))
            scope.enter_context(_current_connection_scope(current_connection, conn))
            return fn(state)

    return wrapper


def wrap_async_setup(engine: Any, setup: Callable[[], T]) -> Callable[[], T]:
    """Return a setup wrapper that disposes the async engine before setup runs."""

    def dispose_pool() -> None:
        sync_engine = engine.sync_engine
        dialect = getattr(sync_engine, "dialect", None)
        if dialect is None or not bool(getattr(dialect, "has_terminate", False)):
            # Older/custom async dialects may have no synchronous termination
            # hook. Detaching is safer than invoking an await-only close from
            # this synchronous setup callback, though the old pool then relies
            # on its normal GC cleanup.
            sync_engine.dispose(close=False)
            return

        # SQLAlchemy's async adapters implement do_terminate() specifically as
        # a synchronous, force-close path when no greenlet is active. Route the
        # pool's normal checked-in close through that hook for this disposal so
        # every execution gets a fresh pool without leaking the detached one.
        instance_dict = getattr(dialect, "__dict__", {})
        missing = object()
        previous = instance_dict.get("do_close", missing)
        try:
            dialect.do_close = dialect.do_terminate
        except (AttributeError, TypeError):
            sync_engine.dispose(close=False)
            return
        try:
            sync_engine.dispose()
        finally:
            if previous is missing:
                del dialect.do_close
            else:
                dialect.do_close = previous

    def wrapped_setup() -> T:
        with _suppress_sync_reporting():
            dispose_pool()
            return setup()

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
