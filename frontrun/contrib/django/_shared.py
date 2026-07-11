"""Shared helpers for Django contrib adapters."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")

#: Default ``trace_packages`` for Django projects.  Traces code inside any
#: ``django_*`` third-party app (e.g. ``django_filters``, ``django_rest_framework``)
#: and ``django.contrib.sites`` submodules, since these commonly participate
#: in race conditions visible during integration tests.
DJANGO_TRACE_PACKAGES: list[str] = ["django_*", "django.contrib.sites.*"]


def wrap_setup(setup: Callable[[], T], close_all: Callable[[], None]) -> Callable[[], T]:
    """Return a setup wrapper that clears all Django connections first."""

    def wrapped_setup() -> T:
        close_all()
        return setup()

    return wrapped_setup


@contextmanager
def _fresh_connection(connections: Any, db_alias: str, lock_timeout: int | None):
    """Open a fresh Django connection for the duration of one task."""
    conn = connections[db_alias]
    conn.close()
    try:
        conn.ensure_connection()
        if lock_timeout is not None:
            with conn.cursor() as cursor:
                cursor.execute(f"SET lock_timeout = '{int(lock_timeout)}ms'")
        yield conn
    finally:
        conn.close()


def wrap_sync_thread(
    fn: Callable[[T], None],
    *,
    connections: Any,
    db_alias: str,
    lock_timeout: int | None,
) -> Callable[[T], None]:
    """Return a thread wrapper that opens a fresh Django connection."""

    def wrapper(state: T) -> None:
        with _fresh_connection(connections, db_alias, lock_timeout):
            fn(state)

    return wrapper


def wrap_async_task(
    fn: Callable[[T], Coroutine[Any, Any, None]],
    *,
    connections: Any,
    db_alias: str,
    lock_timeout: int | None,
) -> Callable[[T], Coroutine[Any, Any, None]]:
    """Return a task wrapper that opens a fresh Django connection."""

    async def wrapper(state: T) -> None:
        with _fresh_connection(connections, db_alias, lock_timeout):
            await fn(state)

    return wrapper


def prepare_django_dpor(
    setup: Callable[[], Any],
    workers: list[Callable[..., Any]],
    wrap_worker: Callable[..., Any],
    *,
    db_alias: str,
    lock_timeout: int | None,
    trace_packages: list[str] | None,
) -> tuple[Callable[[], Any], list[Callable[..., Any]], list[str]]:
    """Wrap *setup* and each worker for Django connection management.

    Called by both ``django_dpor`` (sync) and ``async_django_dpor`` (async).
    Returns ``(wrapped_setup, wrapped_workers, resolved_trace_packages)``.

    Args:
        setup: User-supplied setup callable.
        workers: List of thread or task callables.
        wrap_worker: Either :func:`wrap_sync_thread` or :func:`wrap_async_task`.
        db_alias: Django database alias.
        lock_timeout: Optional per-connection lock timeout in milliseconds.
        trace_packages: Package patterns to trace; ``None`` resolves to
            :data:`DJANGO_TRACE_PACKAGES`.
    """
    from django.db import connections  # type: ignore[import-not-found]

    resolved_packages = list(DJANGO_TRACE_PACKAGES) if trace_packages is None else trace_packages
    wrapped_setup = wrap_setup(setup, connections.close_all)
    wrapped_workers = [
        wrap_worker(fn, connections=connections, db_alias=db_alias, lock_timeout=lock_timeout) for fn in workers
    ]
    return wrapped_setup, wrapped_workers, resolved_packages
