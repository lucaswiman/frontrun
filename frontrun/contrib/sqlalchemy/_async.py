"""Async SQLAlchemy helper for async DPOR integration testing.

Handles per-task async connection management and optional lock_timeout
injection automatically.

Prefer the unified ``sqlalchemy_dpor`` dispatcher (pass ``tasks=`` for async)::

    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_async_connection

    result = await sqlalchemy_dpor(
        engine=async_engine,
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
        lock_timeout=500,
    )
    assert result.property_holds, result.explanation

Inside a task function, retrieve the per-task connection with::

    conn = get_async_connection()
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from frontrun.common import InterleavingResult
from frontrun.contrib.sqlalchemy._shared import (
    make_current_connection_var,
    wrap_async_setup,
    wrap_async_task,
)

T = TypeVar("T")

_current_connection: contextvars.ContextVar[Any] = make_current_connection_var("_async_sa_connection")


def get_async_connection() -> Any:
    """Return the per-task async SQLAlchemy connection set by ``async_sqlalchemy_dpor``."""
    return _current_connection.get()


async def async_sqlalchemy_dpor(
    engine: Any,
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    *,
    lock_timeout: int | None = None,
    detect_sql: bool = True,
    **kwargs: Any,
) -> InterleavingResult:
    """Run async DPOR exploration with per-task async SQLAlchemy connection management.

    Args:
        engine: An async SQLAlchemy ``AsyncEngine`` instance.
        setup: Called once per execution to create fresh shared state.
        tasks: List of async callables, each receiving the shared state.
            Each task gets its own async connection.
            Use :func:`get_async_connection` inside the task to access it.
        invariant: Predicate over shared state after all tasks complete.
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            task's connection.
        detect_sql: Passed through to the async DPOR engine (default True).
        **kwargs: Forwarded verbatim to :func:`frontrun.explore`.
    """
    from frontrun.async_dpor import _explore_async_dpor

    wrapped_setup = wrap_async_setup(engine, setup)
    wrapped_tasks = [wrap_async_task(engine, _current_connection, lock_timeout, fn) for fn in tasks]

    return await _explore_async_dpor(
        setup=wrapped_setup,
        tasks=wrapped_tasks,
        invariant=invariant,
        detect_sql=detect_sql,
        lock_timeout=lock_timeout,
        **kwargs,
    )
