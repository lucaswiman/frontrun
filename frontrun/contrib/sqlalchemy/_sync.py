"""SQLAlchemy helper for DPOR integration testing.

Handles per-thread connection management and optional lock_timeout injection
automatically.

Example::

    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_connection

    result = sqlalchemy_dpor(
        engine=engine,
        setup=_State,
        threads=[thread_a, thread_b],
        invariant=_invariant,
        lock_timeout=500,  # optional, milliseconds
    )
    assert result.property_holds, result.explanation

Inside a thread function, retrieve the per-thread connection with::

    conn = get_connection()
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any, TypeVar

from frontrun.contrib.sqlalchemy._shared import make_current_connection_var, wrap_sync_setup, wrap_sync_thread

T = TypeVar("T")

_current_connection: contextvars.ContextVar[Any] = make_current_connection_var("_current_connection")


def get_connection() -> Any:
    """Return the per-thread SQLAlchemy connection set by ``sqlalchemy_dpor``."""
    return _current_connection.get()


def sqlalchemy_dpor(
    engine: Any,
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    *,
    lock_timeout: int | None = None,
    detect_io: bool = True,
    **kwargs: Any,
) -> Any:
    """Run DPOR exploration with per-thread SQLAlchemy connection management.

    Args:
        engine: A SQLAlchemy ``Engine`` instance.
        setup: Called once per execution to create fresh shared state.
            Any open connections on the engine are disposed before this runs.
        threads: List of callables, each receiving the shared state.
            Each thread gets its own connection scoped to its execution.
            Use :func:`get_connection` inside the thread to access it.
        invariant: Predicate over shared state after all threads complete.
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            thread's connection before running the thread.  Converts
            C-level row-lock blocking into a fast PostgreSQL error.
        detect_io: Passed through to the DPOR engine (default True).
        **kwargs: Forwarded verbatim to :func:`frontrun.explore`.
    """
    from frontrun._dpor_runtime.explore import _explore_dpor

    wrapped_setup = wrap_sync_setup(engine, setup)
    wrapped_threads = [wrap_sync_thread(engine, _current_connection, lock_timeout, fn) for fn in threads]

    return _explore_dpor(
        setup=wrapped_setup,
        threads=wrapped_threads,
        invariant=invariant,
        detect_io=detect_io,
        lock_timeout=lock_timeout,
        **kwargs,
    )
