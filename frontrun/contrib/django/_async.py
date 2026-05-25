"""Async Django helper for async DPOR integration testing.

Handles per-task Django async database connection management and optional
lock_timeout injection automatically.

Prefer the unified ``django_dpor`` dispatcher (pass ``tasks=`` for async)::

    from frontrun.contrib.django import django_dpor

    result = await django_dpor(
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
        lock_timeout=500,
    )
    assert result.property_holds, result.explanation
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from frontrun.common import InterleavingResult
from frontrun.contrib.django._shared import prepare_django_dpor, wrap_async_task

T = TypeVar("T")


async def async_django_dpor(
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    *,
    db_alias: str = "default",
    lock_timeout: int | None = None,
    detect_sql: bool = True,
    trace_packages: list[str] | None = None,
    **kwargs: Any,
) -> InterleavingResult:
    """Run async DPOR exploration with per-task Django async connection management.

    Each task closes the shared Django connection and opens a fresh one.

    By default, ``trace_packages`` is set to :data:`~frontrun.contrib.django._sync.DJANGO_TRACE_PACKAGES`
    so that code inside ``django_*`` apps and ``django.contrib.sites``
    submodules is traced.  Pass an explicit list (or ``[]``) to override.

    Args:
        setup: Called once per execution to create fresh shared state.
        tasks: List of async callables, each receiving the shared state.
        invariant: Predicate over shared state after all tasks complete.
        db_alias: Django database alias to use (default ``"default"``).
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            task's connection.
        detect_sql: Passed through to the async DPOR engine (default True).
        trace_packages: Package name patterns (fnmatch syntax) to trace.
            Defaults to :data:`~frontrun.contrib.django._sync.DJANGO_TRACE_PACKAGES`.
            Pass ``[]`` to disable extra tracing beyond user code.
        **kwargs: Forwarded verbatim to :func:`frontrun.explore`.
    """
    from frontrun.async_dpor import _explore_async_dpor

    wrapped_setup, wrapped_tasks, resolved_packages = prepare_django_dpor(
        setup, tasks, wrap_async_task, db_alias=db_alias, lock_timeout=lock_timeout, trace_packages=trace_packages
    )
    return await _explore_async_dpor(
        setup=wrapped_setup,
        tasks=wrapped_tasks,
        invariant=invariant,
        detect_sql=detect_sql,
        trace_packages=resolved_packages,
        lock_timeout=lock_timeout,
        **kwargs,
    )
