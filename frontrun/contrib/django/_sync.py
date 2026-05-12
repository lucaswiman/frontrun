"""Django helper for DPOR integration testing.

Handles per-thread Django database connection management and optional
lock_timeout injection automatically.

Example::

    from frontrun.contrib.django import django_dpor

    result = django_dpor(
        setup=_State,
        threads=[thread_a, thread_b],
        invariant=_invariant,
        lock_timeout=500,  # optional, milliseconds
    )
    assert result.property_holds, result.explanation
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from frontrun.contrib.django._shared import prepare_django_dpor, wrap_sync_thread

T = TypeVar("T")


def django_dpor(
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    *,
    db_alias: str = "default",
    lock_timeout: int | None = None,
    detect_io: bool = True,
    trace_packages: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Run DPOR exploration with per-thread Django connection management.

    Each thread closes the shared Django connection and opens a fresh one,
    ensuring DPOR sees independent connections per thread.

    By default, ``trace_packages`` is set to :data:`DJANGO_TRACE_PACKAGES`
    so that code inside ``django_*`` apps and ``django.contrib.sites``
    submodules is traced.  Pass an explicit list (or ``[]``) to override.

    Args:
        setup: Called once per execution to create fresh shared state.
            All Django DB connections are closed before this runs.
        threads: List of callables, each receiving the shared state.
        invariant: Predicate over shared state after all threads complete.
        db_alias: Django database alias to use (default ``"default"``).
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            thread's connection before running the thread.
        detect_io: Passed through to the DPOR engine (default True).
        trace_packages: Package name patterns (fnmatch syntax) to trace.
            Defaults to :data:`DJANGO_TRACE_PACKAGES`.  Pass ``[]`` to
            disable extra tracing beyond user code.
        **kwargs: Forwarded verbatim to :func:`frontrun.explore`.
    """
    from frontrun._dpor_runtime.explore import _explore_dpor

    wrapped_setup, wrapped_threads, resolved_packages = prepare_django_dpor(
        setup, threads, wrap_sync_thread, db_alias=db_alias, lock_timeout=lock_timeout, trace_packages=trace_packages
    )
    return _explore_dpor(
        setup=wrapped_setup,
        threads=wrapped_threads,
        invariant=invariant,
        detect_io=detect_io,
        lock_timeout=lock_timeout,
        trace_packages=resolved_packages,
        **kwargs,
    )
