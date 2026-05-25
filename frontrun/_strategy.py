"""Strategy interface used by :func:`frontrun.explore`.

Each exploration approach is wrapped in a small adapter that conforms to the
:class:`Strategy` / :class:`AsyncStrategy` protocols below.  The dispatcher
in :mod:`frontrun.explore` looks adapters up in :data:`STRATEGIES` /
:data:`ASYNC_STRATEGIES`; adding a new approach is a matter of registering
one more adapter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from frontrun.common import InterleavingResult


def _expand_async_io_kwargs(kwargs: dict[str, Any]) -> tuple[bool, bool]:
    """Pop ``detect_io`` / ``detect_sql`` and return ``(detect_sql, detect_io)``.

    The unified ``explore()`` API exposes a single ``detect_io`` flag, but the
    async implementations split it into ``detect_sql`` (+ ``detect_redis`` for
    DPOR).  Both async adapters use this; sync adapters pass ``detect_io``
    through unchanged.
    """
    detect_io = kwargs.pop("detect_io", True)
    detect_sql_explicit = kwargs.pop("detect_sql", False)
    return detect_sql_explicit or detect_io, detect_io


@runtime_checkable
class Strategy(Protocol):
    """Synchronous exploration strategy."""

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult: ...


@runtime_checkable
class AsyncStrategy(Protocol):
    """Asynchronous exploration strategy.  :meth:`run` returns an awaitable."""

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> Awaitable[InterleavingResult]: ...


# ---------------------------------------------------------------------------
# Sync adapters
# ---------------------------------------------------------------------------


_DPOR_SYNC_KEYS: frozenset[str] = frozenset(
    {
        "max_executions",
        "preemption_bound",
        "max_branches",
        "timeout_per_run",
        "stop_on_first",
        "detect_io",
        "deadlock_timeout",
        "reproduce_on_failure",
        "total_timeout",
        "warn_nondeterministic_sql",
        "lock_timeout",
        "trace_packages",
        "track_dunder_dict_accesses",
        "search",
        "patch_sleep",
        "serializable_invariant",
        "error_on_any_race",
    }
)

_NONE_MEANINGFUL_KEYS: frozenset[str] = frozenset(
    {"preemption_bound", "max_executions", "total_timeout", "lock_timeout", "trace_packages", "search"}
)

_RANDOM_SYNC_KEYS: frozenset[str] = frozenset(
    {
        "max_attempts",
        "max_ops",
        "timeout_per_run",
        "seed",
        "debug",
        "detect_io",
        "deadlock_timeout",
        "reproduce_on_failure",
        "total_timeout",
        "warn_nondeterministic_sql",
        "trace_packages",
        "patch_sleep",
        "serializable_invariant",
        "error_on_any_race",
    }
)


class _SyncDporStrategy:
    """Adapter routing sync DPOR through ``_dpor_runtime.explore._explore_dpor``."""

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun._dpor_runtime.explore import _explore_dpor

        return _explore_dpor(
            setup=setup,
            threads=workers,
            invariant=invariant,
            **{
                k: v
                for k, v in kwargs.items()
                if k in _DPOR_SYNC_KEYS and (v is not None or k in _NONE_MEANINGFUL_KEYS)
            },  # pyright: ignore[reportArgumentType]
        )


class _SyncRandomStrategy:
    """Adapter routing sync random exploration through ``bytecode.explore_random``."""

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.bytecode import explore_random as _explore_random

        return _explore_random(
            setup=setup,
            threads=workers,
            invariant=invariant,
            **{
                k: v
                for k, v in kwargs.items()
                if k in _RANDOM_SYNC_KEYS and (v is not None or k in _NONE_MEANINGFUL_KEYS)
            },  # pyright: ignore[reportArgumentType]
        )


# ---------------------------------------------------------------------------
# Async adapters
# ---------------------------------------------------------------------------


_DPOR_ASYNC_KEYS: frozenset[str] = frozenset(
    {
        "max_executions",
        "preemption_bound",
        "max_branches",
        "timeout_per_run",
        "stop_on_first",
        "deadlock_timeout",
        "reproduce_on_failure",
        "total_timeout",
        "warn_nondeterministic_sql",
        "lock_timeout",
        "trace_packages",
        "patch_sleep",
        "serializable_invariant",
        "error_on_any_race",
    }
)

_RANDOM_ASYNC_KEYS: frozenset[str] = frozenset(
    {
        "max_attempts",
        "max_ops",
        "timeout_per_run",
        "seed",
        "deadlock_timeout",
        "trace_packages",
        "patch_sleep",
        "serializable_invariant",
        "error_on_any_race",
        "total_timeout",
    }
)


class _AsyncDporStrategy:
    """Adapter routing async DPOR through ``async_dpor._explore_async_dpor``."""

    async def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.async_dpor import _explore_async_dpor

        detect_sql, detect_io = _expand_async_io_kwargs(kwargs)
        return await _explore_async_dpor(
            setup=setup,
            tasks=workers,
            invariant=invariant,
            detect_sql=detect_sql,
            detect_redis=detect_io,
            **{
                k: v
                for k, v in kwargs.items()
                if k in _DPOR_ASYNC_KEYS and (v is not None or k in _NONE_MEANINGFUL_KEYS)
            },  # pyright: ignore[reportArgumentType]
        )


class _AsyncRandomStrategy:
    """Adapter routing async random through ``async_shuffler.explore_async_random``."""

    async def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.async_shuffler import explore_async_random as _explore_async_random

        detect_sql, _ = _expand_async_io_kwargs(kwargs)
        return await _explore_async_random(
            setup=setup,
            tasks=workers,
            invariant=invariant,
            detect_sql=detect_sql,
            **{
                k: v
                for k, v in kwargs.items()
                if k in _RANDOM_ASYNC_KEYS and (v is not None or k in _NONE_MEANINGFUL_KEYS)
            },  # pyright: ignore[reportArgumentType]
        )


# ---------------------------------------------------------------------------
# Registries — keyed by the user-facing ``strategy=`` value
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, Strategy] = {
    "dpor": _SyncDporStrategy(),
    "random": _SyncRandomStrategy(),
}

ASYNC_STRATEGIES: dict[str, AsyncStrategy] = {
    "dpor": _AsyncDporStrategy(),
    "random": _AsyncRandomStrategy(),
}


__all__ = [
    "ASYNC_STRATEGIES",
    "STRATEGIES",
    "AsyncStrategy",
    "Strategy",
]
