"""Shared auto-pause machinery for async concurrency exploration."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Generator, Mapping
from typing import Any, cast

_scheduler_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar("_scheduler", default=None)
_task_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("_task_id", default=None)
_auto_pause_active: contextvars.ContextVar[bool] = contextvars.ContextVar("_auto_pause_active", default=False)
_in_scheduler_pause: contextvars.ContextVar[int] = contextvars.ContextVar("_in_scheduler_pause", default=0)


def _notify_task_yielded(scheduler: Any, task_id: int) -> None:
    if _in_scheduler_pause.get() > 0:
        return
    on_task_yielded = getattr(scheduler, "on_task_yielded", None)
    if on_task_yielded is not None:
        on_task_yielded(task_id)


async def await_point() -> None:
    """Yield to the active async scheduler, or return immediately if none exists."""
    if _auto_pause_active.get():
        await asyncio.sleep(0)
        return
    scheduler = _scheduler_var.get()
    if scheduler is not None:
        task_id = _task_id_var.get()
        if task_id is not None:
            await scheduler.pause(task_id)


class _AutoPauseIterator:
    """Wrap a coroutine so every natural await can become a scheduling boundary."""

    __slots__ = ("_inner", "_task_id", "_scheduler", "_pause_iter", "_buffered_value")

    def __init__(self, inner_coro: Any, task_id: int, scheduler: Any) -> None:
        self._inner = inner_coro
        self._task_id = task_id
        self._scheduler = scheduler
        self._pause_iter: Any | None = None
        self._buffered_value: Any = None

    def __next__(self) -> Any:
        return self.send(None)

    def __iter__(self) -> _AutoPauseIterator:
        return self

    def _resume_inner(self) -> Any:
        """The pause coroutine finished: clear it, resume the inner coroutine
        with the buffered value, and report the await-point yield."""
        self._pause_iter = None
        yielded = self._inner.send(self._buffered_value)
        _notify_task_yielded(self._scheduler, self._task_id)
        return yielded

    def send(self, value: Any) -> Any:
        if self._pause_iter is not None:
            try:
                return self._pause_iter.send(value)
            except StopIteration:
                return self._resume_inner()

        if _in_scheduler_pause.get() > 0:
            return self._inner.send(value)

        self._buffered_value = value
        pause_coro = self._scheduler.pause(self._task_id)
        self._pause_iter = pause_coro.__await__()
        try:
            return next(cast(Generator[Any, Any, Any], self._pause_iter))
        except StopIteration:
            return self._resume_inner()

    def _close_pause_iter(self) -> None:
        """Close the suspended pause coroutine, tolerating cancellation noise.

        When the pause coroutine is suspended inside
        ``asyncio.wait_for(condition.wait())``, ``GeneratorExit`` cannot always
        unwind cleanly (the inner ``wait_for`` task may still be pending),
        which CPython surfaces as ``RuntimeError("coroutine ignored
        GeneratorExit")``.  That is benign here — the surrounding task is being
        thrown into / closed anyway — so swallow it instead of leaking it out
        of ``throw``/``close`` (finding F9).
        """
        if self._pause_iter is not None:
            try:
                self._pause_iter.close()
            except RuntimeError:
                pass
            self._pause_iter = None

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        self._close_pause_iter()
        if val is None and tb is None:
            return self._inner.throw(typ)
        return self._inner.throw(typ, val, tb)

    def close(self) -> None:
        self._close_pause_iter()
        self._inner.close()


class _AutoPauseCoroutine:
    """Awaitable wrapper that auto-schedules a coroutine at each await."""

    __slots__ = ("_iter",)

    def __init__(self, coro: Any, task_id: int, scheduler: Any) -> None:
        self._iter = _AutoPauseIterator(coro, task_id, scheduler)

    def __await__(self) -> Generator[Any, Any, None]:
        return self._iter  # type: ignore[return-value]


def wrap_auto_paused_tasks(
    task_funcs: Mapping[Any, Callable[..., Awaitable[None]]],
    scheduler: Any,
) -> dict[Any, Callable[..., Awaitable[None]]]:
    """Wrap task callables so every natural await becomes a scheduling point."""
    wrapped: dict[Any, Callable[..., Awaitable[None]]] = {}
    for task_id, func in task_funcs.items():

        async def _wrapped(f: Callable[..., Awaitable[None]] = func, t: Any = task_id) -> None:
            _auto_pause_active.set(True)
            await _AutoPauseCoroutine(f(), t, scheduler)

        wrapped[task_id] = _wrapped
    return wrapped


__all__ = [
    "_scheduler_var",
    "_task_id_var",
    "_auto_pause_active",
    "_in_scheduler_pause",
    "await_point",
    "_AutoPauseIterator",
    "_AutoPauseCoroutine",
    "wrap_auto_paused_tasks",
]
