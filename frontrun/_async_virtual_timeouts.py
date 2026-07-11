"""Virtual-clock-aware patches for asyncio timeout / sleep APIs.

Under a virtual clock the async DPOR / random schedulers must turn wall-clock
waits into schedulable virtual deadlines, otherwise ``asyncio.sleep`` /
``asyncio.wait_for`` / ``asyncio.timeout`` would block on real time and the
autojump clock could never advance them.

This module owns the patched replacements:

- ``_cooperative_async_sleep`` — a positive ``asyncio.sleep`` becomes a timed
  block against the scheduler's virtual clock (``asyncio.sleep(0)`` stays a
  pure yield).
- ``_virtual_asyncio_wait_for`` / ``_virtual_timeout_context`` /
  ``_virtual_timeout_at_context`` — the timeout fires when the virtual clock
  reaches the deadline (registered via the scheduler's
  ``add_timeout_deadline``), not when wall time elapses.

The scheduler is reached through ``_scheduler_var`` and its methods via
``getattr`` so the same patches drive both the DPOR and replay schedulers.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from frontrun._async_autopause import _in_scheduler_pause, _scheduler_var, _task_id_var
from frontrun._async_cooperative import _real_asyncio_sleep
from frontrun.async_scheduler import _in_frontrun_timer

__all__ = [
    "_VirtualAsyncTimeoutContext",
    "_VirtualAsyncTimeoutToken",
    "_cooperative_async_sleep",
    "_patch_asyncio_sleep",
    "_patch_asyncio_timeouts",
    "_unpatch_asyncio_sleep",
    "_unpatch_asyncio_timeouts",
    "_virtual_asyncio_wait_for",
    "_virtual_timeout_at_context",
    "_virtual_timeout_context",
]

_real_asyncio_wait_for = asyncio.wait_for
_real_asyncio_timeout = getattr(asyncio, "timeout", None)
_real_asyncio_timeout_at = getattr(asyncio, "timeout_at", None)
_async_sleep_patched = False
_async_timeout_patched = False


class _VirtualAsyncTimeoutToken:
    """Deadline token whose ``fire()`` resolves its future and runs ``on_fire``.

    Registered with the scheduler's ``add_timeout_deadline``; the clock actor
    calls ``fire()`` when the virtual clock reaches the deadline.  ``on_fire``
    carries the primitive-specific side effect (for ``asyncio.timeout`` /
    ``timeout_at`` it cancels the guarded task) instead of shadowing the
    ``fire`` method with a per-instance closure.
    """

    def __init__(self, on_fire: Callable[[], None] | None = None) -> None:
        self.future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.expired = False
        self._on_fire = on_fire

    def __repr__(self) -> str:
        return "<virtual-async-timeout>"

    def fire(self) -> None:
        self.expired = True
        if not self.future.done():
            self.future.set_result(None)
        if self._on_fire is not None:
            self._on_fire()


async def _cooperative_async_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401
    """No-delay replacement for ``asyncio.sleep`` during exploration.

    Yields to the event loop (``await asyncio.sleep(0)``) so it remains
    a scheduling point, but skips the actual delay.

    When the active async scheduler owns a virtual clock, a positive sleep
    becomes a *timed block*: the task registers a virtual deadline and
    suspends until the scheduler advances the clock to it.
    ``asyncio.sleep(0)`` stays a pure yield, matching stock semantics.
    """
    scheduler = _scheduler_var.get()
    task_id = _task_id_var.get()
    if scheduler is not None and task_id is not None and delay and delay > 0:
        clock = scheduler.virtual_clock
        sleep_until = getattr(scheduler, "sleep_until", None)
        if clock is not None and sleep_until is not None:
            await sleep_until(task_id, clock.now() + delay)
            return result
    await _real_asyncio_sleep(0)
    return result


def _patch_asyncio_sleep() -> None:
    """Replace ``asyncio.sleep`` with a zero-delay version."""
    global _async_sleep_patched  # noqa: PLW0603
    if _async_sleep_patched:
        return
    asyncio.sleep = _cooperative_async_sleep  # type: ignore[assignment]
    _async_sleep_patched = True


def _unpatch_asyncio_sleep() -> None:
    """Restore original ``asyncio.sleep``."""
    global _async_sleep_patched  # noqa: PLW0603
    if not _async_sleep_patched:
        return
    asyncio.sleep = _real_asyncio_sleep  # type: ignore[assignment]
    _async_sleep_patched = False


def _virtual_timeout_impl(value: float | None, *, at: bool) -> Any:
    """Shared body of ``_virtual_timeout_context`` / ``_virtual_timeout_at_context``.

    *at* selects ``asyncio.timeout_at`` semantics (*value* is an absolute loop
    time) versus ``asyncio.timeout`` (*value* is a relative delay).
    """
    scheduler = _scheduler_var.get()
    task_id = _task_id_var.get()
    clock = scheduler.virtual_clock if scheduler is not None else None
    real = _real_asyncio_timeout_at if at else _real_asyncio_timeout
    if _in_frontrun_timer.get() or scheduler is None or task_id is None or clock is None or real is None:
        if real is None:
            name = "timeout_at" if at else "timeout"
            raise RuntimeError(f"asyncio.{name} is not available on this Python version")
        return real(value)
    if value is None:
        return _VirtualAsyncTimeoutContext(scheduler, task_id, None, deadline=None)
    loop_now = asyncio.get_running_loop().time()
    if at:
        when = value
        remaining = value - loop_now
    else:
        when = loop_now + value
        remaining = value
    deadline = clock.now() + max(0.0, remaining)
    return _VirtualAsyncTimeoutContext(scheduler, task_id, when, deadline=deadline)


def _virtual_timeout_context(delay: float | None) -> Any:
    return _virtual_timeout_impl(delay, at=False)


def _virtual_timeout_at_context(when: float | None) -> Any:
    return _virtual_timeout_impl(when, at=True)


class _VirtualAsyncTimeoutContext:
    def __init__(self, scheduler: Any, task_id: int, when: float | None, *, deadline: float | None) -> None:
        self._scheduler = scheduler
        self._task_id = task_id
        self._initial_when = when
        self._initial_deadline = deadline
        self._when = when
        self._token: _VirtualAsyncTimeoutToken | None = None
        self._task: asyncio.Task[Any] | None = None
        self._cancelling = 0
        self._immediate_handle: asyncio.Handle | None = None
        self._entered = False
        self._exited = False

    async def __aenter__(self) -> _VirtualAsyncTimeoutContext:
        if self._entered:
            raise RuntimeError("Timeout has already been entered")
        self._entered = True
        self._task = asyncio.current_task()
        if self._task is not None:
            self._cancelling = self._task.cancelling()
        self._reschedule(self._initial_when, deadline=self._initial_deadline)
        return self

    def when(self) -> float | None:
        return self._when

    def expired(self) -> bool:
        return self._token.expired if self._token is not None else False

    def _to_virtual_deadline(self, when: float | None) -> float | None:
        if when is None:
            return None
        clock = self._scheduler.virtual_clock
        if clock is None:
            return None
        loop_now = asyncio.get_running_loop().time()
        return clock.now() + max(0.0, when - loop_now)

    def reschedule(self, when: float | None) -> None:
        self._reschedule(when, translate=True)

    def _reschedule(self, when: float | None, *, deadline: float | None = None, translate: bool = False) -> None:
        if not self._entered:
            raise RuntimeError("Timeout has not been entered")
        if self._exited:
            raise RuntimeError("Timeout has already exited")
        if self.expired():
            raise RuntimeError("Timeout has already expired")
        if self._immediate_handle is not None:
            self._immediate_handle.cancel()
            self._immediate_handle = None
        if self._token is not None:
            self._scheduler.remove_timeout_deadline(self._task_id, self._token)
        self._token = None
        self._when = when
        if translate:
            deadline = self._to_virtual_deadline(when)
        if deadline is None or self._task is None:
            return

        def _on_fire() -> None:
            if self._task is not None and not self._task.done():
                self._task.cancel()

        token = _VirtualAsyncTimeoutToken(on_fire=_on_fire)
        self._token = token
        clock = self._scheduler.virtual_clock
        if clock is not None and deadline <= clock.now():
            self._immediate_handle = asyncio.get_running_loop().call_soon(token.fire)
            return
        self._scheduler.add_timeout_deadline(self._task_id, deadline, token)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            token = self._token
            expired = bool(token is not None and token.expired)
            if self._immediate_handle is not None:
                self._immediate_handle.cancel()
                self._immediate_handle = None
            if token is not None:
                self._scheduler.remove_timeout_deadline(self._task_id, token)
            if expired:
                uncancel = getattr(self._task, "uncancel", None)
                remaining_cancels = uncancel() if uncancel is not None else self._cancelling
                if self._scheduler._error is None:
                    await self._scheduler.wait_until_scheduled_after_block(self._task_id, "asyncio.timeout")
                if exc_type is asyncio.CancelledError and (uncancel is None or remaining_cancels <= self._cancelling):
                    raise TimeoutError from exc
            return False
        finally:
            self._exited = True


async def _virtual_asyncio_wait_for(awaitable: Awaitable[Any], timeout: float | None) -> Any:
    scheduler = _scheduler_var.get()
    task_id = _task_id_var.get()
    clock = scheduler.virtual_clock if scheduler is not None else None
    if timeout is None or _in_frontrun_timer.get() or scheduler is None or task_id is None or clock is None:
        # A scheduler with a virtual clock always provides the timeout-deadline
        # methods, so ``clock is None`` is the only gate needed here.
        return await _real_asyncio_wait_for(awaitable, timeout)

    if timeout <= 0:
        inner = asyncio.ensure_future(awaitable)
        if inner.done():
            return await inner
        inner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await inner
        raise TimeoutError

    inner_awaitable: Awaitable[Any] = awaitable
    wraps_logical_task = asyncio.iscoroutine(awaitable)
    if wraps_logical_task:
        from frontrun._async_autopause import _AutoPauseCoroutine

        inner_awaitable = _AutoPauseCoroutine(awaitable, task_id, scheduler)
    inner = asyncio.ensure_future(inner_awaitable)

    event_blocked = scheduler._event_blocked
    engine_execution = None if wraps_logical_task else getattr(scheduler, "execution", None)
    # A bare-future wait under an engineless scheduler (the random strategy's
    # AwaitScheduler) has no engine bookkeeping to hide the parked task from
    # the schedule, so register it as a timed park: the schedule skips it like
    # a sleeper and the clock advance is what wakes it.  Without this the
    # schedule head stalls on the parked task until the wall-clock watchdog
    # cancels a *different* task mid-suspension (false counterexample).
    parked = engine_execution is None and not wraps_logical_task
    token = _VirtualAsyncTimeoutToken(
        on_fire=(lambda: scheduler.unpark_timed_wait(task_id)) if parked else None,
    )
    scheduler.add_timeout_deadline(task_id, clock.now() + timeout, token)
    if engine_execution is not None and event_blocked is not None:
        event_blocked.add(task_id)
    if engine_execution is not None:
        engine_execution.block_thread(task_id)
    if parked:
        scheduler.park_timed_wait(task_id)
    depth = _in_scheduler_pause.get()
    _in_scheduler_pause.set(depth + 1)
    unblocked = False
    try:
        if engine_execution is not None or parked:
            await scheduler.kick_stalled_schedule(task_id)
        done, _pending = await asyncio.wait({inner, token.future}, return_when=asyncio.FIRST_COMPLETED)
        if engine_execution is not None:
            engine_execution.unblock_thread(task_id)
            unblocked = True
        if engine_execution is not None and event_blocked is not None:
            event_blocked.discard(task_id)
        if engine_execution is not None and scheduler._error is None:
            await scheduler.wait_until_scheduled_after_block(task_id, "virtual wait_for")
        if inner in done and token.future not in done:
            return await inner
        if token.future in done:
            if engine_execution is None and scheduler._error is None:
                await scheduler.wait_until_scheduled_after_block(task_id, "virtual wait_for timeout")
            if not inner.done():
                inner.cancel()
            try:
                return await inner
            except asyncio.CancelledError as exc:
                raise TimeoutError from exc
        return await inner
    finally:
        _in_scheduler_pause.set(depth)
        scheduler.remove_timeout_deadline(task_id, token)
        if parked:
            scheduler.unpark_timed_wait(task_id)
        if engine_execution is not None and event_blocked is not None:
            event_blocked.discard(task_id)
        if engine_execution is not None and not unblocked:
            engine_execution.unblock_thread(task_id)
        if not inner.done():
            inner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inner


def _patch_asyncio_timeouts() -> None:
    global _async_timeout_patched  # noqa: PLW0603
    if _async_timeout_patched:
        return
    asyncio.wait_for = _virtual_asyncio_wait_for  # type: ignore[assignment]
    if _real_asyncio_timeout is not None:
        asyncio.timeout = _virtual_timeout_context  # type: ignore[assignment]
    if _real_asyncio_timeout_at is not None:
        asyncio.timeout_at = _virtual_timeout_at_context  # type: ignore[assignment]
    _async_timeout_patched = True


def _unpatch_asyncio_timeouts() -> None:
    global _async_timeout_patched  # noqa: PLW0603
    if not _async_timeout_patched:
        return
    asyncio.wait_for = _real_asyncio_wait_for  # type: ignore[assignment]
    if _real_asyncio_timeout is not None:
        asyncio.timeout = _real_asyncio_timeout  # type: ignore[assignment]
    if _real_asyncio_timeout_at is not None:
        asyncio.timeout_at = _real_asyncio_timeout_at  # type: ignore[assignment]
    _async_timeout_patched = False
