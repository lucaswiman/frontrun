"""
Async event loop abstraction for deterministic task interleaving.

This module provides InterleavedLoop, the shared foundation for all async
frontrun POCs. It wraps asyncio's cooperative scheduling to give
deterministic control over which task resumes at each yield point.

In async Python, the event loop decides which ready task to resume after
each await point. InterleavedLoop intercepts this decision, using a
pluggable scheduling policy to control the execution order.

Key insight: async code is single-threaded and cooperative. Context switches
happen ONLY at await points. InterleavedLoop exploits this by gating each
yield point through an asyncio.Condition — tasks wait until the scheduling
policy says it's their turn.

Both async approaches build on this abstraction:
- async_trace_markers (comment annotations): marker-based scheduling
- async_shuffler (property-based): index-based scheduling

Each POC subclasses InterleavedLoop and implements two methods:
- should_proceed(task_id, marker): return True when a task should resume
- on_proceed(task_id, marker): update internal scheduling state

Example — a simple round-robin scheduler:

    >>> class RoundRobinLoop(InterleavedLoop):
    ...     def __init__(self, order):
    ...         super().__init__()
    ...         self._order = order
    ...         self._step = 0
    ...
    ...     def should_proceed(self, task_id, marker=None):
    ...         if self._step >= len(self._order):
    ...             return True
    ...         return self._order[self._step] == task_id
    ...
    ...     def on_proceed(self, task_id, marker=None):
    ...         self._step += 1
"""

import asyncio
import contextlib
import contextvars
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from frontrun._async_autopause import _in_scheduler_pause
from frontrun._virtual_clock import real_monotonic

# Real asyncio.sleep captured before any patching, for the shared pause() yield.
# async_scheduler loads before explore() patches asyncio.sleep, so this is the
# genuine original (same object _async_cooperative captures independently).
_real_asyncio_sleep = asyncio.sleep

__all__ = [
    "InterleavedLoop",
    "SchedulerTimeoutError",
    "_AsyncSchedulerBase",
    "_in_frontrun_timer",
    "_install_frontrun_timer_tagging",
    "_patch_loop_instance_attr",
    "_pin_loop_time",
    "frontrun_wait_for",
]

_T = TypeVar("_T")

# Sentinel for "attribute was absent" when temporarily overriding a loop's
# instance attributes (call_at / call_later / time) and restoring them.
_MISSING = object()

# True while a frontrun-internal timed wait is creating its loop timer.
# Exact async deadlock detection must tell the scheduler's own watchdog
# timers apart from user timers (a pending user timer means a parked task
# may still be woken, so it is not a proven deadlock); the async DPOR
# driver tags timers created under this flag via a loop.call_at wrapper.
_in_frontrun_timer: contextvars.ContextVar[bool] = contextvars.ContextVar("frontrun_in_frontrun_timer", default=False)


class SchedulerTimeoutError(TimeoutError):
    """Timeout raised by frontrun's scheduler machinery, not user code."""


class WorkerCancelledError(asyncio.CancelledError, RuntimeError):
    """A worker cancelled itself rather than scheduler cleanup cancelling it.

    The dual inheritance preserves the public ``CancelledError`` contract for
    exact replay while letting exploration's ordinary ``except Exception``
    crash-classification path turn cancellation into a counterexample.
    """


async def frontrun_wait_for(awaitable: Coroutine[Any, Any, _T] | Awaitable[_T], timeout: float) -> _T:
    """``asyncio.wait_for`` whose timeout timer is tagged as frontrun-internal."""
    token = _in_frontrun_timer.set(True)
    try:
        return await asyncio.wait_for(awaitable, timeout)
    finally:
        _in_frontrun_timer.reset(token)


def _patch_loop_instance_attr(loop: Any, name: str, value: Any) -> Callable[[], None]:
    """Temporarily override ``loop.<name>``; returns a restore callback."""
    previous = getattr(loop, "__dict__", {}).get(name, _MISSING)
    setattr(loop, name, value)

    def restore() -> None:
        if previous is _MISSING:
            with contextlib.suppress(AttributeError):
                delattr(loop, name)
        else:
            setattr(loop, name, previous)

    return restore


def _pin_loop_time(loop: Any) -> Callable[[], None]:
    """Pin the loop's own clock to real monotonic time; returns a restore callback."""
    return _patch_loop_instance_attr(loop, "time", real_monotonic)


def _install_frontrun_timer_tagging(loop: Any) -> tuple[Callable[[], bool], Callable[[], None]]:
    """Wrap ``loop.call_at`` so frontrun's own watchdog timers are tagged.

    Exact deadlock detection may only fire when every pending loop timer is
    one of frontrun's own (a pending *user* timer — e.g. a wall-clock
    ``asyncio.wait_for`` — may still wake a parked task, so the state is not
    a proven deadlock).  Timers created while ``_in_frontrun_timer`` is set
    (see ``frontrun_wait_for``) are collected in a WeakSet; everything else
    counts as a user timer.

    Returns ``(user_timers_pending, uninstall)``.  ``user_timers_pending``
    is conservative: if the loop's timer heap cannot be inspected (a
    non-standard loop without ``_scheduled``), it reports True so exact
    detection stays off and the wall-clock fallback applies.
    """
    tagged: weakref.WeakSet[Any] = weakref.WeakSet()
    orig_call_at = loop.call_at
    orig_call_later = loop.call_later

    def _tagging_call_at(when: float, callback: Any, *args: Any, context: Any = None) -> Any:
        handle = orig_call_at(when, callback, *args, context=context)
        if _in_frontrun_timer.get():
            tagged.add(handle)
        return handle

    def _tagging_call_later(delay: float, callback: Any, *args: Any, context: Any = None) -> Any:
        handle = orig_call_later(delay, callback, *args, context=context)
        if _in_frontrun_timer.get():
            tagged.add(handle)
        return handle

    restore_call_at = _patch_loop_instance_attr(loop, "call_at", _tagging_call_at)
    restore_call_later = _patch_loop_instance_attr(loop, "call_later", _tagging_call_later)

    def _user_timers_pending() -> bool:
        scheduled = getattr(loop, "_scheduled", None)
        ready = getattr(loop, "_ready", None)
        if scheduled is None:
            return True
        if any(not handle.cancelled() and handle not in tagged for handle in scheduled):
            return True
        if ready is None:
            return True
        return any(not handle.cancelled() for handle in ready)

    def _uninstall() -> None:
        restore_call_later()
        restore_call_at()

    return _user_timers_pending, _uninstall


class InterleavedLoop:
    """Wrapped event loop for deterministic async task interleaving.

    This class controls which async task resumes at each yield point.
    Tasks call ``await loop.pause(task_id)`` at points where a context
    switch could happen, and the loop's scheduling policy decides
    whether the task should proceed or wait.

    Subclasses must implement:
        should_proceed(task_id, marker): Is it this task's turn?
        on_proceed(task_id, marker): Update state after a task proceeds.

    The base class provides:
        pause(): Yield point that gates on the scheduling policy
        run_all(): Run tasks with controlled interleaving
        Error propagation, timeout handling, and done-task tracking
    """

    # ------------------------------------------------------------------
    # Async scheduler protocol defaults
    #
    # The cooperative primitives (_async_cooperative) and virtual-timeout
    # wrappers (_async_virtual_timeouts) drive whichever scheduler is active
    # (DPOR exploration, replay, or the random shuffler) through the hooks
    # below.  Defining no-op / None defaults here lets those call sites invoke
    # them unconditionally instead of probing with getattr; each concrete
    # scheduler overrides the subset it actually implements.
    # ------------------------------------------------------------------

    #: Active virtual clock, or None in real-time mode.
    virtual_clock: Any = None
    #: Tasks parked inside a cooperative primitive; None when unused.
    _event_blocked: "set[int] | None" = None
    #: Stable object-id registry, or None (fall back to id()).
    _stable_ids: Any = None

    async def kick_stalled_schedule(self, task_id: int) -> None:
        """Hand the turn onward after a task engine-blocked itself (no-op default)."""

    async def wait_until_scheduled_after_block(self, task_id: int, reason: str) -> None:
        """Wait for a physically-woken task to be scheduled again (no-op default)."""

    def report_task_sync(self, task_id: int, event_type: str, sync_id: int) -> None:
        """Report a happens-before sync edge to the engine (no-op default)."""

    def report_task_access(self, task_id: int, object_id: int, kind: str) -> None:
        """Report a memory / resource access to the engine (no-op default)."""

    def add_timeout_deadline(self, task_id: int, deadline: float, token: object) -> None:
        """Register a virtual timeout deadline (no-op default)."""

    def remove_timeout_deadline(self, task_id: int, token: object) -> None:
        """Cancel a virtual timeout deadline (no-op default)."""

    def park_timed_wait(self, task_id: int) -> None:
        """Register a task parked in a virtual timed wait with no engine
        bookkeeping (e.g. ``asyncio.wait_for`` on a bare future under the
        random strategy); no-op default for engine-backed schedulers."""

    def unpark_timed_wait(self, task_id: int) -> None:
        """Unregister a task from a virtual timed park (no-op default)."""

    def _advance_virtual_deadline_for_idle(self) -> bool:
        """Advance the virtual clock to the next pending deadline when the run is
        idle; returns True if it made progress (default: no virtual clock)."""
        return False

    def __init__(self, *, deadlock_timeout: float = 5.0):
        self._condition = asyncio.Condition()
        self._finished = False
        self._error: Exception | None = None
        self._tasks_done: set[Any] = set()
        self._num_tasks: int = 0  # set by run_all
        self._waiting_count: int = 0
        # Monotonic progress counter: bumped on every pause() call and every
        # task completion. Used by run_all's external-deadlock watchdog to
        # tell "tasks blocked on unmanaged awaitables" from "tasks running".
        self._progress: int = 0
        self.deadlock_timeout = deadlock_timeout

    # ------------------------------------------------------------------
    # Scheduling policy — override in subclasses
    # ------------------------------------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        """Return True if this task should resume now.

        Called while holding the condition lock. Must not await.

        Args:
            task_id: Identity of the calling task (str, int, etc.)
            marker: Optional context from the yield point (e.g. a marker
                    name, an (operation, phase) tuple, or None).
        """
        raise NotImplementedError

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        """Update scheduling state after a task is allowed to proceed.

        Called while holding the condition lock, immediately after
        should_proceed returned True. Must not await.

        Args:
            task_id: Identity of the task that is proceeding.
            marker: Same marker value passed to should_proceed.
        """

    # ------------------------------------------------------------------
    # Per-task context hooks — override if needed
    # ------------------------------------------------------------------

    def _setup_task_context(self, task_id: Any) -> None:
        """Called when a task starts, before running user code.

        Override to set context variables, thread-locals, etc.
        """

    def _cleanup_task_context(self, task_id: Any) -> None:
        """Called when a task finishes, after running user code.

        Override to clean up context set in _setup_task_context.
        """

    # ------------------------------------------------------------------
    # Yield point
    # ------------------------------------------------------------------

    async def pause(self, task_id: Any, marker: Any = None) -> None:
        """Yield point: block until the scheduling policy says to proceed.

        Tasks call this at every point where a context switch could happen.
        The call blocks (yields to the event loop) until should_proceed()
        returns True for this task, then calls on_proceed() and returns.

        Uses all-tasks-waiting detection: if every non-done task is blocked
        in ``pause()`` and none can proceed, deadlock is detected instantly.

        Args:
            task_id: Identity of the calling task.
            marker: Optional scheduling context.
        """
        self._progress += 1
        async with self._condition:
            # First check before entering the waiting state.
            if self._finished or self._error:
                return
            if self.should_proceed(task_id, marker):
                self.on_proceed(task_id, marker)
                self._condition.notify_all()
                return

            # Enter waiting state: increment _waiting_count ONCE and keep
            # it incremented for the entire time this task is blocked.
            # This avoids a false-positive deadlock when multiple tasks
            # wake from condition.wait() simultaneously: the first to
            # reacquire the lock would decrement-then-re-increment before
            # the second can decrement, making _waiting_count transiently
            # equal to alive even though the second task CAN proceed.
            alive = self._num_tasks - len(self._tasks_done)
            self._waiting_count += 1
            try:
                if self._waiting_count >= alive and alive > 0:
                    self._handle_all_waiting_deadlock(task_id, marker)
                    return

                while True:
                    try:
                        await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                    except asyncio.TimeoutError:
                        self._handle_timeout(task_id, marker)
                        return

                    if self._finished or self._error:
                        return
                    if self.should_proceed(task_id, marker):
                        self.on_proceed(task_id, marker)
                        self._condition.notify_all()
                        return
            finally:
                self._waiting_count -= 1

    def _on_error_set(self) -> None:
        """Hook run right after any abort path sets ``self._error`` (no-op default).

        The single point where a scheduler reacts to an abort.  The async DPOR
        scheduler overrides it to wake tasks parked in cooperative primitives so
        they free-run to completion instead of hanging until ``timeout_per_run``.
        """

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        """Handle a timeout in pause(). Sets the error and wakes everyone.

        Override to provide a more informative error message.
        """
        self._error = SchedulerTimeoutError(
            f"Deadlock: task {task_id!r} timed out waiting at marker {marker!r} (fallback timeout)"
        )
        self._on_error_set()
        self._condition.notify_all()

    def _handle_all_waiting_deadlock(self, task_id: Any, marker: Any = None) -> None:
        """Handle instant deadlock: all alive tasks are waiting, none can proceed.

        Override to provide a more informative error message.
        """
        alive = self._num_tasks - len(self._tasks_done)
        self._error = SchedulerTimeoutError(
            f"Deadlock: all {alive} alive tasks are waiting but none can proceed "
            f"(task {task_id!r} at marker {marker!r})"
        )
        self._on_error_set()
        self._condition.notify_all()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished and notify waiting tasks."""
        self._progress += 1
        async with self._condition:
            self._tasks_done.add(task_id)
            self._condition.notify_all()

    async def _report_error(self, error: Exception) -> None:
        """Report an error and wake all waiting tasks."""
        async with self._condition:
            if self._error is None:
                self._error = error
                self._on_error_set()
            self._condition.notify_all()

    async def run_all(
        self,
        task_funcs: dict[Any, Callable[..., Awaitable[None]]] | list[Callable[..., Awaitable[None]]],
        timeout: float = 10.0,
        *,
        detect_external_deadlock: bool = False,
    ) -> None:
        """Run tasks with controlled interleaving.

        Args:
            task_funcs: Either a dict ``{task_id: async_callable}`` or a
                list of async callables (which get integer task_ids
                0, 1, 2, ...).
            timeout: Maximum total time to wait for all tasks.
            detect_external_deadlock: Also detect deadlocks where every
                unfinished task is blocked on an *unmanaged* awaitable (e.g. a
                stock ``asyncio.Lock``) rather than inside ``pause()``. Such
                deadlocks are invisible to the pause-path detection; with this
                flag, a full ``deadlock_timeout`` window with zero scheduler
                progress records a deadlock in ``self._error`` before timing
                out, so callers can tell it from a slow-but-correct run.
        """
        if isinstance(task_funcs, list):
            task_funcs = dict(enumerate(task_funcs))

        self._num_tasks = len(task_funcs)
        errors: dict[Any, Exception] = {}
        cancelling_for_timeout = False

        async def _run(task_id: Any, func: Callable[..., Awaitable[None]]) -> None:
            try:
                self._setup_task_context(task_id)
                await func()
            except asyncio.CancelledError as e:
                if cancelling_for_timeout:
                    raise
                error = WorkerCancelledError(str(e) or f"worker {task_id!r} cancelled itself")
                errors[task_id] = error
                await self._report_error(error)
            except Exception as e:
                errors[task_id] = e
                await self._report_error(e)
            finally:
                self._cleanup_task_context(task_id)
                await self._mark_done(task_id)

        tasks = [asyncio.create_task(_run(tid, func), name=str(tid)) for tid, func in task_funcs.items()]

        try:
            gathered = asyncio.gather(*tasks, return_exceptions=True)
            if detect_external_deadlock:
                await self._wait_watching_progress(gathered, timeout)
            else:
                # Keep timeout cancellation under run_all's control.  On
                # Python 3.10/3.11 wait_for cancels its target before raising,
                # which otherwise reaches _run while
                # cancelling_for_timeout is still false and is misreported as
                # a worker cancelling itself.
                await frontrun_wait_for(asyncio.shield(gathered), timeout=timeout)
        except asyncio.TimeoutError:
            cancelling_for_timeout = True
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise SchedulerTimeoutError("Tasks did not complete within timeout. Check for deadlocks in your schedule.")

        if errors:
            raise next(iter(errors.values()))

        # A scheduler-detected deadlock/timeout sets ``self._error`` and makes
        # every subsequent ``pause()`` short-circuit, so the tasks free-run to
        # completion and the gather above returns normally.  Surface that error
        # here instead of silently scoring the free-run as a valid exploration
        # run.  The exploration loop classifies it (deadlock vs.
        # scheduler-timeout) just like the sync driver does.
        if self._error is not None:
            raise self._error

    async def _wait_watching_progress(self, gathered: "asyncio.Future[Any]", timeout: float) -> None:
        """Await *gathered* like ``asyncio.wait_for``, watching for deadlock.

        If a full ``deadlock_timeout`` window passes with zero progress (no
        pause() call and no task completion) while tasks remain unfinished,
        every unfinished task is blocked on an awaitable the scheduler does
        not manage — a deadlock. Record it in ``self._error`` and time out.

        A CPU-bound task starves the event loop, so our wait can only fire at
        the moment such a task yields — before its pause()/completion has had
        a chance to run. Grant a few loop passes for queued continuations to
        register progress before declaring deadlock.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            before = self._progress
            token = _in_frontrun_timer.set(True)
            try:
                done, _ = await asyncio.wait({gathered}, timeout=min(self.deadlock_timeout, remaining))
            finally:
                _in_frontrun_timer.reset(token)
            if done:
                gathered.result()
                return
            if self._progress != before:
                continue
            for _ in range(10):
                await asyncio.sleep(0)
                if self._progress != before or gathered.done():
                    break
            if gathered.done():
                return
            if self._progress == before:
                async with self._condition:
                    # A virtual clock can still move time forward to the next
                    # pending deadline (default: no clock, returns False).
                    if self._advance_virtual_deadline_for_idle():
                        self._condition.notify_all()
                        continue
                    if self._error is None:
                        self._error = SchedulerTimeoutError(
                            f"Deadlock: no task progressed for {self.deadlock_timeout}s and no task is "
                            "inside the scheduler; unfinished tasks are blocked on unmanaged awaitables "
                            "(e.g. stock asyncio locks)"
                        )
                        self._on_error_set()
                    self._condition.notify_all()
                break
        # run_all owns cancellation so it can mark the cancellation as
        # scheduler cleanup before CancelledError reaches worker wrappers.
        raise asyncio.TimeoutError

    @property
    def had_error(self) -> bool:
        """True if an error was reported during execution."""
        return self._error is not None


class _AsyncSchedulerBase(InterleavedLoop):
    """Shared machinery for the async DPOR exploration and replay schedulers.

    Both drive tasks through the same condition-gated park/wake protocol and
    differ only in how they pick the next task: the exploration scheduler asks
    the DPOR engine (``_schedule_next`` / ``_set_current_task``), while the
    replay scheduler walks a recorded schedule (``_advance``).  The common
    kick / wait / notify skeleton lives here; subclasses fill in the
    scheduling-specific hooks.

    Subclasses set ``_current_task`` / ``_current_task_consumed`` in their own
    ``__init__`` and may override ``_deadlock_prefix`` for error messages.
    """

    _current_task: int | None
    _current_task_consumed: bool
    #: Prefix for the "never scheduled" watchdog error (replay overrides it).
    _deadlock_prefix: str = "Deadlock"

    def _notify_waiters_soon(self) -> None:
        async def _notify() -> None:
            async with self._condition:
                self._condition.notify_all()

        asyncio.get_running_loop().create_task(_notify())

    # -- scheduling hooks (subclass-provided) ---------------------------

    def _should_kick(self, task_id: int) -> bool:
        """Whether ``kick_stalled_schedule`` should hand the turn onward now."""
        raise NotImplementedError

    def _perform_kick(self, task_id: int) -> None:
        """Reschedule after ``_should_kick`` returned True (holding the condition)."""
        raise NotImplementedError

    def _recover_stalled_schedule(self) -> bool:
        """While waiting to be scheduled, try to unstick a stalled current task.

        Returns True if it made progress (the caller re-checks and continues),
        False to fall through to the condition wait.
        """
        return False

    def _on_scheduled_after_block(self, task_id: int) -> None:
        """Run once the task is scheduled again after a physical wake."""
        self._current_task_consumed = True

    # -- shared park/wake control ---------------------------------------

    async def kick_stalled_schedule(self, task_id: int) -> None:
        """Hand the turn onward after *task_id* engine-blocked itself.

        Called by the cooperative primitives right after they block the task:
        the blocked task parks with no further scheduling points, so if it held
        the turn (or nothing else is runnable) no other path would drive the
        next scheduling decision and the run would die by deadlock timeout.
        """
        async with self._condition:
            if self._finished or self._error:
                return
            if self._should_kick(task_id):
                self._perform_kick(task_id)
                self._condition.notify_all()

    async def wait_until_scheduled_after_block(self, task_id: int, reason: str) -> None:
        """Wait for a physically-woken blocked task to be scheduled again."""
        async with self._condition:
            while not (self._finished or self._error) and self._current_task != task_id:
                if self._recover_stalled_schedule():
                    continue
                try:
                    await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                except asyncio.TimeoutError:
                    self._error = SchedulerTimeoutError(
                        f"{self._deadlock_prefix}: task {task_id} woke from {reason} but was never scheduled"
                    )
                    self._condition.notify_all()
                    return
            if not (self._finished or self._error):
                self._on_scheduled_after_block(task_id)

    async def pause(self, task_id: Any, marker: Any = None) -> None:
        """DPOR/replay-aware pause that ensures fair task wakeup.

        After proceeding from a pause, yields to the event loop so other tasks
        that were notified can process their condition waits.  Without this, a
        single task can reacquire the condition lock before other notified tasks,
        causing false deadlock detection.

        Sets ``_in_scheduler_pause`` so the coroutine wrapper knows not to insert
        a redundant scheduling point for this pause's own yields.
        """
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            # Yield to let any previously-notified tasks process their wakeups.
            await _real_asyncio_sleep(0)
            await super().pause(task_id, marker)
        finally:
            _in_scheduler_pause.set(depth)
