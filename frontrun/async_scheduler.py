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
import contextvars
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")

# True while a frontrun-internal timed wait is creating its loop timer.
# Exact async deadlock detection must tell the scheduler's own watchdog
# timers apart from user timers (a pending user timer means a parked task
# may still be woken, so it is not a proven deadlock); the async DPOR
# driver tags timers created under this flag via a loop.call_at wrapper.
_in_frontrun_timer: contextvars.ContextVar[bool] = contextvars.ContextVar("frontrun_in_frontrun_timer", default=False)


class SchedulerTimeoutError(TimeoutError):
    """Timeout raised by frontrun's scheduler machinery, not user code."""


async def frontrun_wait_for(awaitable: Coroutine[Any, Any, _T] | Awaitable[_T], timeout: float) -> _T:
    """``asyncio.wait_for`` whose timeout timer is tagged as frontrun-internal."""
    token = _in_frontrun_timer.set(True)
    try:
        return await asyncio.wait_for(awaitable, timeout)
    finally:
        _in_frontrun_timer.reset(token)


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

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        """Handle a timeout in pause(). Sets the error and wakes everyone.

        Override to provide a more informative error message.
        """
        self._error = SchedulerTimeoutError(
            f"Deadlock: task {task_id!r} timed out waiting at marker {marker!r} (fallback timeout)"
        )
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

        async def _run(task_id: Any, func: Callable[..., Awaitable[None]]) -> None:
            try:
                self._setup_task_context(task_id)
                await func()
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
                await frontrun_wait_for(gathered, timeout=timeout)
        except asyncio.TimeoutError:
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
        # (finding F1).  The exploration loop classifies it (deadlock vs
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
                    if self._error is None:
                        self._error = SchedulerTimeoutError(
                            f"Deadlock: no task progressed for {self.deadlock_timeout}s and no task is "
                            "inside the scheduler; unfinished tasks are blocked on unmanaged awaitables "
                            "(e.g. stock asyncio locks)"
                        )
                    self._condition.notify_all()
                break
        # Mirror asyncio.wait_for's contract on timeout: cancel the awaited
        # future (which cancels the still-pending tasks) before raising.
        gathered.cancel()
        try:
            await gathered
        except asyncio.CancelledError:
            pass
        raise asyncio.TimeoutError

    @property
    def had_error(self) -> bool:
        """True if an error was reported during execution."""
        return self._error is not None
