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
- async_bytecode (property-based): index-based scheduling

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
from collections.abc import Awaitable, Callable
from typing import Any


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

    def __init__(self):
        self._condition = asyncio.Condition()
        self._finished = False
        self._error: Exception | None = None
        self._tasks_done: set[Any] = set()

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

        Args:
            task_id: Identity of the calling task.
            marker: Optional scheduling context.
        """
        async with self._condition:
            while True:
                if self._finished or self._error:
                    return

                if self.should_proceed(task_id, marker):
                    self.on_proceed(task_id, marker)
                    self._condition.notify_all()
                    return

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._handle_timeout(task_id, marker)
                    return

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        """Handle a timeout in pause(). Sets the error and wakes everyone.

        Override to provide a more informative error message.
        """
        self._error = TimeoutError(f"Deadlock: task {task_id!r} timed out waiting at marker {marker!r}")
        self._condition.notify_all()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished and notify waiting tasks."""
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
    ) -> None:
        """Run tasks with controlled interleaving.

        Args:
            task_funcs: Either a dict ``{task_id: async_callable}`` or a
                list of async callables (which get integer task_ids
                0, 1, 2, ...).
            timeout: Maximum total time to wait for all tasks.
        """
        if isinstance(task_funcs, list):
            task_funcs = dict(enumerate(task_funcs))

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
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise TimeoutError("Tasks did not complete within timeout. Check for deadlocks in your schedule.")

        if errors:
            raise next(iter(errors.values()))

    @property
    def had_error(self) -> bool:
        """True if an error was reported during execution."""
        return self._error is not None
