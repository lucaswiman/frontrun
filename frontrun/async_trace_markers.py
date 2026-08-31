"""
Frontrun: Deterministic async task interleaving using comment-based markers.

This module provides a mechanism to control async task execution order by marking
synchronization points in code with ``# frontrun: marker_name`` comments,
matching the elegant syntax of the sync trace_markers module.

Key Insight: Thread-Based Execution with sys.settrace
======================================================

This implementation mirrors the sync architecture exactly:

1. Each async task runs in its own thread via ``asyncio.run(task_fn())``
2. ``sys.settrace`` fires on every line, including inside synchronous sub-coroutines
3. When a marker is detected, the trace function blocks the thread via
   ``ThreadCoordinator.wait_for_turn()``
4. Execution resumes exactly where it paused

This solves the synchronous function body bug: when ``await self.get_balance()``
completes synchronously (no internal yields), the trace still fires on every
line during that execution. After ``get_balance()`` returns, the trace fires for
the next line (after the marker), detects the marker, and blocks before that
line executes.

Marker Semantics
=================

**Marker Placement**: Place markers to gate the operations you want to control.

Inline markers (marker on same line as operation):
    current = self.balance  # frontrun: read_balance

Separate-line markers (marker before operation):
    # frontrun: read_balance
    current = self.balance

Both styles work identically: the marker gates execution of the line, ensuring
it only executes after the scheduler approves this task at this marker.

Example usage::

    async def worker_function():
        # frontrun: read_data
        x = await read_data()
        # frontrun: write_data
        await write_data(x)

    schedule = Schedule([
        Step("task1", "read_data"),
        Step("task2", "read_data"),
        Step("task1", "write_data"),
        Step("task2", "write_data"),
    ])

    executor = AsyncTraceExecutor(schedule)
    executor.run({
        'task1': worker_function,
        'task2': worker_function,
    })

Or using the convenience function::

    async_frontrun(
        schedule=schedule,
        tasks={'task1': worker1, 'task2': worker2},
    )
"""

import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from frontrun._marker_coordination import (
    MarkerRegistry,
    ThreadCoordinator,
    finalize_marker_executor_run,
)
from frontrun._trace_marker_runtime import build_trace_function, run_traced_callable
from frontrun.common import Schedule


class AsyncTraceExecutor:
    """Executes async tasks with interlaced execution according to a schedule.

    This is the main interface for the async frontrun library. It uses
    comment-based markers (# frontrun: marker_name) to control task
    execution order.

    Unlike the sync version which runs tasks in actual threads, this runs
    each async task in its own thread with its own event loop via asyncio.run().
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
            deadlock_timeout: Seconds to wait before declaring a deadlock
                (default 5.0).  Increase for code that legitimately blocks
                in C extensions (NumPy, database queries, network I/O).
        """
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self.coordinator = ThreadCoordinator(schedule, deadlock_timeout=deadlock_timeout)
        self.marker_registry = MarkerRegistry()
        self.task_errors: dict[str, BaseException] = {}

    def _create_trace_function(self, execution_name: str) -> Callable[[Any, str, Any], Any]:
        return build_trace_function(
            self.coordinator,
            self.marker_registry,
            execution_name,
            include_previous_line=True,
        )

    def _thread_wrapper(self, execution_name: str, task_fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Wrapper that runs an async task in its own event loop with tracing.

        Args:
            execution_name: The name of this task
            task_fn: The async function to execute
        """
        run_traced_callable(
            coordinator=self.coordinator,
            execution_name=execution_name,
            trace_function=self._create_trace_function(execution_name),
            body=lambda: asyncio.run(task_fn()),
            error_sink=self.task_errors,
        )

    def run(self, tasks: dict[str, Callable[[], Coroutine[Any, Any, None]]], timeout: float = 10.0) -> None:
        """Run all tasks with controlled interleaving based on comment markers.

        This is now a synchronous method that creates threads and waits for them.

        Args:
            tasks: Dictionary mapping task names to their async functions
            timeout: Timeout in seconds for all tasks to complete

        Raises:
            ValueError: If *tasks* is empty
            TypeError: If any value in *tasks* is not callable
            TimeoutError: If tasks don't complete within the timeout
            Any exception that occurred in a task during execution
        """
        if not tasks:
            raise ValueError(
                "AsyncTraceExecutor.run() received an empty dict — pass at least one {name: callable} entry."
            )
        for name, task_fn in tasks.items():
            if not callable(task_fn):
                raise TypeError(
                    f"AsyncTraceExecutor.run(): value for {name!r} must be callable, got {type(task_fn).__name__!r}."
                )

        threads: list[threading.Thread] = []
        for execution_name, task_fn in tasks.items():
            thread = threading.Thread(
                target=self._thread_wrapper,
                args=(execution_name, task_fn),
                name=execution_name,
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        finalize_marker_executor_run(
            threads=threads,
            timeout=timeout,
            task_errors=self.task_errors,
            coordinator=self.coordinator,
            timeout_message=lambda alive: f"Tasks did not complete within {timeout}s: {[t.name for t in alive]}",
        )

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.task_errors = {}
        self.coordinator = ThreadCoordinator(self.schedule, deadlock_timeout=self.deadlock_timeout)
        self.marker_registry = MarkerRegistry()


def async_frontrun(
    schedule: Schedule,
    tasks: dict[str, Callable[..., Coroutine[Any, Any, None]]],
    task_args: dict[str, tuple[Any, ...]] | None = None,
    task_kwargs: dict[str, dict[str, Any]] | None = None,
    timeout: float = 10.0,
    deadlock_timeout: float = 5.0,
) -> AsyncTraceExecutor:
    """Convenience function to run multiple async tasks with a schedule.

    This is now a synchronous function (not async) that creates an executor
    and runs the tasks.

    Tasks use # frontrun: marker_name comments to mark synchronization points.
    No need to pass marker functions to tasks - the executor automatically
    detects markers via sys.settrace.

    Args:
        schedule: The Schedule defining execution order
        tasks: Dictionary mapping execution unit names to their async target functions
        task_args: Optional dictionary mapping execution unit names to argument tuples
        task_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Timeout in seconds for the entire execution
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).

    Returns:
        The AsyncTraceExecutor instance (useful for inspection)

    Example::

        async def worker(account, amount):
            # frontrun: before_deposit
            await account.deposit(amount)

        async_frontrun(
            schedule=Schedule([
                Step("t1", "before_deposit"),
                Step("t2", "before_deposit")
            ]),
            tasks={"t1": worker, "t2": worker},
            task_args={"t1": (account, 50), "t2": (account, 50)},
        )
    """
    if task_args is None:
        task_args = {}
    if task_kwargs is None:
        task_kwargs = {}

    executor = AsyncTraceExecutor(schedule, deadlock_timeout=deadlock_timeout)

    # Create wrapped tasks that call the target with args/kwargs
    wrapped_tasks: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}
    for execution_name, target in tasks.items():
        args = task_args.get(execution_name, ())
        kwargs = task_kwargs.get(execution_name, {})

        # Create a coroutine function that calls the target with args/kwargs
        async def make_task(
            target: Callable[..., Coroutine[Any, Any, None]] = target,
            args: tuple[Any, ...] = args,
            kwargs: dict[str, Any] = kwargs,
        ) -> None:
            return await target(*args, **kwargs)

        wrapped_tasks[execution_name] = make_task

    # Run the tasks
    executor.run(wrapped_tasks, timeout=timeout)

    return executor
