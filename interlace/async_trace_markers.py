"""
Interlace: Deterministic async task interleaving using comment-based markers.

This module provides a mechanism to control async task execution order by marking
synchronization points in code with ``# interlace: marker_name`` comments,
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
    current = self.balance  # interlace: read_balance

Separate-line markers (marker before operation):
    # interlace: read_balance
    current = self.balance

Both styles work identically: the marker gates execution of the line, ensuring
it only executes after the scheduler approves this task at this marker.

Example usage::

    async def worker_function():
        # interlace: read_data
        x = await read_data()
        # interlace: write_data
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

    async_interlace(
        schedule=schedule,
        tasks={'task1': worker1, 'task2': worker2},
    )
"""

import asyncio
import sys
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from interlace.common import Schedule
from interlace.trace_markers import MarkerRegistry, ThreadCoordinator


class AsyncTraceExecutor:
    """Executes async tasks with interlaced execution according to a schedule.

    This is the main interface for the async interlace library. It uses
    comment-based markers (# interlace: marker_name) to control task
    execution order.

    Unlike the sync version which runs tasks in actual threads, this runs
    each async task in its own thread with its own event loop via asyncio.run().
    """

    def __init__(self, schedule: Schedule):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
        """
        self.schedule = schedule
        self.coordinator = ThreadCoordinator(schedule)
        self.marker_registry = MarkerRegistry()
        self.task_errors: dict[str, Exception] = {}

    def _create_trace_function(self, execution_name: str) -> Callable[[Any, str, Any], Any]:
        """Create a trace function for a specific async task.

        This trace function checks both the current line and previous line for markers
        (to support both inline and separate-line markers) and blocks the thread when
        a marker is detected.

        Args:
            execution_name: The name of the task this trace function is for

        Returns:
            A trace function suitable for sys.settrace
        """
        processed_locations: set[tuple[str, int]] = set()

        def trace_function(frame: Any, event: str, arg: Any) -> Any:
            try:
                # Only care about 'line' events
                if event != "line":
                    return trace_function

                # Scan this file for markers if we haven't already
                self.marker_registry.scan_frame(frame)

                filename = frame.f_code.co_filename
                lineno = frame.f_lineno

                # Check current line (for inline markers like: x = 1  # interlace: marker)
                marker_name = self.marker_registry.get_marker(filename, lineno)
                if marker_name and (filename, lineno) not in processed_locations:
                    processed_locations.add((filename, lineno))
                    # Block this thread until it's our turn
                    self.coordinator.wait_for_turn(execution_name, marker_name)
                    # Check if an error occurred in another task
                    if self.coordinator.error:
                        raise self.coordinator.error
                    return trace_function

                # Check previous line (for separate-line markers)
                if lineno > 1:
                    prev_marker = self.marker_registry.get_marker(filename, lineno - 1)
                    if prev_marker and (filename, lineno - 1) not in processed_locations:
                        processed_locations.add((filename, lineno - 1))
                        # Block this thread until it's our turn
                        self.coordinator.wait_for_turn(execution_name, prev_marker)
                        # Check if an error occurred in another task
                        if self.coordinator.error:
                            raise self.coordinator.error

                return trace_function
            except Exception as e:
                # Report error and stop tracing
                self.coordinator.report_error(e)
                return None

        return trace_function

    def _thread_wrapper(self, execution_name: str, task_fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Wrapper that runs an async task in its own event loop with tracing.

        Args:
            execution_name: The name of this task
            task_fn: The async function to execute
        """
        try:
            # Install trace function for this task
            trace_fn = self._create_trace_function(execution_name)
            sys.settrace(trace_fn)

            # Run the async task in its own event loop
            asyncio.run(task_fn())
        except Exception as e:
            # Store the error
            self.task_errors[execution_name] = e
            self.coordinator.report_error(e)
        finally:
            # Clean up trace function
            sys.settrace(None)

    def run(self, tasks: dict[str, Callable[[], Coroutine[Any, Any, None]]], timeout: float = 10.0) -> None:
        """Run all tasks with controlled interleaving based on comment markers.

        This is now a synchronous method that creates threads and waits for them.

        Args:
            tasks: Dictionary mapping task names to their async functions
            timeout: Timeout in seconds for all tasks to complete

        Raises:
            TimeoutError: If tasks don't complete within the timeout
            Any exception that occurred in a task during execution
        """
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

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=timeout)

        # Check if any threads are still alive after timeout
        alive_threads: list[threading.Thread] = [thread for thread in threads if thread.is_alive()]
        if alive_threads:
            thread_names = [thread.name for thread in alive_threads]
            raise TimeoutError(f"Tasks did not complete within {timeout}s: {thread_names}")

        # If any task had an error, raise the first one
        if self.task_errors:
            first_error = next(iter(self.task_errors.values()))
            raise first_error

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.task_errors = {}
        self.coordinator = ThreadCoordinator(self.schedule)
        self.marker_registry = MarkerRegistry()


def async_interlace(
    schedule: Schedule,
    tasks: dict[str, Callable[..., Coroutine[Any, Any, None]]],
    task_args: dict[str, tuple[Any, ...]] | None = None,
    task_kwargs: dict[str, dict[str, Any]] | None = None,
    timeout: float = 10.0,
) -> AsyncTraceExecutor:
    """Convenience function to run multiple async tasks with a schedule.

    This is now a synchronous function (not async) that creates an executor
    and runs the tasks.

    Tasks use # interlace: marker_name comments to mark synchronization points.
    No need to pass marker functions to tasks - the executor automatically
    detects markers via sys.settrace.

    Args:
        schedule: The Schedule defining execution order
        tasks: Dictionary mapping execution unit names to their async target functions
        task_args: Optional dictionary mapping execution unit names to argument tuples
        task_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Timeout in seconds for the entire execution

    Returns:
        The AsyncTraceExecutor instance (useful for inspection)

    Example::

        async def worker(account, amount):
            # interlace: before_deposit
            await account.deposit(amount)

        async_interlace(
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

    executor = AsyncTraceExecutor(schedule)

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
