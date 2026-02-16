"""
Interlace: Deterministic async task interleaving using explicit markers.

This module provides a mechanism to control async task execution order by marking
synchronization points in code with explicit await calls that enforce a predefined
execution schedule. This is the async/await adaptation of the threading-based
trace_markers module.

Built on the shared InterleavedLoop abstraction, using step-based scheduling
(list of Step(execution_name, marker_name) entries) as its policy.

Key Insight: In async code, race conditions ONLY happen at await points
============================================================================

Unlike threading where race conditions can occur at any line of code (because the OS
can preempt threads at any time), async code in Python uses cooperative multitasking
with a single-threaded event loop. The event loop only yields control at explicit
`await` statements. This means:

1. Race conditions between tasks can ONLY manifest at await points
2. Code between await points runs atomically (no interleaving)
3. We don't need sys.settrace - we can use explicit checkpoint functions

Translation from Threading to Async:
- Thread → Task (coroutine scheduled on event loop)
- # interlace: marker_name comment → await mark('marker_name') call
- sys.settrace() → explicit await checkpoints
- threading.Condition → asyncio.Condition (via InterleavedLoop)

The marker functions are designed to be lightweight and minimally invasive - they
serve the same role as comment markers in the threading version, but work naturally
with async's cooperative scheduling model.

Example usage:
    ```python
    async def worker_function(mark):
        x = await read_data()
        await mark('after_read')  # explicit marker point
        await write_data(x)
        await mark('after_write')

    schedule = Schedule([
        Step("task1", "after_read"),
        Step("task2", "after_read"),
        Step("task1", "after_write"),
        Step("task2", "after_write"),
    ])

    async def main():
        executor = AsyncTraceExecutor(schedule)
        mark1 = executor.marker('task1')
        mark2 = executor.marker('task2')

        await executor.run({
            'task1': lambda: worker_function(mark1),
            'task2': lambda: worker_function(mark2),
        })

    asyncio.run(main())
    ```

Or using the convenience function:
    ```python
    await async_interlace(
        schedule=schedule,
        tasks={'task1': worker1, 'task2': worker2},
    )
    ```
"""

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from interlace.async_scheduler import InterleavedLoop
from interlace.common import Schedule


class AsyncTaskCoordinator(InterleavedLoop):
    """Coordinates async task execution according to a schedule.

    Built on InterleavedLoop, using step-based scheduling: the schedule
    is a list of Step(execution_name, marker_name) entries.  Tasks call
    pause(execution_name, marker_name) at marker points and block until
    the schedule says it's their turn.
    """

    def __init__(self, schedule: Schedule):
        """Initialize the coordinator with a schedule.

        Args:
            schedule: The Schedule defining the execution order
        """
        super().__init__()
        self.schedule = schedule
        self.current_step = 0

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        if self.current_step >= len(self.schedule.steps):
            self._finished = True
            return True

        step = self.schedule.steps[self.current_step]
        return step.execution_name == task_id and step.marker_name == marker

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        if self.current_step < len(self.schedule.steps):
            self.current_step += 1


class AsyncTraceExecutor:
    """Executes async tasks with interlaced execution according to a schedule.

    This is the main interface for the async interlace library. It coordinates
    task execution by providing marker functions that tasks can await.
    """

    def __init__(self, schedule: Schedule):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
        """
        self.schedule = schedule
        self.coordinator = AsyncTaskCoordinator(schedule)
        self.task_errors: Dict[str, Exception] = {}

    def marker(self, execution_name: str) -> Callable[[str], Awaitable[None]]:
        """Returns a marker function for the given execution unit.

        The returned function is an async function that the task calls at
        synchronization points. This is the async equivalent of placing
        # interlace: marker_name comments in threaded code.

        Args:
            execution_name: The name of the execution unit (task/thread) this marker function is for

        Returns:
            An async function that takes a marker name and waits for the
            execution unit's turn to execute

        Usage:
            mark = executor.marker('task1')
            async def my_task():
                x = await read()
                await mark('after_read')  # lightweight marker
                await write(x)
                await mark('after_write')
        """

        async def _mark(marker_name: str):
            await self.coordinator.pause(execution_name, marker_name)

        return _mark

    async def run(self, tasks: Dict[str, Callable[[], Awaitable[None]]]):
        """Run all tasks with controlled interleaving.

        Delegates to the coordinator's InterleavedLoop.run_all() for task
        lifecycle management, error handling, and timeout.

        Args:
            tasks: Dictionary mapping task names to their async functions

        Raises:
            Any exception that occurred in a task during execution
        """
        # Wrap tasks to capture individual exceptions
        wrapped_tasks: Dict[str, Callable[..., Awaitable[None]]] = {}
        for execution_name, task in tasks.items():

            async def _wrapped_task(
                task: Callable[..., Awaitable[None]] = task, execution_name: str = execution_name
            ) -> None:  # type: ignore[assignment]
                try:
                    await task()
                except Exception as e:
                    self.task_errors[execution_name] = e
                    raise

            wrapped_tasks[execution_name] = _wrapped_task

        await self.coordinator.run_all(wrapped_tasks)

        # If any task had an error, raise the first one
        if self.task_errors:
            first_error = next(iter(self.task_errors.values()))
            raise first_error

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.task_errors = {}
        self.coordinator = AsyncTaskCoordinator(self.schedule)


async def async_interlace(
    schedule: Schedule,
    tasks: Dict[str, Callable[..., Awaitable[None]]],
    task_args: Optional[Dict[str, Tuple[Any, ...]]] = None,
    task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    timeout: Optional[float] = None,
) -> "AsyncTraceExecutor":
    """Convenience function to run multiple async tasks with a schedule.

    This function automatically creates marker functions for each task and
    passes them as the first argument (or 'mark' keyword argument) to the
    task function.

    Args:
        schedule: The Schedule defining execution order
        tasks: Dictionary mapping execution unit names to their async target functions
        task_args: Optional dictionary mapping execution unit names to argument tuples
        task_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Optional timeout in seconds for the entire execution

    Returns:
        The AsyncTraceExecutor instance (useful for inspection)

    Example:
        ```python
        async def worker(mark, account, amount):
            await account.deposit(amount)
            await mark('after_deposit')

        await async_interlace(
            schedule=Schedule([
                Step("t1", "after_deposit"),
                Step("t2", "after_deposit")
            ]),
            tasks={"t1": worker, "t2": worker},
            task_args={"t1": (account, 50), "t2": (account, 50)},
        )
        ```
    """
    if task_args is None:
        task_args = {}
    if task_kwargs is None:
        task_kwargs = {}

    executor = AsyncTraceExecutor(schedule)

    # Create wrapped tasks that inject the marker function
    wrapped_tasks: Dict[str, Callable[..., Awaitable[None]]] = {}
    for execution_name, target in tasks.items():
        mark = executor.marker(execution_name)
        args = task_args.get(execution_name, ())
        kwargs = task_kwargs.get(execution_name, {})

        # Create a coroutine that calls the target with mark as first arg
        async def make_task(
            target: Callable[..., Awaitable[None]] = target,
            mark: Callable[[str], Awaitable[None]] = mark,
            args: Tuple[Any, ...] = args,
            kwargs: Dict[str, Any] = kwargs,
        ) -> None:  # type: ignore[assignment]
            return await target(mark, *args, **kwargs)

        wrapped_tasks[execution_name] = make_task

    # Run with optional timeout
    if timeout is not None:
        await asyncio.wait_for(executor.run(wrapped_tasks), timeout=timeout)
    else:
        await executor.run(wrapped_tasks)

    return executor
