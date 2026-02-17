"""
Interlace: Deterministic async task interleaving using comment-based markers.

This module provides a mechanism to control async task execution order by marking
synchronization points in code with ``# interlace: marker_name`` comments,
matching the elegant syntax of the sync trace_markers module.

Built on the shared InterleavedLoop abstraction, using step-based scheduling
(list of Step(execution_name, marker_name) entries) as its policy.

Key Insight: Comment-Based Markers via Coroutine-Driving
==========================================================

Unlike the sync version which uses sys.settrace during normal thread execution,
async requires a different approach because you can't ``await`` from a trace
function.

Solution: Wrap each user coroutine in a "driver" that manually steps it with
``coro.send()``, checking for markers between each yield:

1. Install sys.settrace (marker-detecting trace function)
2. ``coro.send(value)`` -- runs coroutine until next await/return (synchronous)
3. Uninstall sys.settrace
4. Await the yielded value (the thing the coroutine awaited).
   This yields to the event loop, giving other tasks a chance to run.
5. For each marker detected during step 2:
   ``await coordinator.pause(task_id, marker_name)`` -- blocks until schedule
   says this task can proceed.
6. Loop back to step 2 immediately (no yield between approval and send).

This works because:
- Async is single-threaded, so coro.send() is synchronous -- no trace conflicts
- Reuses MarkerRegistry from sync for comment detection
- Reuses InterleavedLoop/AsyncTaskCoordinator for scheduling
- Works with any awaitable (sleep, I/O, sub-coroutines)

Marker Semantics
=================

A ``# interlace: name`` comment followed by ``await asyncio.sleep(0)`` creates
a scheduling gate. The marker is detected during ``send()``, then the driver
yields to the event loop and waits for schedule approval. After approval, the
next ``send()`` executes the code that follows the ``await`` -- this is the code
gated by the marker.

**Important**: Place the marker comment and ``await`` BEFORE the operation being
gated, not after it::

    # interlace: read_balance
    await asyncio.sleep(0)     # yield point -- marker detected here
    current = self.balance     # runs after schedule approves this task

This ensures the scheduler can control when each operation actually executes.

Example usage::

    async def worker_function():
        # interlace: before_read
        await asyncio.sleep(0)       # yield point for marker
        x = await read_data()
        # interlace: before_write
        await asyncio.sleep(0)       # yield point for marker
        await write_data(x)

    schedule = Schedule([
        Step("task1", "before_read"),
        Step("task2", "before_read"),
        Step("task1", "before_write"),
        Step("task2", "before_write"),
    ])

    async def main():
        executor = AsyncTraceExecutor(schedule)
        await executor.run({
            'task1': worker_function,
            'task2': worker_function,
        })

    asyncio.run(main())

Or using the convenience function::

    await async_interlace(
        schedule=schedule,
        tasks={'task1': worker1, 'task2': worker2},
    )
"""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from interlace.async_scheduler import InterleavedLoop
from interlace.common import Schedule
from interlace.trace_markers import MarkerRegistry


class _TraceState:
    """Accumulates markers detected during a coroutine step.

    Used by the trace function to collect markers as it sees them,
    then drained after each send() to process them.

    Deduplicates markers by (filename, lineno) to avoid double-detection
    when both the current-line check and the previous-line check find
    the same marker (which happens for inline markers like
    ``x = 1  # interlace: marker``).
    """

    def __init__(self):
        self.markers: list[str] = []
        self._seen_locations: set[tuple[str, int]] = set()

    def add_marker(self, marker_name: str, filename: str, lineno: int):
        """Add a marker detected during tracing, deduplicating by source location.

        Args:
            marker_name: The marker name
            filename: Source file where the marker was found
            lineno: Line number of the marker in the source file
        """
        key = (filename, lineno)
        if key not in self._seen_locations:
            self._seen_locations.add(key)
            self.markers.append(marker_name)

    def drain(self) -> list[str]:
        """Return and clear all accumulated markers."""
        result = self.markers
        self.markers = []
        self._seen_locations.clear()
        return result


def _make_marker_trace_fn(registry: MarkerRegistry, trace_state: _TraceState) -> Callable[[Any, str, Any], Any]:
    """Create a trace function that detects markers and adds them to trace_state.

    Checks both the current line and the previous line for markers to support
    both inline markers and markers on separate lines.

    Args:
        registry: MarkerRegistry to use for marker lookup
        trace_state: _TraceState to accumulate detected markers

    Returns:
        A trace function suitable for sys.settrace
    """

    def trace_function(frame: Any, event: str, arg: Any) -> Any:
        # Only care about 'line' events
        if event != "line":
            return trace_function

        # Scan this file for markers if we haven't already
        registry.scan_frame(frame)

        # Check if this line or the previous line has a marker.
        # Both checks are needed:
        # - Current line: for inline markers (e.g., ``x = 1  # interlace: marker``)
        # - Previous line: for comment-only markers on a separate line
        # Deduplication in _TraceState prevents double-counting when both
        # checks find the same marker (e.g., inline marker found on its own
        # line, then found again via prev-line check on the next line).
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Check current line (for inline markers)
        marker_name = registry.get_marker(filename, lineno)
        if marker_name:
            trace_state.add_marker(marker_name, filename, lineno)

        # Check previous line (for markers on separate lines)
        if lineno > 1:
            prev_marker = registry.get_marker(filename, lineno - 1)
            if prev_marker:
                trace_state.add_marker(prev_marker, filename, lineno - 1)

        return trace_function

    return trace_function


async def _await_future(future: asyncio.Future[Any]) -> Any:
    """Wait for a Future to complete without directly awaiting it.

    When manually driving a coroutine with send(), the yielded Futures
    cannot be directly ``await``-ed from a different async context because
    CPython raises "await wasn't used with future". This helper uses
    ``add_done_callback`` and an ``asyncio.Event`` to wait for completion.

    Args:
        future: The Future to wait for.

    Returns:
        The Future's result.

    Raises:
        The Future's exception, if any.
    """
    if future.done():
        return future.result()

    event = asyncio.Event()
    future.add_done_callback(lambda _: event.set())
    await event.wait()
    return future.result()


async def _drive_with_markers(
    coro: Any,  # Coroutine object
    task_id: str,
    coordinator: "AsyncTaskCoordinator",
    registry: MarkerRegistry,
) -> Any:
    """Drive a coroutine with marker detection and coordination.

    This is the core of the comment-based async marker system. It manually
    drives the coroutine with send()/throw(), installing settrace around each
    step to detect markers, then processing them via coordinator.pause().

    The execution order within each iteration is critical for correctness:

    1. send(value) — Runs the coroutine synchronously until the next await.
       During this step, the trace function detects any markers passed.
    2. await yielded — Await the value the coroutine yielded (e.g., a Future
       from asyncio.sleep()). This yields to the event loop, giving other
       tasks a chance to run their send() and reach their own marker pauses.
    3. pause() for each marker — Block until the schedule says this task can
       proceed. This is where the scheduling policy is enforced.
    4. Loop back to send() — Immediately after approval, with NO yield in
       between. This ensures the code gated by the marker runs atomically
       before any other task can execute.

    The key invariant: no event loop yield occurs between pause() approval
    (step 3) and the next send() (step 1). This means the marker gates the
    code in the NEXT send(), which is the code after the ``await`` that
    follows the marker comment.

    Recommended code pattern::

        # interlace: marker_name
        await asyncio.sleep(0)     # yield point — marker detected here
        # ... code gated by marker runs in the next send() ...

    Args:
        coro: The coroutine to drive
        task_id: The task identifier for coordination
        coordinator: The AsyncTaskCoordinator managing scheduling
        registry: The MarkerRegistry for marker detection

    Returns:
        The coroutine's return value

    Raises:
        Any exception raised by the coroutine
    """
    trace_state = _TraceState()
    trace_fn = _make_marker_trace_fn(registry, trace_state)
    value = None

    try:
        while True:
            # Step 1: Install trace and drive coroutine to next yield point.
            # This runs synchronously — the coroutine executes code until it
            # hits an ``await``, at which point send() returns the yielded value.
            sys.settrace(trace_fn)
            try:
                yielded = coro.send(value)
            except StopIteration as e:
                # Coroutine finished — process any final markers
                for marker in trace_state.drain():
                    await coordinator.pause(task_id, marker)
                return e.value
            finally:
                sys.settrace(None)

            # Step 2: Collect markers detected during this send().
            markers = trace_state.drain()

            # Step 3: Await the yielded value to get the result to feed back.
            # This yields to the event loop, giving other tasks a chance to
            # run their send() calls and reach their own pause points.
            # When yielded is None (e.g., from asyncio.sleep(0)), we still
            # yield to the event loop explicitly.
            if yielded is None:
                await asyncio.sleep(0)
                value = None
            elif isinstance(yielded, asyncio.Future):
                # Futures from the coroutine protocol (e.g., from asyncio.sleep()
                # with a non-zero delay) cannot be directly awaited from a
                # different async context — doing so raises "await wasn't used
                # with future". Instead, wait for the Future to complete using
                # a done callback and an Event.
                value = await _await_future(yielded)  # type: ignore[reportUnknownArgumentType]
            else:
                value = await yielded

            # Step 4: Process markers — block until the schedule says go.
            # IMPORTANT: This must be the LAST step before looping back to
            # send(). After pause() approves this task, we go directly to
            # send() with no intervening yield, ensuring the gated code
            # runs atomically.
            for marker in markers:
                await coordinator.pause(task_id, marker)
    finally:
        coro.close()


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

    This is the main interface for the async interlace library. It uses
    comment-based markers (# interlace: marker_name) to control task
    execution order.
    """

    def __init__(self, schedule: Schedule):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
        """
        self.schedule = schedule
        self.coordinator = AsyncTaskCoordinator(schedule)
        self.marker_registry = MarkerRegistry()
        self.task_errors: dict[str, Exception] = {}

    async def run(self, tasks: dict[str, Callable[[], Awaitable[None]]]):
        """Run all tasks with controlled interleaving based on comment markers.

        Each task function is wrapped with _drive_with_markers() which detects
        # interlace: marker_name comments and coordinates execution via the
        scheduler.

        Args:
            tasks: Dictionary mapping task names to their async functions

        Raises:
            Any exception that occurred in a task during execution
        """
        # Wrap each task with the marker driver
        wrapped_tasks: dict[str, Callable[..., Awaitable[None]]] = {}
        for execution_name, task_fn in tasks.items():

            async def _wrapped_task(
                task_fn: Callable[..., Awaitable[None]] = task_fn,
                execution_name: str = execution_name,
            ) -> None:
                try:
                    # Create the coroutine from the task function
                    coro = task_fn()
                    # Drive it with marker detection
                    await _drive_with_markers(coro, execution_name, self.coordinator, self.marker_registry)
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
        self.marker_registry = MarkerRegistry()


async def async_interlace(
    schedule: Schedule,
    tasks: dict[str, Callable[..., Awaitable[None]]],
    task_args: dict[str, tuple[Any, ...]] | None = None,
    task_kwargs: dict[str, dict[str, Any]] | None = None,
    timeout: float | None = None,
) -> "AsyncTraceExecutor":
    """Convenience function to run multiple async tasks with a schedule.

    Tasks use # interlace: marker_name comments to mark synchronization points.
    No need to pass marker functions to tasks - the executor automatically
    detects markers via sys.settrace.

    Args:
        schedule: The Schedule defining execution order
        tasks: Dictionary mapping execution unit names to their async target functions
        task_args: Optional dictionary mapping execution unit names to argument tuples
        task_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Optional timeout in seconds for the entire execution

    Returns:
        The AsyncTraceExecutor instance (useful for inspection)

    Example::

        async def worker(account, amount):
            # interlace: before_deposit
            await asyncio.sleep(0)  # yield point for marker
            await account.deposit(amount)

        await async_interlace(
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
    wrapped_tasks: dict[str, Callable[..., Awaitable[None]]] = {}
    for execution_name, target in tasks.items():
        args = task_args.get(execution_name, ())
        kwargs = task_kwargs.get(execution_name, {})

        # Create a coroutine that calls the target with args/kwargs
        async def make_task(
            target: Callable[..., Awaitable[None]] = target,
            args: tuple[Any, ...] = args,
            kwargs: dict[str, Any] = kwargs,
        ) -> None:
            return await target(*args, **kwargs)

        wrapped_tasks[execution_name] = make_task

    # Run with optional timeout
    if timeout is not None:
        await asyncio.wait_for(executor.run(wrapped_tasks), timeout=timeout)
    else:
        await executor.run(wrapped_tasks)

    return executor
