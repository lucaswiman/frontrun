"""
Frontrun: Deterministic thread interleaving using sys.settrace and comment markers.

This module provides a mechanism to control thread execution order by marking
synchronization points in code with special comments and enforcing a predefined
execution schedule using Python's tracing facilities.

Example usage:
    ```python
    from frontrun.common import Schedule, Step
    from frontrun.trace_markers import TraceExecutor

    def worker_function():
        x = read_data()  # frontrun: read
        write_data(x)    # frontrun: write

    schedule = Schedule([
        Step("thread1", "read"),
        Step("thread2", "read"),
        Step("thread1", "write"),
        Step("thread2", "write"),
    ])

    executor = TraceExecutor(schedule)
    executor.run({"thread1": worker_function, "thread2": worker_function})
    ```
"""

import threading
from collections.abc import Callable
from typing import Any

from frontrun._marker_coordination import (
    MARKER_PATTERN,
    MarkerRegistry,
    ThreadCoordinator,
    finalize_marker_executor_run,
)
from frontrun._trace_marker_runtime import build_trace_function, run_traced_callable
from frontrun.common import (
    InterleavingResult,
    Schedule,
    Step,
    _call_sync_setup,
    _is_async_callable,
    _reject_deferred_sync_result,
    any_async,
    check_invariant,
)

__all__ = [
    "MARKER_PATTERN",
    "MarkerRegistry",
    "ThreadCoordinator",
    "TraceExecutor",
    "all_marker_schedules",
    "explore_marker_interleavings",
    "frontrun",
    "marker_schedule_strategy",
]


class _ThreadTraceExecutor:
    """Executes threads with interlaced execution according to a schedule.

    This is the main interface for the frontrun library. It sets up tracing
    for each thread and coordinates their execution.
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
        self.threads: list[threading.Thread] = []
        self.thread_errors: dict[str, BaseException] = {}

    def _create_trace_function(self, execution_name: str) -> Callable[[Any, str, Any], Any]:  # type: ignore[return-value]
        return build_trace_function(
            self.coordinator,
            self.marker_registry,
            execution_name,
            # Standalone-line markers (a `# frontrun: name` comment on its own
            # line) never emit their own `line` event, so they only fire via the
            # prev-line branch.  Enable it — matching the async executor — so
            # documented standalone markers gate the following line.  The branch
            # itself guards against firing on skipped executable lines, so inline
            # markers are unaffected.
            include_previous_line=True,
        )

    def _thread_wrapper(
        self,
        execution_name: str,
        target: Callable[..., None],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Wrapper function that sets up tracing for an execution unit.

        Args:
            execution_name: The name of this execution unit
            target: The function to execute
            args: Positional arguments for the target
            kwargs: Keyword arguments for the target
        """
        run_traced_callable(
            coordinator=self.coordinator,
            execution_name=execution_name,
            trace_function=self._create_trace_function(execution_name),
            # Fail closed if the body returns a coroutine/generator: a sync
            # classification (e.g. a lambda wrapping a coroutine call) would
            # otherwise discard it and report success without executing it.
            body=lambda: _reject_deferred_sync_result(target(*args, **kwargs), target),
            error_sink=self.thread_errors,
        )

    def run(
        self,
        execution_name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """Start a new thread with tracing enabled.

        Args:
            execution_name: The name for this execution unit (must match schedule)
            target: The function to execute in the thread
            args: Positional arguments for the target function
            kwargs: Keyword arguments for the target function
        """
        if kwargs is None:
            kwargs = {}

        thread = threading.Thread(
            target=self._thread_wrapper, args=(execution_name, target, args, kwargs), name=execution_name, daemon=True
        )
        self.threads.append(thread)
        thread.start()

    def wait(self, timeout: float | None = None):
        """Wait for all threads to complete.

        Args:
            timeout: Optional timeout in seconds

        Raises:
            TimeoutError: If threads don't complete within the timeout
            Any exception that occurred in a thread during execution
        """
        finalize_marker_executor_run(
            threads=self.threads,
            timeout=timeout,
            task_errors=self.thread_errors,
            coordinator=self.coordinator,
            timeout_message=lambda alive: (
                "Threads did not complete within timeout: " + ", ".join(t.name for t in alive)
            ),
        )

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.threads = []
        self.thread_errors = {}
        self.coordinator = ThreadCoordinator(self.schedule, deadlock_timeout=self.deadlock_timeout)
        self.marker_registry = MarkerRegistry()


# AsyncTraceExecutor.run requires a numeric timeout; the sync path accepts
# None and defers to _ThreadTraceExecutor.wait's own default, so callers who
# pass None from the dict-form API need a floor when the dict contains async
# coroutines.
_DEFAULT_ASYNC_TIMEOUT = 10.0


class TraceExecutor:
    """Facade over sync and async marker-based schedule execution.

    Pass a dict mapping thread/task names to zero-argument callables (sync
    or async).  ``run`` starts every worker and waits for them in one call:

    .. code-block:: python

       executor = TraceExecutor(schedule)
       executor.run({"t1": worker1, "t2": worker2})

    Async usage accepts the async-task mapping the same way:

    .. code-block:: python

       executor = TraceExecutor(schedule)
       executor.run({"task1": coro1, "task2": coro2})
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self._mode: str | None = None
        self._sync = _ThreadTraceExecutor(schedule, deadlock_timeout=deadlock_timeout)
        self._async: Any | None = None

    def _run_dict(self, tasks: dict[str, Callable[..., Any]], *, timeout: float | None = None) -> None:
        """Start all named sync threads from a dict and wait for them to complete.

        Args:
            tasks: Mapping of execution-unit name to a zero-argument callable.
            timeout: Optional total timeout in seconds.

        Raises:
            ValueError: If *tasks* is empty.
            TypeError: If any value in *tasks* is not callable.
            TimeoutError: If threads do not finish within *timeout*.
        """
        if not tasks:
            raise ValueError("TraceExecutor.run() received an empty dict — pass at least one {name: callable} entry.")
        for name, fn in tasks.items():
            if not callable(fn):
                raise TypeError(f"TraceExecutor.run(): value for {name!r} must be callable, got {type(fn).__name__!r}.")

        if self._mode == "async":
            raise TypeError("TraceExecutor is already running in async mode; cannot switch to sync dict form.")
        self._mode = "sync"

        for name, fn in tasks.items():
            self._sync.run(name, fn)
        self._sync.wait(timeout=timeout)

    def run(self, tasks: dict[str, Callable[..., Any]], *, timeout: float | None = None) -> None:
        """Run all named workers under schedule control and wait for completion.

        Pass a dict mapping thread/task names to zero-argument callables.
        ``run`` starts every worker and joins them in a single call::

            executor.run({"thread1": fn1, "thread2": fn2}, timeout=5.0)

        Async coroutines in the dict are dispatched to the async executor.

        Args:
            tasks: Mapping of execution-unit name to a zero-argument callable.
            timeout: Total wait timeout in seconds.
        """
        if any_async(tasks.values()):
            if self._mode == "sync":
                raise TypeError("TraceExecutor is already running in sync mode")
            self._mode = "async"
            if self._async is None:
                from frontrun.async_trace_markers import AsyncTraceExecutor

                self._async = AsyncTraceExecutor(self.schedule, deadlock_timeout=self.deadlock_timeout)
            self._async.run(tasks, timeout=_DEFAULT_ASYNC_TIMEOUT if timeout is None else timeout)
            return

        # All values are sync callables → use the sync dict form.
        self._run_dict(tasks, timeout=timeout)

    def wait(self, timeout: float | None = None) -> None:
        if self._mode in (None, "async"):
            return
        self._sync.wait(timeout=timeout)

    def reset(self) -> None:
        self._mode = None
        self._sync.reset()
        if self._async is not None:
            self._async.reset()

    @property
    def threads(self) -> list[threading.Thread]:
        return self._sync.threads

    @property
    def thread_errors(self) -> dict[str, BaseException]:
        return self._sync.thread_errors

    @property
    def coordinator(self) -> ThreadCoordinator:
        return self._sync.coordinator

    @property
    def marker_registry(self) -> MarkerRegistry:
        return self._sync.marker_registry

    @property
    def task_errors(self) -> dict[str, Exception]:
        if self._async is None:
            return {}
        return self._async.task_errors


def frontrun(
    schedule: Schedule,
    threads: dict[str, Callable[..., None]],
    thread_args: dict[str, tuple[Any, ...]] | None = None,
    thread_kwargs: dict[str, dict[str, Any]] | None = None,
    timeout: float | None = None,
    deadlock_timeout: float = 5.0,
) -> "TraceExecutor":
    """Convenience function to run multiple threads with a schedule.

    Args:
        schedule: The Schedule defining execution order
        threads: Dictionary mapping execution unit names to their target functions
        thread_args: Optional dictionary mapping execution unit names to argument tuples
        thread_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Optional timeout for waiting
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).

    Returns:
        The TraceExecutor instance (useful for inspection)

    Example:
        ```python
        frontrun(
            schedule=Schedule([Step("t1", "marker1"), Step("t2", "marker1")]),
            threads={"t1": worker_func, "t2": worker_func},
        )
        ```
    """
    if thread_args is None:
        thread_args = {}
    if thread_kwargs is None:
        thread_kwargs = {}

    executor = TraceExecutor(schedule, deadlock_timeout=deadlock_timeout)

    wrapped: dict[str, Callable[..., Any]] = {}
    for execution_name, target in threads.items():
        t_args = thread_args.get(execution_name, ())
        t_kwargs = thread_kwargs.get(execution_name, {})
        if t_args or t_kwargs:
            fn: Callable[..., Any] = target
            a: tuple[Any, ...] = t_args
            kw: dict[str, Any] = t_kwargs

            if _is_async_callable(fn):

                async def _wrap_async(
                    _fn: Callable[..., Any] = fn,
                    _a: tuple[Any, ...] = a,
                    _kw: dict[str, Any] = kw,
                ) -> None:
                    await _fn(*_a, **_kw)

                wrapped[execution_name] = _wrap_async
                continue

            def _wrap_sync(
                _fn: Callable[..., Any] = fn,
                _a: tuple[Any, ...] = a,
                _kw: dict[str, Any] = kw,
            ) -> Any:
                # Preserve the return value so TraceExecutor's outer
                # fail-closed check can reject deferred coroutine/generator
                # bodies instead of certifying a run that executed nothing.
                return _fn(*_a, **_kw)

            wrapped[execution_name] = _wrap_sync
        else:
            wrapped[execution_name] = target

    executor.run(wrapped, timeout=timeout)
    return executor


# ---------------------------------------------------------------------------
# Property-based marker schedule generation
# ---------------------------------------------------------------------------


def marker_schedule_strategy(
    threads: dict[str, list[str]],
) -> Any:
    """Hypothesis strategy that generates valid marker-level Schedule objects.

    A valid schedule interleaves each thread's markers while preserving
    their relative order within each thread.  This provides a much smaller
    search space than opcode-level exploration while still covering all
    meaningful interleavings at the marker granularity.

    Args:
        threads: Mapping from thread/execution names to their ordered list
            of marker names.  Example::

                {"t1": ["read", "write"], "t2": ["read", "write"]}

    Returns:
        A Hypothesis strategy producing :class:`Schedule` objects.

    Example::

        from hypothesis import given
        from frontrun.trace_markers import marker_schedule_strategy

        @given(schedule=marker_schedule_strategy(
            threads={"w1": ["read", "write"], "w2": ["read", "write"]},
        ))
        def test_no_lost_update(schedule):
            ...
    """
    from hypothesis import strategies as st

    thread_items = list(threads.items())

    @st.composite
    def gen(draw: Any) -> Schedule:
        remaining = {name: list(markers) for name, markers in thread_items}
        steps: list[Step] = []

        while any(remaining.values()):
            available = [name for name, m in remaining.items() if m]
            thread_name = draw(st.sampled_from(sorted(available)))
            marker = remaining[thread_name].pop(0)
            steps.append(Step(thread_name, marker))

        return Schedule(steps)

    return gen()


def all_marker_schedules(
    threads: dict[str, list[str]],
) -> list[Schedule]:
    """Enumerate ALL valid interleavings of thread markers.

    A valid interleaving places every thread's markers in the schedule
    while preserving their relative order within each thread.

    For *N* threads with marker counts *k1, k2, ..., kN*, the total
    number of valid interleavings is the multinomial coefficient::

        (k1 + k2 + ... + kN)! / (k1! * k2! * ... * kN!)

    Args:
        threads: Mapping from thread/execution names to their ordered list
            of marker names.

    Returns:
        A list of :class:`Schedule` objects covering every valid interleaving.

    Example::

        schedules = all_marker_schedules(
            threads={"t1": ["a", "b"], "t2": ["x", "y"]},
        )
        assert len(schedules) == 6  # C(4,2)
    """
    thread_items = sorted(threads.items())  # deterministic order
    schedules: list[Schedule] = []

    def _recurse(
        remaining: dict[str, list[str]],
        path: list[Step],
    ) -> None:
        available = [name for name, markers in remaining.items() if markers]
        if not available:
            schedules.append(Schedule(list(path)))
            return

        for name in available:
            marker = remaining[name][0]
            # Pop first marker, recurse, restore
            remaining[name] = remaining[name][1:]
            path.append(Step(name, marker))
            _recurse(remaining, path)
            path.pop()
            remaining[name] = [marker, *remaining[name]]

    initial = {name: list(markers) for name, markers in thread_items}
    _recurse(initial, [])
    return schedules


def explore_marker_interleavings(
    setup: Callable[..., Any],
    threads: dict[str, tuple[Callable[..., None], list[str]]],
    invariant: Callable[..., bool],
    *,
    stop_on_first: bool = True,
    deadlock_timeout: float = 5.0,
    timeout: float | None = 10.0,
) -> InterleavingResult:
    """Explore all marker-level interleavings and check an invariant.

    Generates every valid interleaving of the declared markers (preserving
    per-thread order), runs each one against real code via :class:`TraceExecutor`,
    and checks the invariant after each execution.

    This sits between manual trace markers (exact schedule, one interleaving)
    and bytecode exploration (random, enormous search space).  For *N* threads
    with a few markers each, the search space is small enough to explore
    exhaustively — giving completeness guarantees at the marker granularity.

    Args:
        setup: Factory producing fresh shared state for each execution.
        threads: Mapping from execution name to ``(target_fn, markers)`` where
            *target_fn* takes the setup result and *markers* is the ordered
            list of ``# frontrun:`` marker names that *target_fn* hits.
        invariant: Predicate on the shared state; returns True if correct.
        stop_on_first: Stop after finding the first invariant violation
            (default True).
        deadlock_timeout: Per-schedule deadlock detection timeout.
        timeout: Per-schedule join timeout.

    Returns:
        An :class:`~frontrun.common.InterleavingResult`.  The ``counterexample``
        field is a :class:`Schedule` (not a list of ints) when a violation is
        found.
    """
    marker_decl = {name: markers for name, (_, markers) in threads.items()}
    schedules = all_marker_schedules(marker_decl)

    num_explored = 0
    failures: list[tuple[int, Schedule]] = []
    first_explanation: str | None = None

    for i, schedule in enumerate(schedules):
        state = _call_sync_setup(setup)
        executor = TraceExecutor(schedule, deadlock_timeout=deadlock_timeout)

        runners: dict[str, Callable[..., Any]] = {}
        for exec_name, (target_fn, _markers) in threads.items():

            def _make_runner(s: Any = state, fn: Callable[..., None] = target_fn) -> None:
                _reject_deferred_sync_result(fn(s), fn)

            runners[exec_name] = _make_runner

        try:
            executor.run(runners, timeout=timeout)
            if executor.coordinator.current_step < len(schedule.steps) and not executor.coordinator.completed:
                remaining = schedule.steps[executor.coordinator.current_step :]
                step_strs = [f"Step({s.execution_name!r}, {s.marker_name!r})" for s in remaining]
                raise TimeoutError(
                    f"Schedule incomplete: {len(remaining)} step(s) were never reached: {', '.join(step_strs)}"
                )
        except TimeoutError as exc:
            # An unexecuted schedule proves nothing. Surface it as a failing
            # exploration instead of claiming exhaustive marker coverage.
            num_explored += 1
            explanation = f"TimeoutError: {exc}"
            if first_explanation is None:
                first_explanation = explanation
            failures.append((i, schedule))
            if stop_on_first:
                return InterleavingResult(
                    property_holds=False,
                    counterexample=schedule,
                    num_explored=num_explored,
                    unique_interleavings=num_explored,
                    explanation=explanation,
                )
            continue
        except Exception as exc:
            # Thread exceptions are violations, just like failed invariants.
            num_explored += 1
            explanation = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            if first_explanation is None:
                first_explanation = explanation
            failures.append((i, schedule))
            if stop_on_first:
                return InterleavingResult(
                    property_holds=False,
                    counterexample=schedule,
                    num_explored=num_explored,
                    unique_interleavings=num_explored,
                    explanation=explanation,
                )
            continue

        num_explored += 1

        invariant_failed, assertion_msg = check_invariant(invariant, state)
        if invariant_failed:
            explanation = (
                f"AssertionError: {assertion_msg}"
                if assertion_msg
                else f"Invariant violated under marker schedule: {schedule}"
            )
            if first_explanation is None:
                first_explanation = explanation
            failures.append((i, schedule))
            if stop_on_first:
                return InterleavingResult(
                    property_holds=False,
                    counterexample=schedule,
                    num_explored=num_explored,
                    unique_interleavings=num_explored,
                    explanation=explanation,
                )

    if failures:
        return InterleavingResult(
            property_holds=False,
            counterexample=failures[0][1],
            num_explored=num_explored,
            unique_interleavings=num_explored,
            failures=failures,  # type: ignore[arg-type]
            explanation=first_explanation,
        )

    return InterleavingResult(
        property_holds=True,
        num_explored=num_explored,
        unique_interleavings=num_explored,
    )
