"""
Await-point-level deterministic async concurrency testing.

Uses the shared InterleavedLoop abstraction to control which async task
resumes at each await point, enabling fine-grained control over task
interleaving.

This pairs naturally with property-based testing: rather than specifying exact
schedules, generate random interleavings and check that invariants hold (or
that bugs can be found).

The core insight: in async Python, context switches happen ONLY at await
points. The event loop is single-threaded. By controlling which task resumes
at each await point, we can explore the full space of possible interleavings —
and there are far fewer of them than in threaded code.

Example — find a race condition with random schedule exploration:

    >>> import asyncio
    >>> from frontrun import explore_interleavings
    >>>
    >>> class Counter:
    ...     def __init__(self):
    ...         self.value = 0
    ...     async def increment(self):
    ...         temp = self.value
    ...         await asyncio.sleep(0)  # any natural await is a scheduling point
    ...         self.value = temp + 1
    >>>
    >>> result = asyncio.run(explore_interleavings(
    ...     setup=lambda: Counter(),
    ...     tasks=[lambda c: c.increment(), lambda c: c.increment()],
    ...     invariant=lambda c: c.value == 2,
    ... ))
    >>> assert result.property_holds, result.explanation  # fails — lost update!

Any natural ``await`` in user code is a scheduling point.  ``await_point()``
is still available as an explicit extra yield when a test wants to force
an additional checkpoint.
"""

import asyncio
import random
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager
from typing import Any

# Lazy import for async SQL patching — shared with async_dpor.py
from frontrun._async_autopause import (
    _in_scheduler_pause,
    _scheduler_var,
    _task_id_var,
    await_point,  # noqa: F401
    wrap_auto_paused_tasks,
)
from frontrun._random_schedules import fair_schedule_strategy, random_round_robin_schedule
from frontrun._threaded_runner import PatchScope
from frontrun.async_dpor import _sql_async_available, patch_sql_async, unpatch_sql_async
from frontrun.async_scheduler import InterleavedLoop
from frontrun.common import (
    DEPRECATION_MESSAGES,
    InterleavingResult,
    check_invariant,
    check_serializability_violation,
    deprecate,
)


class AwaitScheduler(InterleavedLoop):
    """Controls async task execution at await-point granularity.

    The schedule is a list of task indices. Each entry means "let this
    task resume from its next await point." When the schedule is
    exhausted, all tasks run freely to completion.

    Built on the shared InterleavedLoop abstraction, using index-based
    scheduling as its policy.
    """

    def __init__(self, schedule: list[int], num_tasks: int, *, deadlock_timeout: float = 5.0):
        super().__init__(deadlock_timeout=deadlock_timeout)
        self.schedule = schedule
        self.num_tasks = num_tasks
        self._index = 0

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        # Skip past done tasks
        while self._index < len(self.schedule):
            if self.schedule[self._index] in self._tasks_done:
                self._index += 1
                continue
            break

        if self._index >= len(self.schedule):
            self._finished = True
            return True

        return self.schedule[self._index] == task_id

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        if self._index < len(self.schedule):
            self._index += 1

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        needed = self.schedule[self._index] if self._index < len(self.schedule) else "?"
        self._error = TimeoutError(
            f"Deadlock: schedule wants task {needed} at index {self._index}/{len(self.schedule)}"
        )
        self._condition.notify_all()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)

    def _cleanup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(None)
        _task_id_var.set(None)

    async def pause(self, task_id: Any, marker: Any = None) -> None:
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await asyncio.sleep(0)
            await super().pause(task_id, marker)
        finally:
            _in_scheduler_pause.set(depth)

    @property
    def had_error(self) -> bool:
        """Check if an error occurred during execution."""
        return self._error is not None


class AsyncShuffler:
    """Run concurrent async functions with await-point-level interleaving control.

    Creates asyncio tasks for each function and delegates to the
    AwaitScheduler (an InterleavedLoop subclass) for execution and
    context setup.
    """

    def __init__(self, scheduler: AwaitScheduler):
        self.scheduler = scheduler
        self.errors: dict[int, Exception] = {}

    async def run(
        self,
        funcs: list[Callable[..., Coroutine[Any, Any, None]]],
        args: list[tuple[Any, ...]] | None = None,
        kwargs: list[dict[str, Any]] | None = None,
        timeout: float = 10.0,
    ):
        """Run async functions concurrently with controlled interleaving.

        Args:
            funcs: One async callable per task.
            args: Per-task positional args.
            kwargs: Per-task keyword args.
            timeout: Max wait time for all tasks.
        """
        if args is None:
            args = [() for _ in funcs]
        if kwargs is None:
            kwargs = [{} for _ in funcs]

        task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
            i: (lambda f=func, a=a, kw=kw: f(*a, **kw))  # type: ignore[assignment]
            for i, (func, a, kw) in enumerate(zip(funcs, args, kwargs))
        }

        try:
            await self.scheduler.run_all(wrap_auto_paused_tasks(task_funcs, self.scheduler), timeout=timeout)
        except TimeoutError:
            pass  # match original behavior: swallow timeout in runner


@contextmanager
def _patch_async_runtime(*, detect_sql: bool = False, patch_sleep: bool = False):
    with PatchScope() as patch_scope:
        patch_scope.add(patch_sql_async, unpatch_sql_async, enabled=detect_sql and _sql_async_available)
        if patch_sleep:
            from frontrun.async_dpor import _patch_asyncio_sleep, _unpatch_asyncio_sleep

            patch_scope.add(_patch_asyncio_sleep, _unpatch_asyncio_sleep)
        yield


@asynccontextmanager
async def controlled_interleaving(schedule: list[int], num_tasks: int = 2) -> AsyncGenerator[AsyncShuffler, None]:
    """Context manager for running async code under a specific interleaving.

    Args:
        schedule: List of task indices controlling await-point execution order.
        num_tasks: Number of tasks.

    Yields:
        AsyncShuffler runner.

    Example:
        >>> async with controlled_interleaving([0, 1, 0, 1], num_tasks=2) as runner:
        ...     await runner.run([coro1, coro2])
    """
    scheduler = AwaitScheduler(schedule, num_tasks)
    runner = AsyncShuffler(scheduler)
    yield runner


# ---------------------------------------------------------------------------
# Property-based testing
# ---------------------------------------------------------------------------


async def run_with_schedule(
    schedule: list[int],
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    timeout: float = 5.0,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
) -> Any:
    """Run one async interleaving and return the state object.

    Args:
        schedule: Await-point-level schedule (list of task indices).
        setup: Returns fresh shared state.
        tasks: Async callables that each receive the state as their argument.
        timeout: Max seconds.
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).
        detect_sql: If ``True``, patch async DBAPI drivers (aiosqlite,
            psycopg AsyncCursor, aiomysql, asyncpg) to intercept SQL
            and report table-level conflicts.

    Returns:
        The state object after execution.
    """
    with _patch_async_runtime(detect_sql=detect_sql):
        scheduler = AwaitScheduler(schedule, len(tasks), deadlock_timeout=deadlock_timeout)
        runner = AsyncShuffler(scheduler)

        state = setup()
        funcs: list[Callable[..., Coroutine[Any, Any, None]]] = [lambda s=state, t=t: t(s) for t in tasks]  # type: ignore[assignment]

        await runner.run(funcs, timeout=timeout)

        return state


async def explore_async_random(
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    invariant: Callable[[Any], bool],
    max_attempts: int = 200,
    max_ops: int = 100,
    timeout_per_run: float = 5.0,
    seed: int | None = None,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
    trace_packages: list[str] | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[Any], Any] | bool = False,
    error_on_any_race: bool = False,
    total_timeout: float | None = None,
) -> InterleavingResult:
    """Search for async interleavings that violate an invariant.

    Generates random await-point-level schedules and tests whether the
    invariant holds under each one. If a violation is found, returns
    immediately with the counterexample schedule.

    This is the async analogue of property-based testing for concurrency:
    instead of generating random *inputs*, we generate random *interleavings*
    and check that the result satisfies an invariant.

    Note: max_ops defaults to 100 (vs 300 for bytecode.py) because async
    code has far fewer interleaving points than threaded bytecode execution.
    Each await_point() call represents a much coarser-grained checkpoint.

    Args:
        setup: Returns fresh shared state for each attempt.
        tasks: Async callables that each receive the state as their argument.
        invariant: Predicate on the state. Returns True if the property holds.
        max_attempts: How many random interleavings to try.
        max_ops: Maximum schedule length per attempt.
        timeout_per_run: Timeout for each individual run.
        seed: Optional RNG seed for reproducibility.
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).
        detect_sql: If ``True``, patch async DBAPI drivers (aiosqlite,
            psycopg AsyncCursor, aiomysql, asyncpg) to intercept SQL
            and report table-level conflicts.
        trace_packages: Accepted for API compatibility but not used.
            The async shuffler operates at await-point granularity and
            does not perform file-level tracing.

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct schedule orderings were observed.
    """
    if error_on_any_race:
        raise ValueError("error_on_any_race requires DPOR (use explore_async_dpor instead)")

    from frontrun._dpor_core import compute_serializable_baseline_async

    serial_valid_states, serial_hash_fn = await compute_serializable_baseline_async(
        setup, tasks, serializable_invariant
    )

    with _patch_async_runtime(detect_sql=detect_sql, patch_sleep=patch_sleep):
        import time

        rng = random.Random(seed)
        num_tasks = len(tasks)
        result = InterleavingResult(property_holds=True, num_explored=0)
        seen_schedule_hashes: set[int] = set()
        total_deadline = time.monotonic() + total_timeout if total_timeout is not None else None

        for _ in range(max_attempts):
            if total_deadline is not None and time.monotonic() > total_deadline:
                break
            schedule = random_round_robin_schedule(rng, num_tasks, max_ops)

            state = await run_with_schedule(
                schedule, setup, tasks, timeout=timeout_per_run, deadlock_timeout=deadlock_timeout
            )
            result.num_explored += 1
            seen_schedule_hashes.add(hash(tuple(schedule)))

            # --- serializable_invariant check ---
            if serial_valid_states is not None:
                explanation = check_serializability_violation(
                    state, serial_valid_states, serial_hash_fn, result.num_explored
                )
                if explanation is not None:
                    result.property_holds = False
                    result.counterexample = schedule
                    result.unique_interleavings = len(seen_schedule_hashes)
                    result.explanation = explanation
                    return result

            invariant_failed, assertion_msg = check_invariant(invariant, state)
            if invariant_failed:
                result.property_holds = False
                result.counterexample = schedule
                result.unique_interleavings = len(seen_schedule_hashes)
                if assertion_msg:
                    result.explanation = f"AssertionError: {assertion_msg}"
                return result

        result.unique_interleavings = len(seen_schedule_hashes)
        return result


explore_interleavings = deprecate(explore_async_random, DEPRECATION_MESSAGES["explore_async_interleavings"])
explore_interleavings.__doc__ = (
    "Deprecated alias for :func:`explore_async_random`.\n\n"
    ".. deprecated:: 0.5\n"
    "    ``explore_interleavings`` (async form) will be removed in 0.6. Use\n"
    "    :func:`frontrun.explore` with ``strategy='random'`` and async worker\n"
    "    functions instead."
)


def schedule_strategy(num_tasks: int, max_ops: int = 100) -> Any:  # type: ignore[name-defined]
    """Hypothesis strategy for generating fair await-point schedules.

    Generates schedules as a sequence of rounds, where each round is a
    random permutation of all task indices.  This guarantees every task
    gets exactly the same number of scheduling slots, preventing starvation.

    For use with hypothesis @given decorator in your own tests:

        >>> from hypothesis import given
        >>> from frontrun.async_shuffler import schedule_strategy, run_with_schedule
        >>> import asyncio
        >>>
        >>> @given(schedule=schedule_strategy(2))
        ... def test_my_invariant(schedule):
        ...     state = asyncio.run(run_with_schedule(schedule, setup, tasks))
        ...     assert state.value == expected

    Note: max_ops defaults to 100 (vs 300 for bytecode.py) because async
    code has far fewer interleaving points. Each schedule entry corresponds
    to one await_point() call, not one bytecode opcode.
    """
    return fair_schedule_strategy(num_tasks, max_ops)
