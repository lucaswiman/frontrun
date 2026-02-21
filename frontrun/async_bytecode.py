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
    >>> from frontrun.async_bytecode import explore_interleavings, await_point
    >>>
    >>> class Counter:
    ...     def __init__(self):
    ...         self.value = 0
    ...     async def increment(self):
    ...         temp = self.value
    ...         await await_point()  # Yield control; race can happen here
    ...         self.value = temp + 1
    >>>
    >>> result = asyncio.run(explore_interleavings(
    ...     setup=lambda: Counter(),
    ...     tasks=[lambda c: c.increment(), lambda c: c.increment()],
    ...     invariant=lambda c: c.value == 2,
    ... ))
    >>> assert not result.property_holds  # race condition found!

The await_point() function marks explicit yield points where context switches
can occur. This is analogous to bytecode.py's opcode-level tracing, but for
async code the number of interleaving points is much smaller and more explicit.

In production async code, every `await` is a potential context switch point.
For testing, call `await await_point()` at each location where a race
condition could manifest.
"""

import asyncio
import contextvars
import random
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Optional

from frontrun.async_scheduler import InterleavedLoop
from frontrun.common import InterleavingResult

# Context variable to track the active scheduler and task ID
_scheduler_var: contextvars.ContextVar[Optional["AwaitScheduler"]] = contextvars.ContextVar("_scheduler", default=None)
_task_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("_task_id", default=None)


async def await_point():
    """Yield to the scheduler at an await point.

    Call this at every point where a context switch could happen in your
    async code. This is the async equivalent of a bytecode opcode — the
    atomic unit of interleaving.

    In typical async code, every `await` statement is a potential context
    switch point. For testing race conditions, replace strategic awaits
    with `await await_point()` to allow the scheduler to control ordering.

    If no scheduler is active (i.e., not running under AsyncBytecodeShuffler),
    this function returns immediately without blocking.
    """
    scheduler = _scheduler_var.get()
    if scheduler is not None:
        task_id = _task_id_var.get()
        if task_id is not None:
            await scheduler.pause(task_id)


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

    @property
    def had_error(self) -> bool:
        """Check if an error occurred during execution."""
        return self._error is not None


class AsyncBytecodeShuffler:
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
            await self.scheduler.run_all(task_funcs, timeout=timeout)  # type: ignore[arg-type]
        except TimeoutError:
            pass  # match original behavior: swallow timeout in runner


@asynccontextmanager
async def controlled_interleaving(
    schedule: list[int], num_tasks: int = 2
) -> AsyncGenerator[AsyncBytecodeShuffler, None]:
    """Context manager for running async code under a specific interleaving.

    Args:
        schedule: List of task indices controlling await-point execution order.
        num_tasks: Number of tasks.

    Yields:
        AsyncBytecodeShuffler runner.

    Example:
        >>> async with controlled_interleaving([0, 1, 0, 1], num_tasks=2) as runner:
        ...     await runner.run([coro1, coro2])
    """
    scheduler = AwaitScheduler(schedule, num_tasks)
    runner = AsyncBytecodeShuffler(scheduler)
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

    Returns:
        The state object after execution.
    """
    scheduler = AwaitScheduler(schedule, len(tasks), deadlock_timeout=deadlock_timeout)
    runner = AsyncBytecodeShuffler(scheduler)

    state = setup()
    funcs: list[Callable[..., Coroutine[Any, Any, None]]] = [lambda s=state, t=t: t(s) for t in tasks]  # type: ignore[assignment]

    try:
        await runner.run(funcs, timeout=timeout)
    except asyncio.TimeoutError:
        pass

    return state


async def explore_interleavings(
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    invariant: Callable[[Any], bool],
    max_attempts: int = 200,
    max_ops: int = 100,
    timeout_per_run: float = 5.0,
    seed: int | None = None,
    deadlock_timeout: float = 5.0,
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

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct schedule orderings were observed.
    """
    rng = random.Random(seed)
    num_tasks = len(tasks)
    result = InterleavingResult(property_holds=True, num_explored=0)
    seen_schedule_hashes: set[int] = set()

    for _ in range(max_attempts):
        num_rounds = rng.randint(1, max(1, max_ops // num_tasks))
        schedule: list[int] = []
        for _ in range(num_rounds):
            round_perm = list(range(num_tasks))
            rng.shuffle(round_perm)
            schedule.extend(round_perm)

        state = await run_with_schedule(
            schedule, setup, tasks, timeout=timeout_per_run, deadlock_timeout=deadlock_timeout
        )
        result.num_explored += 1
        seen_schedule_hashes.add(hash(tuple(schedule)))

        if not invariant(state):
            result.property_holds = False
            result.counterexample = schedule
            result.unique_interleavings = len(seen_schedule_hashes)
            return result

    result.unique_interleavings = len(seen_schedule_hashes)
    return result


def schedule_strategy(num_tasks: int, max_ops: int = 100) -> Any:  # type: ignore[name-defined]
    """Hypothesis strategy for generating fair await-point schedules.

    Generates schedules as a sequence of rounds, where each round is a
    random permutation of all task indices.  This guarantees every task
    gets exactly the same number of scheduling slots, preventing starvation.

    For use with hypothesis @given decorator in your own tests:

        >>> from hypothesis import given
        >>> from frontrun.async_bytecode import schedule_strategy, run_with_schedule
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
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    max_rounds = max(1, max_ops // num_tasks)
    tasks = list(range(num_tasks))

    @st.composite  # type: ignore[attr-defined]
    def _fair_schedule(draw: st.DrawFn) -> list[int]:  # type: ignore[attr-defined,name-defined]
        num_rounds = draw(st.integers(min_value=1, max_value=max_rounds))  # type: ignore[attr-defined]
        schedule: list[int] = []
        for _ in range(num_rounds):
            schedule.extend(draw(st.permutations(tasks)))  # type: ignore[attr-defined]
        return schedule

    return _fair_schedule()
