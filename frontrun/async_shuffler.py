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
    >>> import frontrun
    >>>
    >>> class Counter:
    ...     def __init__(self):
    ...         self.value = 0
    ...     async def increment(self):
    ...         temp = self.value
    ...         await asyncio.sleep(0)  # any natural await is a scheduling point
    ...         self.value = temp + 1
    >>>
    >>> result = asyncio.run(frontrun.explore_async_random(
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
import warnings
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
from frontrun._async_cooperative import _real_asyncio_sleep
from frontrun._random_schedules import burst_round, fair_schedule_strategy, random_round_robin_schedule
from frontrun._threaded_runner import PatchScope
from frontrun._virtual_clock import (
    ClockConfig,
    ClockMode,
    DeadlineCoordinator,
    VirtualClock,
    _real_monotonic,
    clock_context,
    patch_time,
    unpatch_time,
)
from frontrun.async_dpor import (
    _sql_async_available,
    patch_sql_async,
    unpatch_sql_async,
)
from frontrun.async_scheduler import (
    InterleavedLoop,
    SchedulerTimeoutError,
    _pin_loop_time,
    frontrun_wait_for,
)
from frontrun.common import (
    InterleavingResult,
    check_invariant,
    check_serializability_violation,
)

#: How long the schedule must be quiescent (no pauses, no schedule progress,
#: no task completions) before a sleeper concludes that the remaining tasks
#: are parked on something the scheduler cannot see (e.g. an unpatched
#: asyncio.Lock) and autojumps the virtual clock.
_QUIESCENCE_SLICE = 0.01


class AwaitScheduler(InterleavedLoop):
    """Controls async task execution at await-point granularity.

    The schedule is a list of task indices. Each entry means "let this
    task resume from its next await point." When the sampled prefix is
    exhausted, it is extended deterministically so tasks remain controlled.

    Built on the shared InterleavedLoop abstraction, using index-based
    scheduling as its policy.
    """

    def __init__(
        self,
        schedule: list[int],
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        detect_sql: bool = False,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        max_ops: int = 0,
        extension_seed: int | None = None,
    ):
        super().__init__(deadlock_timeout=deadlock_timeout)
        self.schedule = schedule
        self.num_tasks = num_tasks
        self._index = 0
        self._extend_enabled = max_ops > 0
        self._max_ops = max_ops if max_ops > 0 else len(schedule) * 10 + 10_000
        seed = extension_seed if extension_seed is not None else hash(tuple(schedule)) & 0xFFFFFFFF
        self._extend_rng = random.Random(seed)
        self._max_ops_exhausted = False
        self._detect_sql = detect_sql
        # Virtual clock (ideas/virtual_clock.md), mirroring the sync
        # OpcodeScheduler: schedule entries landing on a sleeping task are
        # skipped ("virtual") or advance the clock to that task's deadline
        # ("explored" — the random "maybe advance time" branch); when every
        # live task is deadline-blocked, sleep_until autojumps.
        self.virtual_clock = virtual_clock
        self.clock_mode = clock_mode
        self._deadlines = DeadlineCoordinator()
        self._sleepers: dict[int, float] = {}
        # Tasks parked in a virtual timed wait with no scheduler-side loop
        # (asyncio.wait_for on a bare future/task).  The schedule skips them
        # like sleepers; their pending timeout deadline in _deadlines is the
        # only thing that can wake them, and the token's on_fire unparks.
        self._timed_parked: set[int] = set()
        # Table/row accesses observed via SQL interception, in arrival order.
        # Exposed so callers can inspect cross-task table conflicts.
        self.sql_accesses: list[tuple[int, str, str]] = []

    def _extend_schedule(self) -> bool:
        """Append a deterministic burst round without exceeding max_ops."""
        active = [task_id for task_id in range(self.num_tasks) if task_id not in self._tasks_done]
        remaining = self._max_ops - len(self.schedule)
        if not active or remaining <= 0:
            self._max_ops_exhausted = bool(active)
            return False
        self.schedule.extend(burst_round(self._extend_rng, active)[:remaining])
        return True

    # -- Virtual clock ---------------------------------------------------

    def _advance_clock_to(self, target: float) -> bool:
        """Jump the clock to *target* and wake every due sleeper.

        Caller must hold ``self._condition``.  Returns True when a
        ``timeout``-kind deadline fired: its effect (the waiter cancelling the
        timed-out awaitable) only lands once the event loop next runs, so an
        autojump caller must yield to the loop before advancing further — a
        second synchronous hop would let the doomed task's own later deadline
        fire first, completing work a real timeout would have cancelled.
        """
        clock = self.virtual_clock
        if clock is None:
            return False
        fired_timeout = False
        for event in self._deadlines.advance_to(clock, target):
            if event.kind == "sleep":
                self._sleepers.pop(event.actor_id, None)
            elif event.kind == "timeout":
                fire = getattr(event.token, "fire", None)
                if fire is not None:
                    fire()
                fired_timeout = True
        return fired_timeout

    def add_timeout_deadline(self, task_id: int, deadline: float, token: object) -> None:
        self._deadlines.add_timeout(task_id, deadline, token)

    def remove_timeout_deadline(self, task_id: int, token: object) -> None:
        self._deadlines.cancel(task_id, token)

    def park_timed_wait(self, task_id: int) -> None:
        """Register *task_id* as parked in a virtual timed wait.

        Called by the virtual ``asyncio.wait_for`` wrapper for waits on bare
        futures/tasks: the parked task has no further scheduling points, so
        the schedule must skip its entries (see ``should_proceed``) and the
        clock advance is what wakes it.
        """
        self._timed_parked.add(task_id)

    def unpark_timed_wait(self, task_id: int) -> None:
        self._timed_parked.discard(task_id)

    def _advance_clock_for_parked_if_blocked(self) -> None:
        """Advance to the next deadline when only parked/sleeping tasks remain.

        Caller must hold ``self._condition``.  Sleepers drive their own
        autojump loop inside ``sleep_until``, but a task parked in a virtual
        ``wait_for`` has no scheduler-side loop — so the transitions that can
        leave only blocked tasks alive (the park itself, or another task
        finishing) must advance the clock here, or the run stalls until the
        wall-clock watchdog rescues it a ``deadlock_timeout`` later.
        """
        if self.virtual_clock is None or not self._timed_parked:
            return
        alive = [t for t in range(self.num_tasks) if t not in self._tasks_done]
        if not alive or not all(t in self._timed_parked or t in self._sleepers for t in alive):
            return
        next_deadline = self._deadlines.next_deadline()
        if next_deadline is not None:
            self._advance_clock_to(next_deadline)

    async def kick_stalled_schedule(self, task_id: int) -> None:
        """Wake schedule progress after *task_id* parked itself in a timed wait.

        Tasks waiting in ``pause()`` for the parked task's schedule entries
        must re-check ``should_proceed`` (which now skips those entries), and
        if every live task is blocked only the clock can move.
        """
        async with self._condition:
            if self._error:
                return
            self._advance_clock_for_parked_if_blocked()
            self._condition.notify_all()

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task done; if only parked tasks remain, advance the clock."""
        self._progress += 1
        async with self._condition:
            self._tasks_done.add(task_id)
            self._advance_clock_for_parked_if_blocked()
            self._condition.notify_all()

    def _advance_virtual_deadline_for_idle(self) -> bool:
        if self.virtual_clock is None:
            return False
        next_deadline = self._deadlines.next_deadline()
        if next_deadline is None:
            return False
        self._advance_clock_to(next_deadline)
        return True

    async def sleep_until(self, task_id: int, deadline: float) -> None:
        """Block *task_id* until the virtual clock reaches *deadline*."""
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await _real_asyncio_sleep(0)
            self._progress += 1
            async with self._condition:
                if self._error:
                    return
                if self._finished:
                    # Schedule exhausted: the free-running sleep must still
                    # resolve virtually — advance to its deadline (firing
                    # earlier due deadlines in order) instead of returning
                    # with the clock frozen (a silently truncated sleep).
                    self._advance_clock_to(deadline)
                    self._condition.notify_all()
                    return
                self._sleepers[task_id] = deadline
                self._deadlines.add_sleep(task_id, deadline, wake_id=None)
                self._condition.notify_all()
                try:
                    wait_started = _real_monotonic()
                    while task_id in self._sleepers:
                        if self._error:
                            return
                        if self._finished:
                            # See the pre-registration check above.
                            self._advance_clock_to(deadline)
                            self._condition.notify_all()
                            return
                        alive = [t for t in range(self.num_tasks) if t not in self._tasks_done]
                        if alive and all(t in self._sleepers or t in self._timed_parked for t in alive):
                            # Every live task is asleep or parked in a timed
                            # wait: only time can move.
                            next_deadline = self._deadlines.next_deadline()
                            if next_deadline is None or not self._advance_clock_to(next_deadline):
                                self._condition.notify_all()
                                continue
                            # A timeout fired at this hop.  Do NOT advance
                            # again yet: the waiter it woke (e.g. the virtual
                            # wait_for wrapper) cancels its timed-out task
                            # only when the event loop next runs, and this
                            # sleeper may BE that task.  Fall through to the
                            # condition wait, which releases the lock and
                            # yields — the cancellation lands there (or, if
                            # nothing reacts, the quiescence fallback resumes
                            # the advance).
                            self._condition.notify_all()
                        snapshot = (self._progress, self._index, len(self._tasks_done))
                        quiescence_slice = min(_QUIESCENCE_SLICE, max(0.001, self.deadlock_timeout / 2.0))
                        try:
                            await frontrun_wait_for(self._condition.wait(), timeout=quiescence_slice)
                        except asyncio.TimeoutError:
                            if (self._progress, self._index, len(self._tasks_done)) == snapshot:
                                # Quiescent: the remaining tasks are parked on
                                # something the scheduler can't see (e.g. an
                                # unpatched asyncio.Lock whose holder is this
                                # sleeper).  Advancing time is the only way
                                # forward; without it the run dies by wall
                                # timeout — a false deadlock.
                                next_deadline = self._deadlines.next_deadline()
                                if next_deadline is not None:
                                    self._advance_clock_to(next_deadline)
                                self._condition.notify_all()
                                continue
                            if _real_monotonic() - wait_started > self.deadlock_timeout:
                                self._error = SchedulerTimeoutError(
                                    f"Deadlock: task {task_id} sleeping until t={deadline} was never woken"
                                )
                                self._condition.notify_all()
                                return
                finally:
                    self._sleepers.pop(task_id, None)
                    self._deadlines.cancel_sleep(task_id)
        finally:
            _in_scheduler_pause.set(depth)

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        # Skip past done tasks (and resolve entries for sleeping tasks)
        while self._index < len(self.schedule):
            entry = self.schedule[self._index]
            if entry in self._tasks_done:
                self._index += 1
                continue
            if self.virtual_clock is not None and (entry in self._sleepers or entry in self._timed_parked):
                if self.clock_mode == "explored":
                    # "Maybe advance": the random schedule picked a sleeping
                    # (or timed-parked) task — let time pass toward its
                    # deadline; the woken task then consumes this entry.  Each
                    # speculative hop is clamped to the *earliest* pending
                    # deadline so an earlier timed wait fires (and wakes its
                    # waiter) at its own clock value, never at a later
                    # sleeper's target.  Convergence is automatic: control
                    # returns to the event loop after each hop — letting the
                    # woken waiter run first — and the next should_proceed
                    # call re-advances until the scheduled sleeper's own
                    # deadline is reached.  A timed-parked entry has no
                    # _sleepers deadline; its own timeout is pending in the
                    # coordinator, so the earliest-deadline clamp is the hop.
                    target = self._sleepers.get(entry)
                    next_deadline = self._deadlines.next_deadline()
                    if target is None:
                        target = next_deadline
                    elif next_deadline is not None:
                        target = min(target, next_deadline)
                    if target is None:
                        # No pending deadline (already fired): the task is
                        # waking — skip the stale slot.
                        self._index += 1
                        continue
                    self._advance_clock_to(target)
                    self._condition.notify_all()
                    break
                # Autojump semantics: a sleeping/parked task cannot run before
                # the clock advances; skip its slot.
                self._index += 1
                continue
            break

        if self._index >= len(self.schedule):
            if not self._extend_enabled:
                self._finished = True
                return True
            if not self._extend_schedule():
                self._finished = True
                return True

        return self.schedule[self._index] == task_id

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        if self._index < len(self.schedule):
            self._index += 1

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        needed = self.schedule[self._index] if self._index < len(self.schedule) else "?"
        self._error = SchedulerTimeoutError(
            f"Deadlock: schedule wants task {needed} at index {self._index}/{len(self.schedule)}"
        )
        self._condition.notify_all()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)
        if not self._detect_sql:
            return
        # Install a task-aware DPOR context + IO reporter so patched async SQL
        # cursors report table/row accesses.  The random shuffler has no DPOR
        # engine, so report_and_wait / row-lock methods are no-ops; scheduling
        # already happens at await points.
        from frontrun._io_detection import (
            set_dpor_scheduler_task,
            set_dpor_thread_id_task,
            set_io_reporter,
            set_tx_store_task,
        )

        set_dpor_scheduler_task(self)
        set_dpor_thread_id_task(task_id)
        set_tx_store_task()

        def _io_reporter(resource_id: str, kind: str) -> None:
            current = _task_id_var.get()
            self.sql_accesses.append((current if current is not None else task_id, resource_id, kind))

        set_io_reporter(_io_reporter)

    def _cleanup_task_context(self, task_id: Any) -> None:
        if self._detect_sql:
            from frontrun._io_detection import set_dpor_scheduler_task, set_dpor_thread_id_task, set_io_reporter

            set_dpor_scheduler_task(None)
            set_dpor_thread_id_task(None)
            # Reporter is per-OS-thread (shared by all tasks); only clear when
            # all tasks are done so remaining tasks keep reporting.
            if len(self._tasks_done) + 1 >= self.num_tasks:
                set_io_reporter(None)
        _scheduler_var.set(None)
        _task_id_var.set(None)

    # -- DPOR-compat no-ops (the random shuffler has no DPOR engine) ------

    def report_and_wait(self, _frame: Any, _thread_id: int) -> bool:
        """No-op scheduling hook called by async SQL interception.

        Scheduling already happens at the patched cursor's own await points,
        so this just lets the SQL call proceed.
        """
        return True

    def acquire_row_locks(self, _thread_id: int, _resource_ids: list[str]) -> None:
        """No-op: random exploration does not arbitrate SQL row locks."""

    def release_row_locks(self, _thread_id: int, _resources: object = None) -> None:
        """No-op: random exploration does not arbitrate SQL row locks."""

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
        # Whether the most recent run was cut short by a timeout/deadlock.
        # When True the resulting state is partial/cancelled and must NOT be
        # evaluated as a normal completion.
        self.timed_out = False

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

        self.timed_out = False
        try:
            await self.scheduler.run_all(
                wrap_auto_paused_tasks(task_funcs, self.scheduler),
                timeout=timeout,
                # Detect deadlocks formed entirely outside pause() — every
                # unfinished task blocked on a stock asyncio primitive — so a
                # genuine deadlock sets scheduler._error instead of surfacing
                # as a bare wall-clock timeout that the exploration loop must
                # skip as inconclusive (slow-but-correct runs look identical).
                detect_external_deadlock=True,
            )
        except SchedulerTimeoutError:
            # A timeout means tasks were cancelled mid-flight, so the state is
            # partial. Record it instead of evaluating the invariant.
            self.timed_out = True


@contextmanager
def _patch_async_runtime(
    *, detect_sql: bool = False, patch_sleep: bool = False, virtual_time: bool = False, pin_loop_time: Any = None
):
    restore_loop_time: Callable[[], None] | None = None
    with PatchScope() as patch_scope:
        patch_scope.add(patch_sql_async, unpatch_sql_async, enabled=detect_sql and _sql_async_available)
        if patch_sleep:
            from frontrun._async_virtual_timeouts import (
                _patch_asyncio_sleep,
                _patch_asyncio_timeouts,
                _unpatch_asyncio_sleep,
                _unpatch_asyncio_timeouts,
            )

            patch_scope.add(_patch_asyncio_sleep, _unpatch_asyncio_sleep)
            patch_scope.add(_patch_asyncio_timeouts, _unpatch_asyncio_timeouts, enabled=virtual_time)
        patch_scope.add(patch_time, unpatch_time, enabled=virtual_time)
        # Pin the loop's own clock to real monotonic time while time.monotonic
        # is patched (see the matching comment in async_dpor._explore_async_dpor).
        if pin_loop_time is not None:
            restore_loop_time = _pin_loop_time(pin_loop_time)
        try:
            yield
        finally:
            if restore_loop_time is not None:
                restore_loop_time()


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
        state, runner = await _run_with_schedule_status(
            schedule, setup, tasks, timeout=timeout, deadlock_timeout=deadlock_timeout, detect_sql=detect_sql
        )
        if runner.scheduler._error is not None:
            raise runner.scheduler._error
        if runner.timed_out:
            raise SchedulerTimeoutError("Async schedule replay timed out before all tasks completed")
        return state


async def _run_with_schedule_status(
    schedule: list[int],
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    timeout: float = 5.0,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
) -> tuple[Any, AsyncShuffler]:
    """Run one interleaving and return ``(state, runner)``.

    The runner exposes ``timed_out`` / ``scheduler.had_error`` so callers can
    tell whether the run completed normally or was cut short by a
    timeout/deadlock (in which case ``state`` is partial and must not be
    evaluated as a normal completion).  Assumes the async runtime is already
    patched by the caller.
    """
    scheduler = AwaitScheduler(schedule, len(tasks), deadlock_timeout=deadlock_timeout, detect_sql=detect_sql)
    runner = AsyncShuffler(scheduler)

    state = setup()
    funcs: list[Callable[..., Coroutine[Any, Any, None]]] = [lambda s=state, t=t: t(s) for t in tasks]  # type: ignore[assignment]

    await runner.run(funcs, timeout=timeout)

    return state, runner


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
    clock: ClockMode = "real",
    clock_diagnostics: bool = False,
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
        patch_sleep: If True (default), ``asyncio.sleep`` yields to the
            scheduler instead of waiting.  Required for ``clock != "real"``.
        serializable_invariant: Check serializability against sequential
            runs.  Cannot be combined with a virtual clock.
        error_on_any_race: Not supported here — requires the DPOR strategy.
        clock: ``"real"`` (default), ``"virtual"`` (autojump virtual clock:
            time reads are virtual, ``asyncio.sleep`` costs zero wall time),
            or ``"explored"`` (schedule entries landing on a sleeping task
            advance the clock, exploring early timer firings).  Tasks that
            block on primitives the scheduler cannot see (e.g. a raw
            ``asyncio.Lock``) are handled by a quiescence heuristic; prefer
            the DPOR strategy for lock-heavy async code.  ``asyncio.wait_for``,
            ``asyncio.timeout``, and ``asyncio.timeout_at`` inside explored
            tasks use virtual deadlines. See :doc:`/virtual_clock`.
        clock_diagnostics: Accepted for API consistency. Async random does not
            trace worker frames, so captured ``time.*`` references cannot be
            diagnosed and a ``RuntimeWarning`` is emitted.

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct schedule orderings were observed.
    """
    if error_on_any_race:
        raise ValueError("error_on_any_race requires DPOR (use frontrun.explore with strategy='dpor' instead)")
    clock_config = ClockConfig(mode=clock, diagnostics=clock_diagnostics).validate(
        patch_sleep=patch_sleep,
        serializable_invariant=serializable_invariant,
    )
    clock = clock_config.mode
    if clock_diagnostics:
        # The async random strategy operates at await-point granularity and does
        # not trace worker opcodes, so it cannot detect captured time references.
        warnings.warn(
            "clock_diagnostics is not supported for the async random strategy "
            "(it does not trace worker opcodes); use strategy='dpor' for diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )

    from frontrun._dpor_core import compute_serializable_baseline_async

    serial_valid_states, serial_hash_fn = await compute_serializable_baseline_async(
        setup, tasks, serializable_invariant
    )

    with _patch_async_runtime(
        detect_sql=detect_sql,
        patch_sleep=patch_sleep,
        virtual_time=clock != "real",
        pin_loop_time=asyncio.get_running_loop() if clock != "real" else None,
    ):
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

            # setup() runs OUTSIDE the crash handler below: a broken setup or
            # test harness is an error that must propagate, not a
            # "counterexample".  Only exceptions from the task bodies are turned
            # into findings.  (Mirrors _run_with_schedule_status construction.)
            attempt_clock = clock_config.new_clock()
            scheduler = AwaitScheduler(
                schedule,
                num_tasks,
                deadlock_timeout=deadlock_timeout,
                detect_sql=detect_sql,
                virtual_clock=attempt_clock,
                clock_mode=clock,
                max_ops=max_ops,
            )
            runner = AsyncShuffler(scheduler)
            # One clock_context owns the time.* patch for this attempt across
            # setup + tasks + invariant; tasks created by runner.run inherit the
            # contextvar, so they see the same virtual time as the driver.
            with clock_context(attempt_clock):
                state = setup()
                funcs: list[Callable[..., Coroutine[Any, Any, None]]] = [
                    lambda s=state, t=t: t(s)  # type: ignore[misc]
                    for t in tasks
                ]

                try:
                    await runner.run(funcs, timeout=timeout_per_run)
                except Exception as task_err:  # noqa: BLE001
                    if scheduler._max_ops_exhausted:
                        result.num_explored += 1
                        seen_schedule_hashes.add(hash(tuple(schedule)))
                        continue
                    # A task raised under this interleaving.  That is a legitimate
                    # counterexample (IndexError/KeyError/AssertionError in the task
                    # body is a common way a race manifests), not a fatal error that
                    # should abort exploration.  Surface it the same way the DPOR
                    # path does instead of letting it propagate.
                    result.num_explored += 1
                    seen_schedule_hashes.add(hash(tuple(schedule)))
                    result.property_holds = False
                    result.counterexample = schedule
                    result.unique_interleavings = len(seen_schedule_hashes)
                    result.explanation = (
                        f"Task crash in execution {result.num_explored}: {type(task_err).__name__}: {task_err}"
                    )
                    return result
                result.num_explored += 1
                seen_schedule_hashes.add(hash(tuple(schedule)))

                if scheduler._max_ops_exhausted:
                    continue

                # A timeout/deadlock means tasks were cancelled mid-flight: the
                # state is partial and does not describe any completed interleaving,
                # so the invariant must NOT be evaluated against it.
                #
                # Distinguish the two cases the way the sync bytecode explorer does
                # does: a genuinely-detected deadlock (the scheduler proved no
                # task can proceed — e.g. a lock-order inversion) is a
                # constructive counterexample and is surfaced; a plain wall-clock
                # timeout with no scheduler-detected deadlock only means the
                # worker was slow, which is inconclusive.
                if runner.scheduler.had_error:
                    result.property_holds = False
                    result.counterexample = schedule
                    result.unique_interleavings = len(seen_schedule_hashes)
                    sched_err = runner.scheduler._error
                    detail = str(sched_err) if sched_err is not None else "tasks did not complete within the timeout"
                    result.explanation = f"Deadlock detected after {result.num_explored} interleaving(s).\n\n{detail}"
                    return result
                if runner.timed_out:
                    continue

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
