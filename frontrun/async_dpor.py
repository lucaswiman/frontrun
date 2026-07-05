"""Async DPOR (Dynamic Partial Order Reduction) for frontrun.

Combines the DPOR engine's systematic interleaving exploration with the
async scheduler's await-point-level control.  Instead of the random
schedule sampling used by ``async_shuffler.py``, this module uses the
Rust DPOR engine to explore every meaningfully distinct interleaving
exactly once.

The approach:
1. A Rust DPOR engine (frontrun._dpor) manages the exploration tree,
   vector clocks, and wakeup tree exploration.
2. Python drives execution: each task's coroutine is wrapped with
   ``_AutoPauseCoroutine`` which intercepts every ``await`` yield and
   inserts a DPOR scheduling decision.  No user code changes needed.
3. At each yield, the scheduler reports the access to the DPOR engine
   and asks it which task to run next.
4. ``asyncio.Lock`` is monkey-patched to a cooperative version with
   deadlock detection (WaitForGraph) and explicit scheduling points.
5. SQL queries are intercepted via async cursor patching and reported
   as I/O resource accesses to the DPOR engine.

Usage::

    import asyncio
    import frontrun

    class Counter:
        def __init__(self):
            self.value = 0

        async def increment(self):
            temp = self.value
            await asyncio.sleep(0)  # any natural await works
            self.value = temp + 1

    result = await frontrun.explore(
        setup=lambda: Counter(),
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    assert result.property_holds, result.explanation  # fails — lost update!
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from frontrun._async_autopause import (
    _in_scheduler_pause,
    _scheduler_var,
    _task_id_var,
    await_point,  # noqa: F401
    wrap_auto_paused_tasks,
)
from frontrun._deadlock import DeadlockError, WaitForGraph, format_cycle
from frontrun._dpor_core import (
    NoOpLock,
    ReplayEngine,
    ReplayExecution,
    RowLockRegistry,
    advance_replay_index,
    apply_lock_blocked_override,
    compute_serializable_baseline_async,
    dpor_exploration_iter,
    event_wake_sync_id,
    extend_replay_schedule,
    format_race_failure_explanation,
    group_schedule_runs,
    is_reproduction_run,
    make_deadline,
    make_dpor_engine,
    record_dpor_failure,
    wake_sync_id,
)
from frontrun._opcode_observer import (
    OpcodeTraceHandle,
    ShadowStack,
    StableObjectIds,
    _make_object_key,
    _process_opcode,
    install_thread_opcode_trace,
    start_opcode_trace,
    stop_opcode_trace,
    uninstall_thread_opcode_trace,
)
from frontrun._sql_cursor import clear_sql_metadata, get_lock_timeout, set_lock_timeout
from frontrun._sql_insert_tracker import check_uncaptured_inserts
from frontrun._threaded_runner import PatchScope
from frontrun._tracing import TraceFilter as _TraceFilter
from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun._virtual_clock import (
    ClockMode,
    VirtualClock,
    clock_context,
    patch_time,
    real_monotonic,
    unpatch_time,
    validate_clock_options,
)
from frontrun.async_scheduler import InterleavedLoop, _in_frontrun_timer, frontrun_wait_for
from frontrun.common import (
    InterleavingResult,
    check_invariant,
    check_serializability_violation,
)

try:
    from frontrun._dpor import PyDporEngine, PyExecution  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "frontrun.explore with async workers requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14   (or build-dpor-3.10 / build-dpor-3.14t)\n"
        "Or install from source:  pip install -e ."
    ) from _err

# Lazy import for async SQL patching (avoid hard dependency)
_sql_async_available = False
try:
    from frontrun._sql_cursor_async import patch_sql_async, unpatch_sql_async

    _sql_async_available = True
except ImportError:

    def patch_sql_async() -> None:  # type: ignore[misc]
        pass

    def unpatch_sql_async() -> None:  # type: ignore[misc]
        pass


# Lazy import for async Redis patching (avoid hard dependency)
_redis_async_available = False
try:
    from frontrun._redis_client_async import patch_redis_async, unpatch_redis_async

    _redis_async_available = True
except ImportError:

    def patch_redis_async() -> None:  # type: ignore[misc]
        pass

    def unpatch_redis_async() -> None:  # type: ignore[misc]
        pass


T = TypeVar("T")


# Guards against re-entering async opcode tracing while _process_opcode()
# is already running for the current task.
_in_trace_processing: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_trace_processing", default=False)


# ---------------------------------------------------------------------------
# Object key helper (shared with sync dpor.py)
# ---------------------------------------------------------------------------


# Synchronization acquisition points must still be explored even when the
# individual lock resources differ, because future blocking can make those
# orderings observably distinct (deadlock, starvation, etc.).
_SHARED_SYNC_ACQUIRE_KEY = _make_object_key(0, "__async_dpor_sync_acquire__")


# ---------------------------------------------------------------------------
# Cooperative asyncio.Lock with deadlock detection
# ---------------------------------------------------------------------------

_real_asyncio_lock = asyncio.Lock
_async_lock_patched = False

# Per-lock wait-for graph for async DPOR deadlock detection.
_async_wait_graph: WaitForGraph | None = None
# Map from lock id(obj) → owning task_id
_async_lock_owners: dict[int, int] = {}
# Reverse map: task_id → set of lock objects held by that task.
# Used to force-release locks when a task finishes without calling release().
_async_task_held_locks: dict[int, set[Any]] = {}


def _reset_async_lock_state() -> None:
    """Clear all cooperative-lock global state (graph edges, owners, held sets).

    Must run before each exploration execution AND each replay attempt: stale
    wait-for edges / lock owners / held-locks from a prior run (compounded by
    id() reuse) leak into the next one and cause spurious DeadlockError or
    phantom ownership (finding F5).
    """
    if _async_wait_graph is not None:
        _async_wait_graph.clear()
    _async_lock_owners.clear()
    _async_task_held_locks.clear()
    _async_parked_events.clear()


class _CooperativeAsyncLock:
    """Drop-in asyncio.Lock replacement with wait-for graph deadlock detection.

    When a task tries to acquire a held lock, registers a waiting edge
    in the global WaitForGraph.  If adding the edge creates a cycle,
    raises DeadlockError immediately instead of blocking forever.

    Every acquire() is also a DPOR scheduling point (via await_point()).
    This is necessary because asyncio.Lock.acquire() on a free lock
    completes synchronously without yielding, which would prevent DPOR
    from interleaving lock acquisitions across tasks.
    """

    def __init__(self) -> None:
        self._lock = _real_asyncio_lock()
        self._owner: int | None = None

    def locked(self) -> bool:
        return self._lock.locked()

    async def acquire(self) -> bool:
        task_id = _task_id_var.get()
        graph = _async_wait_graph

        # Make lock acquisition a DPOR scheduling point so the engine
        # can interleave different tasks' lock acquisitions.  Without
        # this, asyncio.Lock.acquire() on a free lock completes
        # synchronously and DPOR never sees the interleaving where
        # two tasks hold conflicting locks simultaneously.
        #
        # The scheduler.pause() call sets _in_scheduler_pause so the
        # coroutine wrapper won't insert a redundant scheduling point
        # for the pause's own yields.
        scheduler = _scheduler_var.get()
        if scheduler is not None and task_id is not None and _in_scheduler_pause.get() == 0:
            await scheduler.pause(task_id, ("lock_acquire", id(self)))

        if task_id is not None and graph is not None and self._lock.locked():
            lock_id = id(self)
            # Register: this task is waiting for this lock
            cycle = graph.add_waiting(task_id, lock_id)
            if cycle is not None:
                graph.remove_waiting(task_id, lock_id)
                desc = format_cycle(cycle)
                raise DeadlockError(f"Async lock deadlock detected: {desc}", desc)
            # Mark this task as blocked in the DPOR execution so the engine
            # won't schedule it while it's waiting for the lock.  Also track
            # the lock holder so _schedule_next can redirect to the holder
            # if needed.
            if scheduler is not None:
                scheduler.execution.block_thread(task_id)
                if self._owner is not None:
                    scheduler._lock_blocked[task_id] = self._owner
            # Set _in_scheduler_pause so the AutoPauseCoroutine passes
            # the lock's internal yields through to the event loop without
            # inserting scheduling points.  Without this, the DPOR scheduler
            # would try to schedule at every yield of the lock's acquire,
            # creating a deadlock (the blocked task can't proceed but the
            # scheduler keeps picking it).
            depth = _in_scheduler_pause.get()
            _in_scheduler_pause.set(depth + 1)
            try:
                # If blocking ourselves left nothing engine-runnable (the
                # holder may be parked in sleep_until with no scheduling
                # points until the clock advances), hand the turn onward now
                # — otherwise no one ever calls _schedule_next and the run
                # dies by deadlock timeout: a false deadlock counterexample.
                kick = getattr(scheduler, "kick_stalled_schedule", None)
                if kick is not None:
                    await kick(task_id)
                result = await self._lock.acquire()
            finally:
                _in_scheduler_pause.set(depth)
                graph.remove_waiting(task_id, lock_id)
                if scheduler is not None:
                    scheduler.execution.unblock_thread(task_id)
                    scheduler._lock_blocked.pop(task_id, None)
        else:
            result = await self._lock.acquire()

        # Record ownership
        if task_id is not None and graph is not None:
            lock_id = id(self)
            self._owner = task_id
            _async_lock_owners[lock_id] = task_id
            _async_task_held_locks.setdefault(task_id, set()).add(self)
            graph.add_holding(task_id, lock_id)
            scheduler = _scheduler_var.get()
            if scheduler is not None:
                scheduler.engine.report_sync(scheduler.execution, task_id, "lock_acquire", lock_id)

        return result

    def release(self) -> None:
        graph = _async_wait_graph
        if self._owner is not None:
            task_id = self._owner
            lock_id = id(self)
            if graph is not None:
                graph.remove_holding(task_id, lock_id)
            _async_lock_owners.pop(lock_id, None)
            held = _async_task_held_locks.get(task_id)
            if held is not None:
                held.discard(self)
            scheduler = _scheduler_var.get()
            if scheduler is not None:
                scheduler.engine.report_sync(scheduler.execution, task_id, "lock_release", lock_id)
            self._owner = None
        self._lock.release()

    async def __aenter__(self) -> bool:
        return await self.acquire()

    async def __aexit__(self, *args: Any) -> None:
        self.release()


def _release_task_async_locks(task_id: int) -> None:
    """Force-release all asyncio.Lock objects held by *task_id*.

    Called when a task finishes (normally or via exception) without
    explicitly releasing its locks.  Cleans up both the WaitForGraph
    holding edges and the underlying real asyncio.Lock objects.
    """
    held = _async_task_held_locks.pop(task_id, None)
    if not held:
        return
    graph = _async_wait_graph
    for lock_obj in list(held):
        lock_id = id(lock_obj)
        if graph is not None:
            graph.remove_holding(task_id, lock_id)
        _async_lock_owners.pop(lock_id, None)
        lock_obj._owner = None
        if lock_obj._lock.locked():
            lock_obj._lock.release()


def _patch_asyncio_lock() -> None:
    """Replace asyncio.Lock with cooperative deadlock-detecting version."""
    global _async_lock_patched, _async_wait_graph  # noqa: PLW0603
    if _async_lock_patched:
        return
    _async_wait_graph = WaitForGraph()
    _async_lock_owners.clear()
    asyncio.Lock = _CooperativeAsyncLock  # type: ignore[assignment,misc]
    _async_lock_patched = True


def _unpatch_asyncio_lock() -> None:
    """Restore original asyncio.Lock."""
    global _async_lock_patched, _async_wait_graph  # noqa: PLW0603
    if not _async_lock_patched:
        return
    asyncio.Lock = _real_asyncio_lock  # type: ignore[assignment,misc]
    _reset_async_lock_state()
    _async_wait_graph = None
    _async_lock_patched = False


# ---------------------------------------------------------------------------
# Cooperative asyncio.Event
# ---------------------------------------------------------------------------

_real_asyncio_event = asyncio.Event

# Cooperative events that currently have at least one task parked in wait().
# When exact deadlock detection aborts a run, these waiters must be woken
# (by setting the real event) so the tasks free-run to completion and
# run_all can surface the DeadlockError instead of hanging to the timeout.
_async_parked_events: set[_CooperativeAsyncEvent] = set()


def _wake_parked_async_event_waiters() -> None:
    for event in list(_async_parked_events):
        event._event.set()


class _CooperativeAsyncEvent:
    """Drop-in asyncio.Event replacement that engine-blocks its waiters.

    A task parked on a stock ``asyncio.Event`` is invisible to the DPOR
    engine: it still looks runnable, so the scheduler keeps picking it, no
    other task ever gets the turn, and the run dies by wall timeout — scored
    as a false deadlock counterexample (the event analogue of a task parked
    on an ``asyncio.Lock``).  Blocking the waiter in the engine hands the
    turn onward, and ``set()`` / wake report the ``set() → wait()-return``
    happens-before edge (mirroring the sync ``CooperativeEvent``).
    """

    def __init__(self) -> None:
        self._event = _real_asyncio_event()
        # task_ids currently engine-blocked in wait(), in park order.
        self._waiters: list[int] = []

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        self._event.clear()

    async def wait(self) -> bool:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()

        # Same scheduling point as _CooperativeAsyncLock.acquire: without it
        # the set()/wait() ordering on an already-set event is never a DPOR
        # choice (Event.wait on a set event completes synchronously).
        if scheduler is not None and task_id is not None and _in_scheduler_pause.get() == 0:
            await scheduler.pause(task_id, ("event_wait", id(self)))

        if self._event.is_set():
            return True
        # A scheduler-detected deadlock/timeout aborts the run: tasks free-run
        # to completion (see InterleavedLoop.run_all), so waiting here would
        # hang until the outer wall timeout.
        if scheduler is not None and scheduler._error is not None:
            return True

        engine = getattr(scheduler, "engine", None)
        if scheduler is None or task_id is None or engine is None:
            return await self._event.wait()

        # Between the is_set() probe above and block_thread below there are
        # no awaits, and async DPOR is single-threaded — no set() can slip
        # in unobserved (the sync CooperativeEvent needs an engine-lock
        # dance for the same guarantee).
        self._waiters.append(task_id)
        _async_parked_events.add(self)
        scheduler.execution.block_thread(task_id)
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            # Blocking ourselves may have left nothing engine-runnable
            # (see _CooperativeAsyncLock.acquire): hand the turn onward.
            kick = getattr(scheduler, "kick_stalled_schedule", None)
            if kick is not None:
                await kick(task_id)
            result = await self._event.wait()
        finally:
            _in_scheduler_pause.set(depth)
            if task_id in self._waiters:
                self._waiters.remove(task_id)
            if not self._waiters:
                _async_parked_events.discard(self)
            scheduler.execution.unblock_thread(task_id)
        # Close the set() → wake happens-before edge (skip if the run was
        # aborted — the free-run's events are not part of the exploration).
        if scheduler._error is None:
            engine.report_sync(scheduler.execution, task_id, "lock_acquire", event_wake_sync_id(id(self), task_id))
        return result

    def set(self) -> None:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        engine = getattr(scheduler, "engine", None)
        if scheduler is not None and engine is not None and task_id is not None and scheduler._error is None:
            for waiter in list(self._waiters):
                engine.report_sync(scheduler.execution, task_id, "lock_release", event_wake_sync_id(id(self), waiter))
                scheduler.execution.unblock_thread(waiter)
        self._event.set()

    def __repr__(self) -> str:
        return f"<_CooperativeAsyncEvent set={self.is_set()}>"


def _patch_asyncio_event() -> None:
    asyncio.Event = _CooperativeAsyncEvent  # type: ignore[assignment,misc]


def _unpatch_asyncio_event() -> None:
    asyncio.Event = _real_asyncio_event  # type: ignore[assignment,misc]
    _async_parked_events.clear()


# ---------------------------------------------------------------------------
# Frontrun-internal loop timer tagging (exact deadlock detection)
# ---------------------------------------------------------------------------


def _install_frontrun_timer_tagging(loop: Any) -> tuple[Callable[[], bool], Callable[[], None]]:
    """Wrap ``loop.call_at`` so frontrun's own watchdog timers are tagged.

    Exact deadlock detection may only fire when every pending loop timer is
    one of frontrun's own (a pending *user* timer — e.g. a wall-clock
    ``asyncio.wait_for`` — may still wake a parked task, so the state is not
    a proven deadlock).  Timers created while ``_in_frontrun_timer`` is set
    (see ``frontrun_wait_for``) are collected in a WeakSet; everything else
    counts as a user timer.

    Returns ``(user_timers_pending, uninstall)``.  ``user_timers_pending``
    is conservative: if the loop's timer heap cannot be inspected (a
    non-standard loop without ``_scheduled``), it reports True so exact
    detection stays off and the wall-clock fallback applies.
    """
    tagged: weakref.WeakSet[Any] = weakref.WeakSet()
    orig_call_at = loop.call_at

    def _tagging_call_at(when: float, callback: Any, *args: Any, context: Any = None) -> Any:
        handle = orig_call_at(when, callback, *args, context=context)
        if _in_frontrun_timer.get():
            tagged.add(handle)
        return handle

    loop.call_at = _tagging_call_at

    def _user_timers_pending() -> bool:
        scheduled = getattr(loop, "_scheduled", None)
        if scheduled is None:
            return True
        return any(not handle.cancelled() and handle not in tagged for handle in scheduled)

    def _uninstall() -> None:
        del loop.call_at

    return _user_timers_pending, _uninstall


# ---------------------------------------------------------------------------
# asyncio.sleep patching
# ---------------------------------------------------------------------------

_real_asyncio_sleep = asyncio.sleep
_async_sleep_patched = False


async def _cooperative_async_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401
    """No-delay replacement for ``asyncio.sleep`` during exploration.

    Yields to the event loop (``await asyncio.sleep(0)``) so it remains
    a scheduling point, but skips the actual delay.

    When the active async scheduler owns a virtual clock, a positive sleep
    becomes a *timed block*: the task registers a virtual deadline and
    suspends until the scheduler advances the clock to it.
    ``asyncio.sleep(0)`` stays a pure yield, matching stock semantics.
    """
    scheduler = _scheduler_var.get()
    task_id = _task_id_var.get()
    if scheduler is not None and task_id is not None and delay and delay > 0:
        clock = getattr(scheduler, "virtual_clock", None)
        sleep_until = getattr(scheduler, "sleep_until", None)
        if clock is not None and sleep_until is not None:
            await sleep_until(task_id, clock.now() + delay)
            return result
    await _real_asyncio_sleep(0)
    return result


def _patch_asyncio_sleep() -> None:
    """Replace ``asyncio.sleep`` with a zero-delay version."""
    global _async_sleep_patched  # noqa: PLW0603
    if _async_sleep_patched:
        return
    asyncio.sleep = _cooperative_async_sleep  # type: ignore[assignment]
    _async_sleep_patched = True


def _unpatch_asyncio_sleep() -> None:
    """Restore original ``asyncio.sleep``."""
    global _async_sleep_patched  # noqa: PLW0603
    if not _async_sleep_patched:
        return
    asyncio.sleep = _real_asyncio_sleep  # type: ignore[assignment]
    _async_sleep_patched = False


# ---------------------------------------------------------------------------
# Async DPOR Scheduler
# ---------------------------------------------------------------------------


class AsyncDporScheduler(InterleavedLoop):
    """Controls async task execution at await-point granularity using DPOR.

    Instead of following a fixed schedule, uses the Rust DPOR engine to
    decide which task runs next.  Shared-memory accesses inside each
    await-delimited block are traced at opcode/instruction granularity
    and reported to the engine before the next scheduling decision.
    """

    def __init__(
        self,
        engine: PyDporEngine,
        execution: PyExecution,
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        detect_sql: bool = False,
        detect_redis: bool = False,
        stable_ids: StableObjectIds | None = None,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_actor_id: int | None = None,
        user_timers_pending: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self.engine = engine
        self.execution = execution
        self._num_engine_tasks = num_tasks
        # Exact deadlock detection (virtual clock only): a pending confirm
        # task re-checks a "nobody engine-runnable, no deadline pending"
        # observation after the loop's in-flight wake callbacks have drained.
        self._user_timers_pending = user_timers_pending
        self._deadlock_confirm_pending = False
        self._deadlock_confirm_task: asyncio.Task[None] | None = None
        # Virtual clock (ideas/virtual_clock.md), mirroring the sync
        # DporScheduler: an extra engine thread (the clock actor) whose steps
        # advance the clock to the earliest pending deadline.  "virtual" =
        # autojump (actor enabled only when nothing is runnable); "explored" =
        # the advance is a schedulable choice whenever a deadline is pending.
        self.virtual_clock = virtual_clock
        self._clock_mode = clock_mode
        self._clock_actor_id = clock_actor_id
        # task_id → virtual deadline for tasks parked in sleep_until().  No
        # async analogue of the sync _timed_waits exists: asyncio.Lock.acquire
        # has no timeout parameter (asyncio timeouts stay on the wall clock;
        # see docs/virtual_clock.rst).
        self._sleepers: dict[int, float] = {}
        self._current_task: int | None = None
        self._detect_sql = detect_sql
        self._detect_redis = detect_redis
        self._engine_lock = NoOpLock()
        self.trace_recorder = None
        self._iter_to_container: dict[int, Any] = {}
        self._shadow_stacks: dict[int, ShadowStack] = {}
        self._opcode_handle: OpcodeTraceHandle | None = None
        self._stable_ids = stable_ids if stable_ids is not None else StableObjectIds()
        # Pending I/O accesses per task (from SQL interception)
        self._pending_io: dict[int, list[tuple[int, str, bool]]] = {i: [] for i in range(num_tasks)}

        # Track tasks blocked on asyncio.Lock: task_id → lock-holder task_id.
        # When DPOR schedules a blocked task, override to run the holder.
        self._lock_blocked: dict[int, int] = {}

        # Row lock tracking: state and _row_lock_int_id() live in RowLockRegistry;
        # alias dicts into this namespace so the rest of the class is unchanged.
        self._row_lock_registry = RowLockRegistry()
        self._active_row_locks: dict[str, int] = self._row_lock_registry._active_row_locks
        self._task_row_locks: dict[int, set[str]] = self._row_lock_registry._task_row_locks
        self._row_lock_ids: dict[str, int] = self._row_lock_registry._row_lock_ids

        # The clock actor starts blocked; it becomes runnable only when a
        # deadline is pending (see _sync_clock_actor / _schedule_next).
        if self._clock_actor_id is not None:
            self.execution.block_thread(self._clock_actor_id)

        # Request the first scheduling decision
        self._current_task = self._schedule_next()

    # -- Virtual clock ---------------------------------------------------

    def _has_pending_deadlines(self) -> bool:
        return bool(self._sleepers)

    def _sync_clock_actor(self) -> None:
        """Keep the clock actor's enabledness in step with pending deadlines."""
        if self._clock_actor_id is None:
            return
        if self._clock_mode == "explored" and self._has_pending_deadlines():
            self.execution.unblock_thread(self._clock_actor_id)
        else:
            self.execution.block_thread(self._clock_actor_id)

    def _advance_virtual_clock(self) -> None:
        """One clock-actor step: jump to the earliest deadline, wake sleepers."""
        clock = self.virtual_clock
        if clock is None:
            return
        pending = list(self._sleepers.values())
        if not pending:
            self._sync_clock_actor()
            return
        clock.advance_to(min(pending))
        now = clock.now()
        for tid, dl in sorted(self._sleepers.items(), key=lambda kv: (kv[1], kv[0])):
            if dl <= now:
                del self._sleepers[tid]
                self.execution.unblock_thread(tid)
                self.engine.report_sync(self.execution, self._clock_actor_id, "lock_release", wake_sync_id(tid))
        self._sync_clock_actor()

    async def sleep_until(self, task_id: int, deadline: float) -> None:
        """Block *task_id* until the virtual clock reaches *deadline*.

        Mirrors the sync ``DporScheduler.sleep_until``: register the
        deadline, block in the engine, release the turn, then wait for the
        clock-actor advance and for the engine to schedule us again.
        """
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            # Let any previously-notified tasks process their wakeups first
            # (same fairness yield as pause()).
            await _real_asyncio_sleep(0)
            self._progress += 1
            async with self._condition:
                if self._finished or self._error:
                    return
                self._sleepers[task_id] = deadline
                self.execution.block_thread(task_id)
                self._sync_clock_actor()
                # Reschedule if we held the turn — or if nothing is engine-
                # runnable anymore (e.g. every other task is parked on an
                # asyncio.Lock we hold: they have no scheduling points, so
                # no one else will ever call _schedule_next and the run
                # would die by deadlock timeout instead of autojumping).
                cur = self._current_task
                if cur == task_id or cur is None or not self.execution.runnable_threads():
                    next_task = self._schedule_next()
                    if next_task is not None:
                        self._current_task = next_task
                self._condition.notify_all()
                # Phase 1: wait for the clock advance that removes us.
                while task_id in self._sleepers:
                    if self._finished or self._error:
                        self._sleepers.pop(task_id, None)
                        self.execution.unblock_thread(task_id)
                        self._sync_clock_actor()
                        return
                    try:
                        await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                    except asyncio.TimeoutError:
                        self._error = TimeoutError(
                            f"Deadlock: task {task_id} sleeping until t={deadline} was never woken"
                        )
                        self._condition.notify_all()
                        self._sleepers.pop(task_id, None)
                        return
                # Phase 2: woken — wait until the engine schedules us again.
                while not (self._finished or self._error) and self._current_task != task_id:
                    try:
                        await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                    except asyncio.TimeoutError:
                        self._error = TimeoutError(
                            f"Deadlock: task {task_id} woke at t={deadline} but was never rescheduled"
                        )
                        self._condition.notify_all()
                        return
                if self._finished or self._error:
                    return
                # Close the wake happens-before edge (clock advance → resume).
                self.engine.report_sync(self.execution, task_id, "lock_acquire", wake_sync_id(task_id))
        finally:
            _in_scheduler_pause.set(depth)

    async def kick_stalled_schedule(self, task_id: int) -> None:
        """Hand the turn onward after *task_id* engine-blocked itself.

        Called by ``_CooperativeAsyncLock.acquire`` right after it blocks the
        task, because the blocked task parks in the real lock acquire with no
        further scheduling points: if it held the turn — or if nothing is
        engine-runnable anymore — no other code path would ever call
        ``_schedule_next`` (whose runnable-empty branch performs the
        virtual-clock autojump).
        """
        async with self._condition:
            if self._finished or self._error:
                return
            cur = self._current_task
            if cur != task_id and cur is not None and self.execution.runnable_threads():
                return
            next_task = self._schedule_next()
            if next_task is not None:
                self._current_task = next_task
            self._condition.notify_all()

    def _maybe_confirm_exact_deadlock(self) -> None:
        """Arm a deferred exact-deadlock check (virtual clock only).

        Called from ``_schedule_next`` when nothing is engine-runnable and no
        deadline is pending.  That state is a *candidate* deadlock, not a
        proven one: engine-unblock happens lazily when a woken task resumes
        (in the lock/event wrappers' ``finally``), so a wake callback may
        still be sitting in the loop's ready queue.  The confirm task yields
        twice to drain in-flight wakes, then re-checks under the condition.
        """
        if self.virtual_clock is None or self._deadlock_confirm_pending:
            return
        if self._error is not None or self._finished:
            return
        if len(self._tasks_done) >= self._num_engine_tasks:
            return
        self._deadlock_confirm_pending = True
        self._deadlock_confirm_task = asyncio.get_running_loop().create_task(self._confirm_exact_deadlock())

    async def _confirm_exact_deadlock(self) -> None:
        try:
            for _ in range(2):
                await _real_asyncio_sleep(0)
            async with self._condition:
                if self._error is not None or self._finished:
                    return
                if len(self._tasks_done) >= self._num_engine_tasks:
                    return
                if self.execution.runnable_threads():
                    return
                if self._has_pending_deadlines():
                    return
                checker = self._user_timers_pending
                if checker is None or checker():
                    # A pending user timer (e.g. a wall-clock asyncio.wait_for
                    # around a blocked await) may still wake a parked task —
                    # not a proven deadlock; leave it to the wall fallback.
                    return
                desc = (
                    "all live tasks are blocked and no virtual-clock deadline is pending "
                    f"(sleepers={sorted(self._sleepers)}, done={sorted(self._tasks_done)})"
                )
                self._error = DeadlockError(f"Deadlock detected by virtual clock: {desc}", desc)
                # Wake tasks parked on cooperative events so they free-run to
                # completion and run_all can raise the DeadlockError (tasks
                # parked on cooperative locks unstick when their holders
                # finish and force-release).
                _wake_parked_async_event_waiters()
                self._condition.notify_all()
        finally:
            self._deadlock_confirm_pending = False

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which task to run next.

        If the engine picks a task blocked on an asyncio.Lock, override
        the decision to schedule the lock holder instead.  This prevents
        the scheduler from cycling between a blocked task and the event
        loop, causing false deadlock timeouts.

        Clock-actor steps are handled inline: when the engine schedules the
        actor, the clock advances to the earliest deadline and the loop asks
        the engine again (see the sync DporScheduler._schedule_next).
        """
        while True:
            runnable = self.execution.runnable_threads()
            if not runnable:
                if (
                    self.virtual_clock is not None
                    and self._clock_actor_id is not None
                    and self._has_pending_deadlines()
                ):
                    # Autojump: everything is blocked and timers are pending —
                    # the clock advance is the only possible transition.
                    self.execution.unblock_thread(self._clock_actor_id)
                    continue
                self._maybe_confirm_exact_deadlock()
                return None
            scheduled = self.engine.schedule(self.execution)
            if scheduled is not None and scheduled == self._clock_actor_id:
                self._advance_virtual_clock()
                continue
            # Shared with the sync scheduler: redirect to the lock holder when the
            # engine picks a lock-blocked task, or drop a stale entry whose holder
            # has finished.
            return apply_lock_blocked_override(scheduled, self._lock_blocked, self._tasks_done)

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        if self._current_task is None:
            self._finished = True
            return True
        if self._current_task == task_id:
            return True
        # If the currently-scheduled task is blocked on a lock held by
        # task_id, let task_id proceed so it can release the lock.
        if self._current_task in self._lock_blocked:
            holder = self._lock_blocked[self._current_task]
            if holder == task_id:
                return True
        return False

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        # Flush any pending I/O accesses before advancing
        self._flush_pending_io(task_id)
        if isinstance(marker, tuple) and marker and marker[0] == "lock_acquire":
            # When SQL/Redis detection is active, opcode-level tracing is
            # disabled, so the only way to create cross-resource conflicts
            # (asyncio.Lock vs row lock) is via _SHARED_SYNC_ACQUIRE_KEY.
            # Without it, DPOR won't explore interleavings needed to find
            # cross-resource deadlocks.
            #
            # When SQL/Redis detection is NOT active, skip this write.
            # The Rust engine's process_sync("lock_acquire") already creates
            # a virtual lock-object Write access WITH the happens-before
            # join from the prior release applied first.  Reporting
            # _SHARED_SYNC_ACQUIRE_KEY here (before report_sync runs)
            # creates a write-write conflict that isn't ordered by the
            # lock's HB, causing spurious trace exploration (e.g. 4 traces
            # instead of 2 for two tasks with a single lock).
            if self._detect_sql or self._detect_redis:
                self.engine.report_access(
                    self.execution,
                    task_id,
                    _SHARED_SYNC_ACQUIRE_KEY,
                    "write",
                )

        # Schedule next task.  If the engine returns None (no runnable
        # threads), keep _current_task as the current task_id so the
        # current task can continue.  Only set _finished when ALL tasks
        # are done.
        next_task = self._schedule_next()
        if next_task is not None:
            self._current_task = next_task
        # If next_task is None but there are alive tasks, keep current
        # task so it can continue running to completion.

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        self._error = TimeoutError(
            f"Deadlock: DPOR async scheduler wants task {self._current_task} "
            f"but task {task_id} is waiting at marker {marker!r}"
        )
        self._condition.notify_all()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)
        # Set up IO reporter context so SQL interception can report to us.
        # Async DPOR runs all tasks on one event-loop thread, so the DPOR
        # scheduler/thread-id and transaction state must be task-aware
        # (contextvar-backed) rather than per-thread (findings F2, F4).
        from frontrun._io_detection import (
            set_dpor_scheduler_task,
            set_dpor_thread_id_task,
            set_io_reporter,
            set_tx_store_task,
        )

        set_dpor_scheduler_task(self)
        set_dpor_thread_id_task(task_id)

        if self._detect_sql or self._detect_redis:

            def _io_reporter(resource_id: str, kind: str) -> None:
                # Dynamically read the current task ID so that when multiple
                # async tasks share the same thread-local reporter, each
                # I/O event is attributed to the task that actually runs the
                # Redis/SQL command, not whichever task was set up last.
                current_task = _task_id_var.get()
                if current_task is None:
                    current_task = task_id
                object_key = _make_object_key(hash(resource_id), resource_id)
                self._pending_io.setdefault(current_task, []).append((object_key, kind, True))

            set_io_reporter(_io_reporter)

        # Reset transaction state for this task in a fresh per-task store.
        store = set_tx_store_task()
        store._in_transaction = False
        store._is_autobegin = False
        store._tx_buffer = []
        store._tx_savepoints = {}

    async def run_all(
        self,
        task_funcs: dict[Any, Callable[..., Awaitable[None]]] | list[Callable[..., Awaitable[None]]],
        timeout: float = 10.0,
        *,
        detect_external_deadlock: bool = False,
    ) -> None:
        """Run tasks with DPOR-controlled interleaving.

        Each task's coroutine is wrapped with ``_AutoPauseCoroutine``,
        which automatically inserts a DPOR scheduling point before every
        step of the inner coroutine (i.e. at every natural ``await``).
        No explicit ``await await_point()`` calls are needed.
        """
        if isinstance(task_funcs, list):
            task_funcs = dict(enumerate(task_funcs))

        wrapped = wrap_auto_paused_tasks(task_funcs, self)
        self._start_opcode_trace()
        try:
            await super().run_all(wrapped, timeout=timeout, detect_external_deadlock=detect_external_deadlock)
        except DeadlockError:
            raise
        except Exception:
            # An exact-deadlock abort makes the remaining tasks free-run to
            # completion under false pretenses (parked event waits return on
            # an unset event); anything they raise while free-running is an
            # artifact, and the base run_all surfaces task errors before
            # self._error — reassert the DeadlockError's priority.
            if isinstance(self._error, DeadlockError):
                raise self._error from None
            raise
        finally:
            confirm = self._deadlock_confirm_task
            if confirm is not None and not confirm.done():
                confirm.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await confirm
            self._stop_opcode_trace()
            self._shadow_stacks.clear()

    async def pause(self, task_id: Any, marker: Any = None) -> None:
        """DPOR-aware pause that ensures fair task wakeup.

        After proceeding from a pause, yields to the event loop so other
        tasks that were notified can process their condition waits.
        Without this, a single task can reacquire the condition lock
        before other notified tasks, causing false deadlock detection.

        Sets ``_in_scheduler_pause`` so the coroutine wrapper knows not
        to insert a redundant scheduling point for this pause's yields.
        """
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            # Yield to let any previously-notified tasks process their wakeups
            await asyncio.sleep(0)
            await super().pause(task_id, marker)
        finally:
            _in_scheduler_pause.set(depth)

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished in both InterleavedLoop and the DPOR engine."""
        self.execution.finish_thread(task_id)
        # If this was the current task, schedule the next one
        async with self._condition:
            self._tasks_done.add(task_id)
            # Drop stale virtual-clock deadlines (safety net) and retire the
            # clock actor once every real task finished.
            if self.virtual_clock is not None:
                self._sleepers.pop(task_id, None)
                self._sync_clock_actor()
            if self._clock_actor_id is not None and len(self._tasks_done) >= self._num_engine_tasks:
                self.execution.unblock_thread(self._clock_actor_id)
                self.execution.finish_thread(self._clock_actor_id)
            if self._current_task == task_id:
                next_task = self._schedule_next()
                self._current_task = next_task
                if next_task is None and len(self._tasks_done) >= self._num_engine_tasks:
                    self._finished = True
            self._condition.notify_all()

    def _cleanup_task_context(self, task_id: Any) -> None:
        # Flush any remaining pending I/O before cleanup
        self._flush_pending_io(task_id)

        # Release any row locks still held (task finished without COMMIT)
        self.release_row_locks(task_id)

        # Release any asyncio.Lock objects still held (task crashed without release())
        _release_task_async_locks(task_id)

        _scheduler_var.set(None)
        _task_id_var.set(None)
        from frontrun._io_detection import set_dpor_scheduler_task, set_dpor_thread_id_task, set_io_reporter

        # Clear this task's task-aware DPOR context.  These contextvars are
        # per-task (they die with the task), but clearing keeps any further
        # interception in this context from resolving a stale scheduler.
        set_dpor_scheduler_task(None)
        set_dpor_thread_id_task(None)

        # The IO reporter is per-OS-thread (shared by all tasks on the event
        # loop), so only clear it when ALL tasks are done — clearing it when
        # one task finishes would break I/O detection for the rest.
        # Note: _cleanup_task_context runs BEFORE _mark_done, so the current
        # task_id is not yet in _tasks_done; +1 accounts for it.
        if len(self._tasks_done) + 1 >= self._num_engine_tasks and (self._detect_sql or self._detect_redis):
            set_io_reporter(None)

    def get_shadow_stack(self, frame_id: int) -> ShadowStack:
        stack = self._shadow_stacks.get(frame_id)
        if stack is None:
            stack = ShadowStack()
            self._shadow_stacks[frame_id] = stack
        return stack

    def remove_shadow_stack(self, frame_id: int) -> None:
        self._shadow_stacks.pop(frame_id, None)

    def _get_task_id(self) -> int | None:
        if _task_id_var.get() is not None and _scheduler_var.get() is self:
            return _task_id_var.get()
        return None

    def _on_opcode(self, code: Any, offset: int, frame: Any, task_id: int) -> bool:
        if _in_trace_processing.get():
            return False
        token = _in_trace_processing.set(True)
        try:
            with self._engine_lock:
                _process_opcode(frame, self, task_id)  # type: ignore[arg-type]
        finally:
            _in_trace_processing.reset(token)
        return False  # async never yields at opcode level

    def _start_opcode_trace(self) -> None:
        """Install the opcode tracer and activate it for the event-loop thread.

        Async DPOR runs all tasks on a single event-loop thread, so we install
        the per-thread trace exactly once (on sys.settrace; no-op on
        sys.monitoring, which is global).
        """
        self._opcode_handle = start_opcode_trace(
            get_thread_id=self._get_task_id,
            on_opcode=self._on_opcode,
            remove_shadow_stack=self.remove_shadow_stack,
            tool_name="frontrun.async_dpor",
        )
        install_thread_opcode_trace(self._opcode_handle)

    def _stop_opcode_trace(self) -> None:
        """Uninstall the per-thread tracer and tear down backend resources."""
        handle = getattr(self, "_opcode_handle", None)
        if handle is None:
            return
        uninstall_thread_opcode_trace(handle)
        stop_opcode_trace(handle)
        self._opcode_handle = None

    def _row_lock_int_id(self, res_id: str) -> int:
        """Return a stable integer ID for *res_id* (allocated on first call)."""
        return self._row_lock_registry._row_lock_int_id(res_id)

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Track SQL row locks in the async WaitForGraph for cross-resource deadlock detection.

        In the async single-threaded context we cannot block waiting for a
        holder to release.  Instead we:
        - Record the holding edge when the lock is free (or already ours).
        - Detect cycles instantly via WaitForGraph when another task holds
          the lock, raising DeadlockError so frontrun.explore reports it.
        """
        graph = _async_wait_graph
        self.engine.report_access(
            self.execution,
            thread_id,
            _SHARED_SYNC_ACQUIRE_KEY,
            "write",
        )
        for res_id in resource_ids:
            lock_int_id = self._row_lock_int_id(res_id)
            holder = self._active_row_locks.get(res_id)
            if holder is not None and holder != thread_id:
                # Another task holds this — check for deadlock cycle
                if graph is not None:
                    cycle = graph.add_waiting(thread_id, lock_int_id, kind="row_lock")
                    if cycle is not None:
                        graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                        desc = format_cycle(cycle, self._row_lock_registry.id_to_resource())
                        raise DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                    # No cycle but contention — remove waiting edge (we can't
                    # actually block in async), let the SQL proceed.  The DB
                    # will handle the actual blocking and lock_timeout safety
                    # net prevents indefinite hangs.
                    graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                # Even without a graph, record ownership optimistically so
                # that the DPOR engine can explore alternative interleavings.
            # Record ownership and notify graph — shared logic via registry.
            self._row_lock_registry.record_acquire(thread_id, res_id, graph)

    def release_row_locks(self, thread_id: int) -> None:
        """Release all row locks held by *thread_id* (called on COMMIT/ROLLBACK)."""
        graph = _async_wait_graph
        # Shared release logic via registry (sync also uses pop_all; async skips
        # engine.report_sync because row-lock release is tracked at await points).
        self._row_lock_registry.pop_all(thread_id, graph)

    def _flush_pending_io(self, task_id: int) -> None:
        """Flush pending I/O accesses to the DPOR engine."""
        pending = self._pending_io.get(task_id)
        if pending:
            for obj_key, kind, synced in pending:
                if synced:
                    self.engine.report_synced_io_access(self.execution, task_id, obj_key, kind)
                else:
                    self.engine.report_io_access(self.execution, task_id, obj_key, kind)
            pending.clear()

    def report_and_wait_sync(self, task_id: int) -> None:
        """Synchronous report-and-wait for use from SQL cursor interception.

        SQL cursor patching calls ``_get_dpor_context()`` which returns
        ``(scheduler, thread_id)``.  The sync DPOR scheduler has a
        ``report_and_wait(frame, thread_id)`` method.  For async DPOR,
        SQL interception needs a way to force a scheduling point
        synchronously (the cursor patch is called from inside an await).
        We just flush pending I/O here; the actual scheduling happens
        at the next ``await_point()``.
        """
        self._flush_pending_io(task_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Compatibility method for SQL cursor interception.

        The sync ``_sql_cursor.py`` and ``_sql_cursor_async.py`` call
        ``dpor_ctx[0].report_and_wait(None, thread_id)`` to force a
        scheduling point after SQL operations.  For async DPOR, we
        flush pending I/O but the actual scheduling happens at await
        points.  Returns True to indicate the task should continue.
        """
        self._flush_pending_io(thread_id)
        return True

    def finish_task(self, task_id: int) -> None:
        """Mark a task as finished in the DPOR engine."""
        self.execution.finish_thread(task_id)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


async def run_with_schedule_dpor(
    engine: PyDporEngine,
    execution: PyExecution,
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    timeout: float = 5.0,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
    detect_redis: bool = False,
) -> Any:
    """Run one async DPOR execution and return the state object.

    Args:
        engine: The DPOR engine instance.
        execution: The current execution instance from the engine.
        setup: Returns fresh shared state.
        tasks: Async callables that each receive the state as their argument.
        timeout: Max seconds.
        deadlock_timeout: Seconds to wait before declaring a deadlock.
        detect_sql: If True, patch async DBAPI drivers for SQL tracking.
        detect_redis: If True, patch async Redis clients for key-level
            conflict detection.

    Returns:
        The state object after execution.
    """
    num_tasks = len(tasks)
    scheduler = AsyncDporScheduler(
        engine,
        execution,
        num_tasks,
        deadlock_timeout=deadlock_timeout,
        detect_sql=detect_sql,
        detect_redis=detect_redis,
    )

    state = setup()

    task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
        i: (lambda s=state, t=t: t(s))  # type: ignore[assignment]
        for i, t in enumerate(tasks)
    }

    try:
        await scheduler.run_all(task_funcs, timeout=timeout)  # type: ignore[arg-type]
    except TimeoutError:
        pass

    # Mark any unfinished tasks as done in the DPOR engine
    for i in range(num_tasks):
        if i not in scheduler._tasks_done:
            scheduler.finish_task(i)

    return state


def _format_async_trace(schedule: list[int], num_tasks: int) -> str:
    """Generate a human-readable explanation of an async DPOR interleaving.

    Converts the raw schedule (list of task IDs at each scheduling point)
    into a readable description of task switches, making it easier to
    understand the interleaving that caused an invariant violation.
    """
    if not schedule:
        return "Invariant violation detected (empty schedule)."

    lines: list[str] = []
    lines.append("Invariant violation found after exploring interleaving schedule.")
    lines.append(f"Tasks: {num_tasks}, Schedule steps: {len(schedule)}")
    lines.append("")
    lines.append("Task interleaving (task ID at each scheduling point):")

    runs = group_schedule_runs(schedule)

    for i, (tid, count) in enumerate(runs):
        step_label = "step" if count == 1 else "steps"
        lines.append(f"  [{i + 1}] Task {tid}: {count} {step_label}")

    return "\n".join(lines)


class _ReplayAsyncScheduler(InterleavedLoop):
    """Replay a fixed schedule for async counterexample reproduction."""

    def __init__(
        self,
        schedule: list[int],
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_actor_id: int | None = None,
    ) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self._replay_schedule = list(schedule)
        # Start at index 1 because schedule[0] is consumed as the initial _current_task.
        self._replay_index = 1 if schedule else 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        self._num_replay_tasks = num_tasks
        # Virtual clock replay state (see AsyncDporScheduler): recorded
        # clock-actor entries in the schedule become clock advances.
        self.virtual_clock = virtual_clock
        self._clock_mode = clock_mode
        self._clock_actor_id = clock_actor_id
        self._sleepers: dict[int, float] = {}
        # Recorded actor entries reached before any deadline was registered
        # (drift): the owed advance is performed at the next registration.
        self._pending_clock_advances = 0
        self._current_task: int | None = None
        if schedule:
            first = schedule[0]
            if clock_actor_id is not None and first == clock_actor_id:
                # A leading actor entry cannot advance anything yet (no
                # deadlines registered — a stale wakeup-tree step); skip it
                # without owing an advance.
                self._current_task = None
                self._replay_index = 0
                self._advance()
                self._pending_clock_advances = 0
            else:
                self._current_task = first
        # Stubs so the patched cooperative asyncio.Lock can call
        # engine.report_sync / execution.block_thread without crashing during
        # replay (finding F3).  _lock_blocked mirrors the DPOR scheduler's
        # attribute so the same lock-acquire code path works unmodified.
        self.engine: Any = ReplayEngine()
        self.execution: Any = ReplayExecution()
        self._lock_blocked: dict[int, int] = {}

    def _extend_schedule(self) -> bool:
        return extend_replay_schedule(
            self._replay_schedule,
            self._replay_index,
            self._replay_max_ops,
            self._num_replay_tasks,
            self._tasks_done,
        )

    def _replay_advance_clock(self, target: float | None = None) -> None:
        """Advance the virtual clock during replay and wake due deadlines."""
        clock = self.virtual_clock
        if clock is None:
            return
        pending = list(self._sleepers.values())
        if not pending:
            return
        clock.advance_to(min(pending) if target is None else target)
        now = clock.now()
        for tid, dl in list(self._sleepers.items()):
            if dl <= now:
                del self._sleepers[tid]

    def _advance(self) -> None:
        """Advance ``_replay_index`` and ``_current_task`` to the next live actor."""
        while True:
            self._replay_index, next_actor = advance_replay_index(
                self._replay_schedule,
                self._replay_index,
                self._extend_schedule,
                self._tasks_done,
            )
            if next_actor is not None and self._clock_actor_id is not None and next_actor == self._clock_actor_id:
                # Recorded clock-actor step: advance the clock and keep going.
                if self._sleepers:
                    self._replay_advance_clock()
                else:
                    # Drift: the sleeper has not registered yet — owe the
                    # advance and perform it on registration (sleep_until).
                    self._pending_clock_advances += 1
                continue
            break
        self._current_task = next_actor
        if next_actor is None:
            self._finished = True

    async def sleep_until(self, task_id: int, deadline: float) -> None:
        """Replay counterpart of ``AsyncDporScheduler.sleep_until``."""
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await _real_asyncio_sleep(0)
            self._progress += 1
            async with self._condition:
                if self._finished or self._error:
                    return
                self._sleepers[task_id] = deadline
                if self._pending_clock_advances > 0:
                    # Replay owed us an actor step that arrived before this
                    # registration (drift): perform it now.
                    self._pending_clock_advances -= 1
                    self._replay_advance_clock()
                if self._current_task == task_id:
                    self._advance()
                self._condition.notify_all()
                try:
                    while task_id in self._sleepers:
                        if self._finished or self._error:
                            return
                        alive = [t for t in range(self._num_replay_tasks) if t not in self._tasks_done]
                        if alive and all(t in self._sleepers for t in alive):
                            # Every live task is asleep: only time can move.
                            self._replay_advance_clock()
                            self._condition.notify_all()
                            continue
                        try:
                            await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                        except asyncio.TimeoutError:
                            self._error = TimeoutError(
                                f"Replay deadlock: task {task_id} sleeping until t={deadline} was never woken"
                            )
                            self._condition.notify_all()
                            return
                finally:
                    self._sleepers.pop(task_id, None)
        finally:
            _in_scheduler_pause.set(depth)

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        # Schedule-drift safety net: if replay scheduled a sleeping task,
        # time must pass before it can move — jump to its deadline.
        cur = self._current_task
        if cur is not None and self.virtual_clock is not None and cur in self._sleepers:
            self._replay_advance_clock(self._sleepers[cur])
            self._condition.notify_all()
        if self._current_task is None:
            self._finished = True
            return True
        if self._current_task == task_id:
            return True
        # Mirror AsyncDporScheduler.should_proceed: if the currently-scheduled
        # task is blocked on a lock held by task_id, let task_id proceed so it
        # can release the lock.  The recorded schedule contains the engine's
        # raw pick of the blocked task (the exploration-time holder override is
        # not written to the trace), so without this override the picked task is
        # stuck inside the real lock.acquire() and replay deadlocks.
        if self._current_task in self._lock_blocked:
            holder = self._lock_blocked[self._current_task]
            if holder == task_id:
                return True
        return False

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        self._advance()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)

    def _cleanup_task_context(self, task_id: Any) -> None:
        # Release any asyncio.Lock objects still held by this task (e.g. the
        # task crashed or was cancelled without release()).  Without this,
        # stale holding edges / lock owners leak into later replay attempts
        # and cause spurious DeadlockError or phantom ownership (finding F5).
        _release_task_async_locks(task_id)
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

    def finish_task(self, task_id: int) -> None:
        self._tasks_done.add(task_id)

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished and update _current_task if needed."""
        async with self._condition:
            self._tasks_done.add(task_id)
            if self._current_task == task_id:
                # Advance to the next scheduled task so other tasks can proceed.
                self._advance()
            self._condition.notify_all()


async def _reproduce_async_counterexample(
    schedule_list: list[int],
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool] | None,
    num_tasks: int,
    reproduce_on_failure: int,
    timeout_per_run: float,
    deadlock_timeout: float,
    clock: ClockMode = "real",
) -> tuple[int, int]:
    """Measure how often an async DPOR counterexample reproduces."""
    successes = 0
    for _ in range(reproduce_on_failure):
        # Clear cooperative-lock global state before EACH replay attempt (F5).
        _reset_async_lock_state()

        replay_clock = VirtualClock() if clock != "real" else None
        scheduler = _ReplayAsyncScheduler(
            schedule_list,
            num_tasks,
            deadlock_timeout=deadlock_timeout,
            virtual_clock=replay_clock,
            clock_mode=clock,
            clock_actor_id=num_tasks if clock != "real" else None,
        )
        with clock_context(replay_clock):
            state = setup()
        task_funcs: dict[int, Callable[..., Awaitable[None]]] = {}
        for i, task in enumerate(tasks):

            def _make_task(task_fn: Callable[[T], Coroutine[Any, Any, None]]) -> Callable[..., Awaitable[None]]:
                def _wrapped_task() -> Awaitable[None]:
                    return task_fn(state)

                return _wrapped_task

            task_funcs[i] = _make_task(task)

        task_funcs = wrap_auto_paused_tasks(task_funcs, scheduler)

        deadlocked = False
        try:
            with clock_context(replay_clock):
                await scheduler.run_all(task_funcs, timeout=timeout_per_run)
        except DeadlockError:
            deadlocked = True
        except TimeoutError:
            # Run didn't complete within the budget — a failed reproduction.
            continue
        except (AttributeError, TypeError, NameError):
            # Programming errors in our own replay plumbing (e.g. a missing
            # scheduler attribute) must NOT be silently scored as a failed
            # reproduction — surface them instead of hiding the bug (F3).
            raise
        except Exception:
            # The user's task body raised: the run produced no usable state,
            # so this attempt did not reproduce the counterexample.
            continue
        with clock_context(replay_clock):
            inv_failed, _ = (
                check_invariant(invariant, state) if (invariant is not None and not deadlocked) else (False, None)
            )
        if is_reproduction_run(deadlocked=deadlocked, has_invariant=invariant is not None, invariant_failed=inv_failed):
            successes += 1
    return reproduce_on_failure, successes


async def _explore_async_dpor(  # pyright: ignore[reportUnusedFunction]  # called cross-module by frontrun._strategy and contrib helpers
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
    detect_redis: bool = False,
    trace_packages: list[str] | None = None,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[T], Any] | bool = False,
    error_on_any_race: bool = False,
    clock: ClockMode = "real",
) -> InterleavingResult:
    """Systematically explore async interleavings using DPOR.

    Async DPOR implementation; called via :func:`frontrun.explore` with async
    workers.  Instead of threads, it runs async tasks with ``await_point()`` as
    the scheduling granularity.
    The Rust DPOR engine systematically explores every distinct interleaving,
    using vector clocks to prune redundant orderings.

    When ``detect_sql=True``, async database drivers (asyncpg, aiosqlite,
    psycopg AsyncCursor, aiomysql) are monkey-patched to report SQL
    table-level accesses to the DPOR engine.  This enables DPOR to detect
    SQL-level conflicts (e.g. two tasks writing the same table) and explore
    their orderings.

    When ``detect_redis=True``, async Redis clients (redis.asyncio, coredis)
    are monkey-patched to report key-level accesses to the DPOR engine.

    Args:
        setup: Creates fresh shared state for each execution.
        tasks: List of async callables, each receiving the shared state.
        invariant: Predicate over shared state; must be True after all
            tasks complete.
        max_executions: Safety limit on total executions (None = unlimited).
        preemption_bound: Limit on preemptions per execution. 2 catches most
            bugs. None = unbounded (full DPOR).
        max_branches: Maximum scheduling points per execution.
        timeout_per_run: Timeout for each individual run.
        stop_on_first: If True (default), stop on first invariant violation.
        deadlock_timeout: Seconds to wait before declaring a deadlock.
        detect_sql: If True, patch async DBAPI drivers for SQL tracking.
        detect_redis: If True, patch async Redis clients for key-level
            conflict detection.
        trace_packages: List of package name patterns (fnmatch syntax) to
            trace in addition to user code.  By default, code in
            site-packages is skipped.  Use this to include specific
            installed packages, e.g. ``["django_*", "mylib.*"]``.
        reproduce_on_failure: When a counterexample is found, replay the
            same schedule this many times to measure reproducibility.
            Set to 0 to skip.
        total_timeout: Maximum total time in seconds for the entire
            exploration.  None means no global deadline.
        warn_nondeterministic_sql: If True (default), raise
            :class:`~frontrun.common.NondeterministicSQLError` when SQL
            INSERT statements are detected but ``lastrowid`` capture
            failed (e.g. psycopg2 without RETURNING).  Set to False to
            suppress.  When capture succeeds, INSERTs use stable
            indexical resource IDs automatically.
        lock_timeout: If set, automatically execute
            ``SET lock_timeout = '<N>ms'`` on every new PostgreSQL
            connection created through the patched ``psycopg2.connect``
            (or ``psycopg.connect``).  This prevents the cooperative
            scheduler from deadlocking when two tasks contend on the
            same PostgreSQL row lock.  Value is in milliseconds;
            2000 (2 seconds) is a good default.
        patch_sleep: If True (default), ``asyncio.sleep`` yields to the
            scheduler instead of waiting.  Required for ``clock != "real"``.
        serializable_invariant: Check serializability against sequential
            runs.  Cannot be combined with a virtual clock.
        error_on_any_race: Treat unsynchronized races as failures.
        clock: ``"real"`` (default), ``"virtual"`` (autojump virtual clock),
            or ``"explored"`` (clock advances become schedulable DPOR steps
            via a synthetic clock-actor task).  ``asyncio.wait_for`` /
            ``asyncio.timeout`` stay on the wall clock.  See
            :doc:`/virtual_clock`.

    Returns:
        InterleavingResult with exploration statistics and any counterexample.
    """
    clock = validate_clock_options(clock, patch_sleep=patch_sleep, serializable_invariant=serializable_invariant)
    if trace_packages is not None:
        _set_active_trace_filter(_TraceFilter(trace_packages))

    # Compute serializable baseline if requested.
    serial_valid_states, serial_hash_fn = await compute_serializable_baseline_async(
        setup, tasks, serializable_invariant
    )

    num_tasks = len(tasks)
    # With a virtual clock the engine gets one extra thread — the clock actor
    # (id == num_tasks); see AsyncDporScheduler.
    clock_actor_id = num_tasks if clock != "real" else None
    engine = make_dpor_engine(
        num_threads=num_tasks + (1 if clock_actor_id is not None else 0),
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
    )

    result = InterleavingResult(property_holds=True)
    stable_ids = StableObjectIds()
    total_deadline = make_deadline(total_timeout)

    clear_sql_metadata()

    # Inject SET lock_timeout on new PG connections (defect #6 workaround).
    prev_lock_timeout = get_lock_timeout()
    if lock_timeout is not None:
        set_lock_timeout(lock_timeout)

    async def _record_reproduction(
        schedule_list: list[int],
        invariant_fn: Callable[[T], bool] | None,
    ) -> None:
        if reproduce_on_failure <= 0 or result.reproduction_attempts != 0:
            return
        attempts, successes = await _reproduce_async_counterexample(
            schedule_list=schedule_list,
            setup=setup,
            tasks=tasks,
            invariant=invariant_fn,
            num_tasks=num_tasks,
            reproduce_on_failure=reproduce_on_failure,
            timeout_per_run=timeout_per_run,
            deadlock_timeout=deadlock_timeout,
            clock=clock,
        )
        result.reproduction_attempts = attempts
        result.reproduction_successes = successes

    # Single-threaded async DPOR doesn't need a real engine lock; the
    # shared NoOpLock keeps the dpor_exploration_iter contract uniform
    # across the sync (threading.Lock) and async drivers.
    engine_lock = NoOpLock()

    # With a virtual clock, pin the event loop's own clock to real monotonic
    # time.  ``BaseEventLoop.time()`` calls ``time.monotonic()``, which is
    # patched; without the pin, a loop timer scheduled from a task context
    # (where the contextvar-gated patch resolves to virtual time) would carry
    # a virtual ``when`` that the loop compares against wall-clock time —
    # deadlock-timeout timers would then never fire.  Loop timers (and
    # therefore asyncio.wait_for / asyncio.timeout deadlines) deliberately
    # stay on the wall clock; see docs/virtual_clock.rst.
    _loop = asyncio.get_running_loop()
    _loop_time_pinned = False
    _user_timers_check: Callable[[], bool] | None = None
    _untag_timers: Callable[[], None] | None = None
    if clock != "real":
        _loop.time = real_monotonic  # type: ignore[method-assign]
        _loop_time_pinned = True
        # Tag frontrun's own watchdog timers so exact deadlock detection can
        # tell them apart from user timers (see _install_frontrun_timer_tagging).
        _user_timers_check, _untag_timers = _install_frontrun_timer_tagging(_loop)

    try:
        with PatchScope() as patch_scope:
            patch_scope.add(patch_sql_async, unpatch_sql_async, enabled=detect_sql and _sql_async_available)
            patch_scope.add(patch_redis_async, unpatch_redis_async, enabled=detect_redis and _redis_async_available)
            patch_scope.add(_patch_asyncio_lock, _unpatch_asyncio_lock)
            patch_scope.add(_patch_asyncio_event, _unpatch_asyncio_event)
            patch_scope.add(_patch_asyncio_sleep, _unpatch_asyncio_sleep, enabled=patch_sleep)
            patch_scope.add(patch_time, unpatch_time, enabled=clock != "real")

            for step in dpor_exploration_iter(
                engine=engine,
                engine_lock=engine_lock,
                stable_ids=stable_ids,
                total_deadline=total_deadline,
            ):
                execution = step.execution

                # Clear wait-for graph and held-locks tracking between executions
                _reset_async_lock_state()

                # Fresh virtual clock per execution so every interleaving
                # starts from the same deterministic epoch.
                virtual_clock = VirtualClock() if clock != "real" else None
                scheduler = AsyncDporScheduler(
                    engine,
                    execution,
                    num_tasks,
                    deadlock_timeout=deadlock_timeout,
                    detect_sql=detect_sql,
                    detect_redis=detect_redis,
                    stable_ids=stable_ids,
                    virtual_clock=virtual_clock,
                    clock_mode=clock,
                    clock_actor_id=clock_actor_id,
                    user_timers_pending=_user_timers_check,
                )

                with clock_context(virtual_clock):
                    state = setup()
                stable_ids.pre_register(state)

                task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
                    i: (lambda s=state, t=t: t(s))  # type: ignore[assignment]
                    for i, t in enumerate(tasks)
                }

                deadlock_error: DeadlockError | None = None
                task_error: Exception | None = None
                timed_out = False
                try:
                    # clock_context propagates to the worker tasks: contexts
                    # copy at create_task time (inside run_all).
                    with clock_context(virtual_clock):
                        await scheduler.run_all(task_funcs, timeout=timeout_per_run)  # type: ignore[arg-type]
                except DeadlockError as e:
                    deadlock_error = e
                except TimeoutError:
                    timed_out = True
                except Exception as e:
                    # Task raised an exception (not deadlock/timeout).
                    # This is a valid exploration outcome — the cleanup already
                    # happened in _run's finally block, so lock state is clean.
                    # Record it and check the invariant below.
                    task_error = e

                # Mark any unfinished tasks as done in the DPOR engine
                unfinished = [i for i in range(num_tasks) if i not in scheduler._tasks_done]
                for i in unfinished:
                    scheduler.finish_task(i)

                result.num_explored += 1

                # Check for deadlock: explicit DeadlockError from wait-for
                # graph cycle detection, or timeout (tasks blocked on locks
                # get cancelled by run_all and appear "done" via _mark_done
                # in the finally block, so we can't rely on unfinished alone).
                is_deadlock = False
                deadlock_explanation = ""
                if deadlock_error is not None:
                    is_deadlock = True
                    deadlock_explanation = f"Deadlock detected: {deadlock_error.cycle_description}"
                elif timed_out:
                    is_deadlock = True
                    if unfinished:
                        stuck = ", ".join(str(t) for t in unfinished)
                        deadlock_explanation = (
                            f"Deadlock detected: tasks [{stuck}] did not complete. "
                            f"All tasks were blocked and could not make progress."
                        )
                    else:
                        deadlock_explanation = (
                            "Deadlock detected: all tasks were blocked and timed out. "
                            "Tasks could not make progress (likely waiting on locks held by each other)."
                        )

                if is_deadlock:
                    schedule_list = record_dpor_failure(result, list(execution.schedule_trace), deadlock_explanation)
                    await _record_reproduction(schedule_list, None)
                    if stop_on_first:
                        return result
                elif task_error is not None:
                    exc_type = type(task_error).__name__
                    record_dpor_failure(
                        result,
                        list(execution.schedule_trace),
                        f"Task crash in execution {result.num_explored}: {exc_type}: {task_error}",
                    )
                    if stop_on_first:
                        return result

                if warn_nondeterministic_sql:
                    check_uncaptured_inserts()

                # --- error_on_any_race: treat unsynchronized races as failures ---
                if error_on_any_race and not is_deadlock and task_error is None:
                    raw_races_check = engine.attribute_races()
                    if raw_races_check:
                        record_dpor_failure(
                            result,
                            list(execution.schedule_trace),
                            format_race_failure_explanation(
                                result.num_explored,
                                len(raw_races_check),
                                actor_plural="tasks",
                            ),
                            races_detected=True,
                        )
                        if stop_on_first:
                            return result

                # --- serializable_invariant: check against sequential baselines ---
                if serial_valid_states is not None and not is_deadlock and task_error is None:
                    ser_explanation = check_serializability_violation(
                        state, serial_valid_states, serial_hash_fn, result.num_explored
                    )
                    if ser_explanation is not None:
                        record_dpor_failure(result, list(execution.schedule_trace), ser_explanation)
                        if stop_on_first:
                            return result

                if not is_deadlock and task_error is None:
                    with clock_context(virtual_clock):
                        invariant_failed, assertion_msg = check_invariant(invariant, state)
                    if invariant_failed:
                        schedule_list = list(execution.schedule_trace)
                        trace_explanation = _format_async_trace(schedule_list, num_tasks)
                        explanation = (
                            f"AssertionError: {assertion_msg}\n\n{trace_explanation}"
                            if assertion_msg
                            else trace_explanation
                        )
                        schedule_list = record_dpor_failure(result, schedule_list, explanation)
                        await _record_reproduction(schedule_list, invariant)
                        if stop_on_first:
                            return result
    finally:
        if trace_packages is not None:
            _set_active_trace_filter(None)
        set_lock_timeout(prev_lock_timeout)
        if _untag_timers is not None:
            _untag_timers()
        if _loop_time_pinned:
            del _loop.time  # restore BaseEventLoop.time

    return result
