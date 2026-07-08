"""Cooperative asyncio primitives for async DPOR / random exploration.

Async mirror of the sync ``_cooperative.py``: drop-in replacements for
``asyncio.Lock`` / ``asyncio.Event`` / ``asyncio.Queue`` / ``asyncio.Condition``
whose blocking points are made visible to the DPOR (or replay) scheduler.

A task parked on a stock async primitive is invisible to the engine: it still
looks runnable, so the scheduler keeps picking it, no other task ever gets the
turn, and the run dies by wall timeout — scored as a false deadlock
counterexample.  These wrappers engine-block their waiters, hand the turn
onward, and report the wake happens-before edge so the exploration stays
faithful.

The scheduler is reached through ``_scheduler_var`` (async DPOR runs every task
on one event-loop thread, so scheduler identity is contextvar- rather than
thread-backed).  Scheduler methods used here are looked up with ``getattr`` so
the same primitive code drives both the exploration scheduler and the replay
scheduler.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from frontrun._async_autopause import _in_scheduler_pause, _scheduler_var, _task_id_var
from frontrun._deadlock import DeadlockError, WaitForGraph, format_cycle
from frontrun._dpor_core import event_wake_sync_id
from frontrun._opcode_observer import _make_object_key
from frontrun.async_scheduler import SchedulerTimeoutError

__all__ = [
    "_CooperativeAsyncCondition",
    "_CooperativeAsyncEvent",
    "_CooperativeAsyncLock",
    "_CooperativeAsyncQueue",
    "_async_parked_conditions",
    "_async_parked_events",
    "_async_parked_queues",
    "_async_task_held_locks",
    "_async_wake_sync_id",
    "_patch_asyncio_event",
    "_patch_asyncio_lock",
    "_patch_asyncio_queue_condition",
    "_real_asyncio_condition",
    "_real_asyncio_sleep",
    "_release_task_async_locks",
    "_reset_async_lock_state",
    "_unpatch_asyncio_event",
    "_unpatch_asyncio_lock",
    "_unpatch_asyncio_queue_condition",
    "_wake_parked_async_event_waiters",
    "_wake_parked_async_primitive_waiters",
]

# Saved originals, captured before any patching runs.
_real_asyncio_lock = asyncio.Lock
_real_asyncio_event = asyncio.Event
_real_asyncio_queue = asyncio.Queue
_real_asyncio_condition = asyncio.Condition
_real_asyncio_sleep = asyncio.sleep

_async_lock_patched = False

# Per-lock wait-for graph for async DPOR deadlock detection.
_async_wait_graph: WaitForGraph | None = None
# Map from lock id(obj) → owning task_id
_async_lock_owners: dict[int, int] = {}
# Reverse map: task_id → set of lock objects held by that task.
# Used to force-release locks when a task finishes without calling release().
_async_task_held_locks: dict[int, set[Any]] = {}

# Cooperative primitives that currently have at least one task parked in a
# blocking call.  When an abort (exact deadlock / watchdog timeout) fires,
# these waiters must be woken so the tasks free-run to completion and the
# outer driver surfaces the real error instead of hanging to the wall timeout.
_async_parked_events: set[_CooperativeAsyncEvent] = set()
_async_parked_queues: set[_CooperativeAsyncQueue[Any]] = set()
_async_parked_conditions: set[_CooperativeAsyncCondition] = set()


# ---------------------------------------------------------------------------
# Shared state-access / wake-sync helpers (single copy, shared by all primitives)
# ---------------------------------------------------------------------------


def _report_state_access(obj: object, suffix: str, kind: str) -> None:
    """Report an access to a primitive's synthetic ``__*_state__`` object.

    Shared by every cooperative primitive; ``suffix`` selects the per-type
    state object (``"__lock_state__"`` / ``"__event_state__"``).
    """
    task_id = _task_id_var.get()
    scheduler = _scheduler_var.get()
    engine = getattr(scheduler, "engine", None)
    if scheduler is None or engine is None or task_id is None or scheduler._error is not None:
        return
    stable_ids = scheduler._stable_ids
    obj_id = stable_ids.get(obj) if stable_ids is not None else id(obj)
    key = _make_object_key(obj_id, suffix)
    scheduler.report_task_access(task_id, key, kind)


def _async_wake_sync_id(scheduler: Any, obj: object, waiter: int) -> int:
    stable_ids = scheduler._stable_ids
    obj_id = stable_ids.get(obj) if stable_ids is not None else id(obj)
    return event_wake_sync_id(obj_id, waiter)


async def _engine_parked_wait(
    scheduler: Any,
    task_id: int,
    fut: Awaitable[Any],
    *,
    parked_set: set[Any],
    obj: Any,
    reason: str,
    cleanup: Callable[[], None],
    on_wake: Callable[[], Awaitable[None]] | None = None,
    on_finally: Callable[[], Awaitable[None]] | None = None,
) -> Any:
    """Shared park/wake protocol for cooperative Event/Queue/Condition waits.

    Centralises the sequence that was copy-pasted across ``Event.wait`` /
    ``Queue.get`` / ``Queue.put`` / ``Condition.wait``: register in *parked_set*
    → engine-``block_thread`` → bump ``_in_scheduler_pause`` → ``kick`` the turn
    onward → ``await`` the physical wake *fut* → ``unblock_thread`` →
    ``wait_until_scheduled_after_block`` → report the ``lock_acquire`` wake edge,
    with a finally that runs *cleanup*, unblocks if the wake never landed, runs
    *on_finally*, and finally restores the pause depth.

    The caller performs any primitive-specific setup (registering its waiter,
    and for ``Condition`` releasing its lock) *before* calling, and passes:

    - *cleanup*: removes the caller's waiter and drops *obj* from *parked_set*
      when none remain; runs exactly once in the finally.
    - *on_wake*: optional hook run after the physical unblock and before the
      reschedule wait — this is where ``Condition.wait`` re-acquires its lock
      (kept in the caller, not inlined here) and ``Event.wait`` self-removes its
      waiter before a possible concurrent ``set()``.
    - *on_finally*: optional hook run in the finally after the unblock-if-needed
      and before the pause-depth restore, still under the elevated pause depth —
      this is where ``Condition.wait`` runs its exception-safe lock re-acquire.
    """
    parked_set.add(obj)
    event_blocked = scheduler._event_blocked
    if event_blocked is not None:
        event_blocked.add(task_id)
    scheduler.execution.block_thread(task_id)
    depth = _in_scheduler_pause.get()
    _in_scheduler_pause.set(depth + 1)
    unblocked = False
    try:
        await scheduler.kick_stalled_schedule(task_id)
        result = await fut
        scheduler.execution.unblock_thread(task_id)
        unblocked = True
        if event_blocked is not None:
            event_blocked.discard(task_id)
        if on_wake is not None:
            await on_wake()
        if scheduler._error is None:
            await scheduler.wait_until_scheduled_after_block(task_id, reason)
        if scheduler._error is None:
            scheduler.report_task_sync(task_id, "lock_acquire", _async_wake_sync_id(scheduler, obj, task_id))
        return result
    finally:
        cleanup()
        if event_blocked is not None:
            event_blocked.discard(task_id)
        if not unblocked:
            scheduler.execution.unblock_thread(task_id)
        if on_finally is not None:
            await on_finally()
        _in_scheduler_pause.set(depth)


def _reset_async_lock_state() -> None:
    """Clear all cooperative-lock global state (graph edges, owners, held sets).

    Must run before each exploration execution AND each replay attempt: stale
    wait-for edges / lock owners / held-locks from a prior run (compounded by
    id() reuse) leak into the next one and cause spurious DeadlockError or
    phantom ownership.
    """
    if _async_wait_graph is not None:
        _async_wait_graph.clear()
    _async_lock_owners.clear()
    _async_task_held_locks.clear()
    _async_parked_events.clear()
    _async_parked_queues.clear()
    _async_parked_conditions.clear()


def _wake_parked_async_event_waiters() -> None:
    for event in list(_async_parked_events):
        event._event.set()


def _wake_parked_async_primitive_waiters() -> None:
    _wake_parked_async_event_waiters()
    for queue_obj in list(_async_parked_queues):
        queue_obj._wake_all_for_abort()
    for condition in list(_async_parked_conditions):
        condition._wake_all_for_abort()


# ---------------------------------------------------------------------------
# Cooperative asyncio.Lock with deadlock detection
# ---------------------------------------------------------------------------


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
        result = self._lock.locked()
        _report_state_access(self, "__lock_state__", "read")
        return result

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
        needs_cross_resource_pause = bool(
            scheduler is not None
            and (getattr(scheduler, "_detect_sql", False) or getattr(scheduler, "_detect_redis", False))
        )
        already_holds_lock = bool(task_id is not None and _async_task_held_locks.get(task_id))
        if (
            scheduler is not None
            and task_id is not None
            and _in_scheduler_pause.get() == 0
            and (self._lock.locked() or already_holds_lock or needs_cross_resource_pause)
        ):
            await scheduler.pause(task_id, ("lock_acquire", id(self)))

        lock_was_held = self._lock.locked()
        if task_id is not None and graph is not None and lock_was_held:
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
                if scheduler is not None:
                    await scheduler.kick_stalled_schedule(task_id)
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
                scheduler.report_task_sync(task_id, "lock_acquire", lock_id)
                _report_state_access(self, "__lock_state__", "weak_write")
                if not already_holds_lock and not lock_was_held and _in_scheduler_pause.get() == 0:
                    await scheduler.pause(task_id, ("lock_held", lock_id))
                    on_task_yielded = getattr(scheduler, "on_task_yielded", None)
                    if on_task_yielded is not None:
                        on_task_yielded(task_id)
                    await _real_asyncio_sleep(0)

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
                scheduler.report_task_sync(task_id, "lock_release", lock_id)
                _report_state_access(self, "__lock_state__", "weak_write")
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
        result = self._event.is_set()
        _report_state_access(self, "__event_state__", "read")
        return result

    def clear(self) -> None:
        _report_state_access(self, "__event_state__", "write")
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
            _report_state_access(self, "__event_state__", "read")
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

        def _drop_waiter() -> None:
            # set() iterates _waiters without removing, so a woken waiter must
            # self-remove before a possible concurrent second set() re-reports it.
            if task_id in self._waiters:
                self._waiters.remove(task_id)
            if not self._waiters:
                _async_parked_events.discard(self)

        async def _on_wake() -> None:
            _drop_waiter()

        return await _engine_parked_wait(
            scheduler,
            task_id,
            self._event.wait(),
            parked_set=_async_parked_events,
            obj=self,
            reason="event wait",
            cleanup=_drop_waiter,
            on_wake=_on_wake,
        )

    def set(self) -> None:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        engine = getattr(scheduler, "engine", None)
        if scheduler is not None and engine is not None and task_id is not None and scheduler._error is None:
            _report_state_access(self, "__event_state__", "write")
            for waiter in list(self._waiters):
                scheduler.report_task_sync(task_id, "lock_release", _async_wake_sync_id(scheduler, self, waiter))
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
# Cooperative asyncio.Queue / asyncio.Condition
# ---------------------------------------------------------------------------


class _CooperativeAsyncQueue(_real_asyncio_queue):  # type: ignore[misc,valid-type]
    """asyncio.Queue wrapper whose get/put waiters are visible to DPOR."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._frontrun_get_waiters: list[tuple[int, asyncio.Future[None]]] = []
        self._frontrun_put_waiters: list[tuple[int, asyncio.Future[None]]] = []

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        return cls

    def _pop_waiter(self, waiters: list[tuple[int, asyncio.Future[None]]]) -> tuple[int, asyncio.Future[None]] | None:
        while waiters:
            waiter, fut = waiters.pop(0)
            if not fut.done():
                return waiter, fut
        return None

    def _wake_waiter(
        self,
        waiters: list[tuple[int, asyncio.Future[None]]],
        scheduler: Any,
        task_id: int | None,
    ) -> None:
        waiter_info = self._pop_waiter(waiters)
        if waiter_info is None:
            return
        waiter, fut = waiter_info
        if scheduler is not None and task_id is not None:
            scheduler.report_task_sync(task_id, "lock_release", _async_wake_sync_id(scheduler, self, waiter))
            scheduler.execution.unblock_thread(waiter)
        fut.set_result(None)

    def _wake_all_for_abort(self) -> None:
        for _waiter, fut in self._frontrun_get_waiters + self._frontrun_put_waiters:
            if not fut.done():
                fut.set_result(None)

    def get_nowait(self) -> Any:
        item = super().get_nowait()
        self._wake_waiter(self._frontrun_put_waiters, _scheduler_var.get(), _task_id_var.get())
        if not self._frontrun_get_waiters and not self._frontrun_put_waiters:
            _async_parked_queues.discard(self)
        return item

    def put_nowait(self, item: Any) -> None:
        super().put_nowait(item)
        self._wake_waiter(self._frontrun_get_waiters, _scheduler_var.get(), _task_id_var.get())
        if not self._frontrun_get_waiters and not self._frontrun_put_waiters:
            _async_parked_queues.discard(self)

    async def get(self) -> Any:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        if scheduler is None or task_id is None:
            return await super().get()
        await scheduler.pause(task_id, ("queue_get", id(self)))
        while self.empty():
            if scheduler._error is not None:
                raise SchedulerTimeoutError("queue get aborted by scheduler")
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._frontrun_get_waiters.append((task_id, fut))

            def _cleanup(fut: asyncio.Future[None] = fut) -> None:
                self._frontrun_get_waiters = [
                    (waiter, waiter_fut) for waiter, waiter_fut in self._frontrun_get_waiters if waiter_fut is not fut
                ]
                if not self._frontrun_get_waiters and not self._frontrun_put_waiters:
                    _async_parked_queues.discard(self)

            await _engine_parked_wait(
                scheduler,
                task_id,
                fut,
                parked_set=_async_parked_queues,
                obj=self,
                reason="queue get",
                cleanup=_cleanup,
            )
            if not self.empty():
                break
        return self.get_nowait()

    async def put(self, item: Any) -> None:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        if scheduler is None or task_id is None:
            await super().put(item)
            return
        await scheduler.pause(task_id, ("queue_put", id(self)))
        while self.full():
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._frontrun_put_waiters.append((task_id, fut))

            def _cleanup(fut: asyncio.Future[None] = fut) -> None:
                self._frontrun_put_waiters = [
                    (waiter, waiter_fut) for waiter, waiter_fut in self._frontrun_put_waiters if waiter_fut is not fut
                ]
                if not self._frontrun_get_waiters and not self._frontrun_put_waiters:
                    _async_parked_queues.discard(self)

            await _engine_parked_wait(
                scheduler,
                task_id,
                fut,
                parked_set=_async_parked_queues,
                obj=self,
                reason="queue put",
                cleanup=_cleanup,
            )
        self.put_nowait(item)


class _CooperativeAsyncCondition:
    """asyncio.Condition wrapper with engine-visible wait/notify."""

    def __init__(self, lock: Any | None = None) -> None:
        # Default to an engine-visible cooperative lock: since patching makes
        # asyncio.Condition() produce this class, a raw asyncio.Lock default
        # would be invisible to the DPOR engine, so a task contending on
        # `async with cond:` would park in an unmodelled acquire while the
        # engine still treats it as runnable (inconclusive timeout / false
        # counterexample).  A user-supplied lock is honoured as-is.
        self._lock = lock if lock is not None else _CooperativeAsyncLock()
        # _CooperativeAsyncLock duck-types asyncio.Lock (used only on the
        # no-scheduler-context fallback path).
        self._real_condition = _real_asyncio_condition(self._lock)  # type: ignore[arg-type]
        self._waiters: list[tuple[int, asyncio.Future[None]]] = []

    def locked(self) -> bool:
        locked = getattr(self._lock, "locked", None)
        return bool(locked()) if locked is not None else False

    async def acquire(self) -> bool:
        return await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    async def __aenter__(self) -> _CooperativeAsyncCondition:
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.release()

    def _wake_all_for_abort(self) -> None:
        for _waiter, fut in self._waiters:
            if not fut.done():
                fut.set_result(None)

    async def wait(self) -> bool:
        if not self.locked():
            raise RuntimeError("cannot wait on un-acquired lock")
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        if scheduler is None or task_id is None:
            return await self._real_condition.wait()
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append((task_id, fut))
        # Release before parking so a notifier can take the lock; re-acquired on
        # the way out (below).  The park/block/report protocol is shared via
        # _engine_parked_wait; only the lock re-acquire stays here.
        self.release()
        acquired = False

        async def _reacquire_on_wake() -> None:
            nonlocal acquired
            await self.acquire()
            acquired = True

        def _cleanup() -> None:
            self._waiters = [(waiter, waiter_fut) for waiter, waiter_fut in self._waiters if waiter_fut is not fut]
            if not self._waiters:
                _async_parked_conditions.discard(self)

        async def _reacquire_exception_safe() -> None:
            if acquired:
                return
            # Mirror stock asyncio.Condition.wait: always re-acquire the lock
            # before propagating, even under cancellation.  Only this task's own
            # re-acquire counts — checking locked() is wrong because it reports
            # whether ANYONE holds the lock, so skipping re-acquire while another
            # task holds it would let the caller's `async with cond:` __aexit__
            # release that other task's lock.  Loop to shield the acquire itself
            # from cancellation, re-raising the caught CancelledError after.
            # _in_scheduler_pause stays elevated (_engine_parked_wait restores it
            # only after this hook) so the cooperative lock re-acquires without
            # inserting a fresh scheduling point.
            err: BaseException | None = None
            while True:
                try:
                    await self.acquire()
                    break
                except asyncio.CancelledError as exc:  # noqa: PERF203
                    err = exc
            if err is not None:
                try:
                    raise err
                finally:
                    err = None

        await _engine_parked_wait(
            scheduler,
            task_id,
            fut,
            parked_set=_async_parked_conditions,
            obj=self,
            reason="condition wait",
            cleanup=_cleanup,
            on_wake=_reacquire_on_wake,
            on_finally=_reacquire_exception_safe,
        )
        return True

    async def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> Any:
        if timeout is not None:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await self.wait()
        else:
            while not predicate():
                await self.wait()
        return predicate()

    def notify(self, n: int = 1) -> None:
        if not self.locked():
            raise RuntimeError("cannot notify on un-acquired lock")
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        budget = n
        if scheduler is None or task_id is None:
            # Wake real-condition waiters first, then count those against the
            # budget so the cooperative-waiter loop below cannot push the total
            # number of wakes past n (notify(1) with a mixed population used to
            # wake two).
            real_waiters = getattr(self._real_condition, "_waiters", ())
            pending_before = sum(1 for waiter_fut in real_waiters if not waiter_fut.done())
            self._real_condition.notify(n)
            pending_after = sum(1 for waiter_fut in real_waiters if not waiter_fut.done())
            budget = max(0, n - (pending_before - pending_after))
            if budget == 0 or not self._waiters:
                return
        woke = 0
        while self._waiters and woke < budget:
            waiter, fut = self._waiters.pop(0)
            if fut.done():
                continue
            if scheduler is not None and task_id is not None and scheduler._error is None:
                scheduler.report_task_sync(task_id, "lock_release", _async_wake_sync_id(scheduler, self, waiter))
                scheduler.execution.unblock_thread(waiter)
            fut.set_result(None)
            woke += 1
        if not self._waiters:
            _async_parked_conditions.discard(self)

    def notify_all(self) -> None:
        task_id = _task_id_var.get()
        scheduler = _scheduler_var.get()
        if scheduler is None or task_id is None:
            if not self.locked():
                raise RuntimeError("cannot notify on un-acquired lock")
            self._real_condition.notify_all()
            if not self._waiters:
                return
        self.notify(len(self._waiters))


def _patch_asyncio_queue_condition() -> None:
    asyncio.Queue = _CooperativeAsyncQueue  # type: ignore[assignment,misc]
    asyncio.Condition = _CooperativeAsyncCondition  # type: ignore[assignment,misc]


def _unpatch_asyncio_queue_condition() -> None:
    asyncio.Queue = _real_asyncio_queue  # type: ignore[assignment,misc]
    asyncio.Condition = _real_asyncio_condition  # type: ignore[assignment,misc]
    _async_parked_queues.clear()
    _async_parked_conditions.clear()
