"""
Shared cooperative threading primitives for frontrun.

Both bytecode.py (random exploration) and dpor.py (systematic DPOR) need
cooperative versions of threading/queue primitives that yield scheduler
turns instead of blocking in C.  This module provides a single set of
implementations used by both.

The key idea: when ``acquire()`` or ``wait()`` would block (lock held,
queue empty, event not set, …) the cooperative wrapper spins with
non-blocking attempts, calling ``scheduler.wait_for_turn(thread_id)``
between each attempt.  This gives the lock-holding / event-setting /
queue-producing thread a chance to execute opcodes and make progress.

An optional *sync reporter* callback (stored in thread-local storage)
lets DPOR report ``lock_acquire`` / ``lock_release`` events to the Rust
happens-before engine without changing the core spin-yield logic.

**Deadlock detection** — cooperative Lock and RLock register waiting/holding
edges in a global :class:`~frontrun._deadlock.WaitForGraph`.  If adding a
waiting edge creates a cycle, a :class:`~frontrun._deadlock.DeadlockError`
is raised immediately.  All spin loops also check ``scheduler._error``
eagerly (before each iteration) and bail via
:class:`~frontrun._deadlock.SchedulerAbort` when the scheduler has been
torn down.
"""

import math
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._deadlock import DeadlockError, SchedulerAbort, format_cycle

# The saved C-level time.monotonic itself (not the real_monotonic() wrapper):
# these deadline checks run in every spin-loop iteration.
from frontrun._virtual_clock import _active_virtual_clock, _real_monotonic

# ---------------------------------------------------------------------------
# Real (non-cooperative) factories, saved before any patching happens.
# ---------------------------------------------------------------------------

real_lock = _rt.lock
real_rlock = _rt.rlock
real_semaphore = _rt.semaphore
real_bounded_semaphore = _rt.bounded_semaphore
real_event = _rt.event
real_condition = _rt.condition
real_queue = _rt.queue_
real_lifo_queue = _rt.lifo_queue
real_priority_queue = _rt.priority_queue
make_real_event = _rt.make_event
make_real_queue = _rt.make_queue
make_real_lifo_queue = _rt.make_lifo_queue
make_real_priority_queue = _rt.make_priority_queue

# ---------------------------------------------------------------------------
# Thread-local scheduler context
# ---------------------------------------------------------------------------

_scheduler_tls = threading.local()


def get_context() -> tuple[Any, int] | None:
    """Return ``(scheduler, thread_id)`` from TLS, or ``None``."""
    scheduler = getattr(_scheduler_tls, "scheduler", None)
    thread_id = getattr(_scheduler_tls, "thread_id", None)
    if scheduler is not None and thread_id is not None:
        return scheduler, thread_id
    return None


def set_context(scheduler: Any, thread_id: int) -> None:
    """Store the active scheduler and thread id in TLS."""
    _scheduler_tls.scheduler = scheduler
    _scheduler_tls.thread_id = thread_id


def clear_context() -> None:
    """Remove the scheduler context from TLS."""
    _scheduler_tls.scheduler = None
    _scheduler_tls.thread_id = None


# ---------------------------------------------------------------------------
# Optional sync reporter (used by DPOR for happens-before tracking)
# ---------------------------------------------------------------------------

SyncReporter = Callable[[str, int, object], None]  # (event_name, object_id, lock_object) -> None


def get_sync_reporter() -> SyncReporter | None:
    """Return the per-thread sync reporter, or ``None``."""
    return getattr(_scheduler_tls, "sync_reporter", None)


def set_sync_reporter(reporter: SyncReporter | None) -> None:
    """Install a per-thread sync reporter (or clear with ``None``)."""
    _scheduler_tls.sync_reporter = reporter


def suppress_sync_reporting() -> None:
    """Suppress sync reporting for the current thread (for SQL internal locks).

    Supports nesting: each call increments a counter, and reporting is
    suppressed as long as the counter is positive.
    """
    depth = getattr(_scheduler_tls, "_sync_suppress_depth", 0)
    _scheduler_tls._sync_suppress_depth = depth + 1


def unsuppress_sync_reporting() -> None:
    """Decrement the sync suppression counter for the current thread."""
    depth = getattr(_scheduler_tls, "_sync_suppress_depth", 0)
    _scheduler_tls._sync_suppress_depth = max(0, depth - 1)


def is_sync_suppressed() -> bool:
    """Check if sync reporting is suppressed for the current thread."""
    return getattr(_scheduler_tls, "_sync_suppress_depth", 0) > 0


def _in_dpor_machinery() -> bool:
    """Return ``True`` if the current thread is already inside DPOR machinery.

    When this is set, cooperative locks fall back to real blocking to avoid
    reentrancy deadlocks (e.g., when GC triggers ``__del__`` during
    ``_process_opcode`` or ``_sync_reporter``).
    """
    return getattr(_scheduler_tls, "_in_dpor_machinery", False)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _check_lock_cycle(graph: Any, thread_id: int, object_id: int, scheduler: Any) -> None:
    """If *graph* contains a cycle after adding a waiting edge, raise SchedulerAbort.

    Must be called before the spin loop.  Removes the waiting edge and reports
    a :class:`~frontrun._deadlock.DeadlockError` via the scheduler if a cycle
    is found.
    """
    cycle = graph.add_waiting(thread_id, object_id)
    if cycle is not None:
        graph.remove_waiting(thread_id, object_id)
        # Pass the stable-ID mapping so the cycle description uses the same
        # integer lock IDs as the lock-event timeline in HTML reports.
        lock_id_map = getattr(getattr(scheduler, "_stable_ids", None), "_map", None)
        desc = format_cycle(cycle, lock_id_map=lock_id_map)
        scheduler.report_error(DeadlockError(f"Lock-ordering deadlock detected: {desc}", desc))
        raise SchedulerAbort(desc)


def _timed_acquire_state(
    timeout: float, scheduler: Any = None, thread_id: int | None = None
) -> tuple[float | None, Any, Any]:
    """Return ``(deadline, graph, clock)`` for a contended acquire.

    A timed acquire (``timeout >= 0``) cannot participate in a deadlock: it
    gives up after its deadline, which releases whatever locks the caller
    holds (the classic timeout-based avoidance pattern).  So it must NOT
    register a wait edge in the wait-for graph (that would create a spurious
    cycle) and must honor the deadline by returning ``False``.  ``graph`` is
    therefore ``None`` for timed acquires, suppressing all wait-edge
    bookkeeping in the spin loop.

    When the active scheduler owns a :class:`~frontrun._virtual_clock.VirtualClock`,
    the deadline is *virtual*: it is registered with the scheduler as a timed
    wait so the clock can advance to it when nothing else is runnable, making
    the timeout deterministic instead of host-speed-dependent.  ``clock`` is
    the virtual clock in that case (``None`` for wall-clock acquires).
    """
    from frontrun._deadlock import get_wait_for_graph

    if timeout >= 0:
        clock = getattr(scheduler, "virtual_clock", None) if scheduler is not None else None
        if clock is not None and thread_id is not None:
            deadline = clock.now() + timeout
            add_timed_wait = getattr(scheduler, "add_timed_wait", None)
            if add_timed_wait is None:
                # Degrading to a wall-clock deadline here would silently make
                # the timeout host-speed-dependent under a clock that promises
                # determinism.  Every shipped scheduler with a virtual clock
                # implements add_timed_wait, so this is a contract violation.
                raise RuntimeError(
                    f"scheduler {type(scheduler).__name__} exposes virtual_clock but not add_timed_wait; "
                    "virtual timed waits need both"
                )
            add_timed_wait(thread_id, deadline)
            return deadline, None, clock
        return _real_monotonic() + timeout, None, None
    return None, get_wait_for_graph(), None


def _timed_acquire_expired(deadline: float | None, clock: Any) -> bool:
    """Whether a timed acquire's deadline has passed (virtual or wall clock)."""
    if deadline is None:
        return False
    now = clock.now() if clock is not None else _real_monotonic()
    return now >= deadline


def _timed_acquire_cleanup(scheduler: Any, thread_id: int, clock: Any, *, gave_up: bool) -> None:
    """Deregister a virtual timed wait; on give-up, clear any engine block.

    A DPOR lock waiter is marked blocked in the engine (via the ``lock_wait``
    sync event); a waiter that acquires the lock is unblocked by the
    ``lock_acquire`` event, but one that *gives up* must unblock itself or the
    engine would never schedule it again.
    """
    if clock is None:
        return
    if gave_up:
        # Prefer the atomic unblock+deregister so the waiter is never left
        # engine-blocked with no pending deadline (a spurious exact-deadlock
        # window).  Fall back to the two-call path for schedulers that lack it.
        give_up_timed_wait = getattr(scheduler, "give_up_timed_wait", None)
        if give_up_timed_wait is not None:
            give_up_timed_wait(thread_id)
            return
        remove_timed_wait = getattr(scheduler, "remove_timed_wait", None)
        if remove_timed_wait is not None:
            remove_timed_wait(thread_id)
        clear_engine_block = getattr(scheduler, "clear_engine_block", None)
        if clear_engine_block is not None:
            clear_engine_block(thread_id)
        return
    remove_timed_wait = getattr(scheduler, "remove_timed_wait", None)
    if remove_timed_wait is not None:
        remove_timed_wait(thread_id)


def _finish_virtual_timed_wait(scheduler: Any, thread_id: int, deadline: float | None, clock: Any) -> bool:
    """Resolve a virtual timed wait once the scheduler has finished.

    After the random scheduler exhausts its schedule/op budget it grants no
    more turns, and the spin loops fall back to real waits.  Under a virtual
    clock that would freeze the clock and run the timeout on wall time, so the
    wait must instead observe its own virtual deadline: advance the clock to
    it (firing earlier due deadlines in order) and deregister the wait.
    Returns True when resolved virtually — the caller then takes its timeout
    branch — and False when no virtual deadline is registered or the scheduler
    lacks the advance hook (DPOR is engine-driven, with no positional budget
    to exhaust, and keeps its existing finished-path behaviour).
    """
    if clock is None or deadline is None:
        return False
    advance = getattr(scheduler, "advance_clock_after_finish", None)
    if advance is None:
        return False
    advance(deadline)
    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
    return True


def _timed_wait_deadline(timeout: float | None, scheduler: Any, thread_id: int) -> tuple[float, Any]:
    """Return ``(deadline, clock)`` for Event/Condition/Queue waits."""
    if timeout is None:
        return math.inf, None
    if timeout < 0:
        return _real_monotonic() + timeout, None
    deadline, _, clock = _timed_acquire_state(timeout, scheduler, thread_id)
    assert deadline is not None
    return deadline, clock


def _spin_hook_for_wait(scheduler: Any, timeout: float | None, clock: Any) -> Any | None:
    """Untimed waits and virtual timed waits must be visible as blocked spins.

    Virtual timed waits pass ``timed=True`` so a flag queued behind an autojump
    that already fired the wait's deadline is refused rather than landing stale
    (see ``VirtualClockPort.note_blocking_spin``).
    """
    if timeout is None or clock is not None:
        return _spin_note_hook(scheduler, timed=timeout is not None)
    return None


# Records, per resource and thread, the scheduler a managed waiter flagged a
# blocking spin with.  release()/set()/put() resolve from this recording rather
# than from the releaser's TLS context, so unmanaged helper threads can still
# clear random-strategy autojump flags and DPOR timed-wait engine blocks.
_spin_flag_schedulers: dict[int, dict[int, Any]] = {}
_spin_flag_lock = real_lock()


def _record_spin_scheduler(resource_id: int, thread_id: int, scheduler: Any) -> None:
    with _spin_flag_lock:
        _spin_flag_schedulers.setdefault(resource_id, {})[thread_id] = scheduler


def _forget_spin_scheduler(resource_id: int, thread_id: int) -> None:
    with _spin_flag_lock:
        schedulers = _spin_flag_schedulers.get(resource_id)
        if schedulers is None:
            return
        schedulers.pop(thread_id, None)
        if not schedulers:
            _spin_flag_schedulers.pop(resource_id, None)


def _purge_spin_schedulers(scheduler: Any) -> None:  # pyright: ignore[reportUnusedFunction]  # used by DPOR scheduler teardown
    """Drop registry entries that could retain or route to a finished scheduler."""
    with _spin_flag_lock:
        for resource_id, schedulers in list(_spin_flag_schedulers.items()):
            for thread_id, sched in list(schedulers.items()):
                if sched is scheduler:
                    del schedulers[thread_id]
            if not schedulers:
                del _spin_flag_schedulers[resource_id]


def _spin_schedulers_for(resource_id: int) -> list[Any]:
    """Distinct schedulers a waiter flagged a spin on *resource_id* with."""
    with _spin_flag_lock:
        found = list(_spin_flag_schedulers.get(resource_id, {}).values())
    deduped: dict[int, Any] = {}
    for scheduler in found:
        deduped.setdefault(id(scheduler), scheduler)
    return list(deduped.values())


def _spin_note_hook(scheduler: Any, *, timed: bool = False) -> Any | None:
    """``note_blocking_spin`` hook, if the scheduler has one *and* runs a
    virtual clock (random-strategy autojump needs to know about untimed
    spinners; see OpcodeScheduler._spin_waiters).  ``None`` otherwise.

    The returned hook records/forgets the scheduler on the shared spin-flag
    registry alongside flagging the spin, so a later release from an unmanaged
    thread can still resolve the scheduler that must clear it.  ``timed=True``
    (virtual timed waits) lets the scheduler refuse a flag whose deadline has
    already fired."""
    if getattr(scheduler, "virtual_clock", None) is None:
        return None
    note = getattr(scheduler, "note_blocking_spin", None)
    if note is None:
        return None

    def _hook(thread_id: int, resource_id: int, waiting: bool) -> None:
        if waiting:
            _record_spin_scheduler(resource_id, thread_id, scheduler)
        else:
            _forget_spin_scheduler(resource_id, thread_id)
        note(thread_id, resource_id, waiting, timed_wait=timed)

    return _hook


def _note_spin_release(resource_id: int) -> None:
    """Clear blocking-spin flags for *resource_id* on every scheduler a waiter
    recorded, so spinners re-probe before counting as blocked — and so an
    *unmanaged* releaser (no scheduler in TLS) still reaches them."""
    for scheduler in _spin_schedulers_for(resource_id):
        note = getattr(scheduler, "note_spin_release", None)
        if note is not None:
            note(resource_id)


def _record_holding(thread_id: int, object_id: int) -> None:
    """Add a holding edge for a just-acquired lock.

    Done even after a timed acquire (which registered no wait edge), so that
    other threads waiting on this holder are tracked correctly.
    """
    from frontrun._deadlock import get_wait_for_graph

    graph = get_wait_for_graph()
    if graph is not None:
        graph.add_holding(thread_id, object_id)


# ---------------------------------------------------------------------------
# Cooperative Lock
# ---------------------------------------------------------------------------


class CooperativeLock:
    """A Lock replacement that yields scheduler turns instead of blocking.

    When ``acquire()`` would block (lock held by another thread), this
    spins with non-blocking attempts, calling
    ``scheduler.wait_for_turn()`` between each attempt.  This gives the
    lock-holding thread a chance to execute opcodes and release the lock.

    Registers edges in the global :class:`WaitForGraph` so that lock-
    ordering deadlocks are detected instantly via cycle detection.
    """

    def __init__(self) -> None:
        self._lock = real_lock()
        self._object_id = id(self)
        self._owner_thread_id: int | None = None  # frontrun thread_id, not OS tid

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        # Reentrancy guard: if we're already inside DPOR machinery (e.g.,
        # _sync_reporter or _process_opcode), GC-triggered __del__ chains
        # must not re-enter the scheduler.  Fall back to real blocking.
        if _in_dpor_machinery():
            result = self._lock.acquire(blocking=blocking, timeout=timeout if timeout >= 0 else -1)
            return result

        if not blocking or timeout == 0:
            result = self._lock.acquire(blocking=False)
            if result:
                self._set_owner_and_report("lock_acquire")
                # Mark this acquire as a trylock so the DPOR engine's
                # release backtracking explores the ordering where the
                # attempt lands while the lock is still held (and fails).
                self._report("lock_attempt_ok")
            else:
                # A failed trylock is observable behavior (the caller takes
                # a different branch); report it so DPOR can explore the
                # orderings a real C-level lock would admit.
                self._report("lock_attempt_fail")
            return result

        ctx = get_context()
        if ctx is None:
            if self._lock.acquire(blocking=False):
                self._set_owner_and_report("lock_acquire")
                return True
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._set_owner_and_report("lock_acquire")
            return result

        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)
        if before_sync_retry is not None:
            assert after_sync_retry is not None
            if before_sync_retry(thread_id):
                acquired = self._lock.acquire(blocking=False)
                if acquired:
                    self._set_owner_and_report("lock_acquire")
                after_sync_retry(thread_id)
                if acquired:
                    return True
        elif self._lock.acquire(blocking=False):
            self._set_owner_and_report("lock_acquire")
            return True

        # Register waiting edge in the wait-for graph; raises SchedulerAbort on
        # cycle.  Skipped (graph is None) for timed acquires — see
        # _timed_acquire_state.
        deadline, graph, clock = _timed_acquire_state(timeout, scheduler, thread_id)
        if graph is not None:
            _check_lock_cycle(graph, thread_id, self._object_id, scheduler)

        note_spin = _spin_note_hook(scheduler) if timeout < 0 else None
        spin_flagged = False
        try:
            while True:
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    return False
                if before_sync_retry is not None:
                    assert after_sync_retry is not None
                    if not before_sync_retry(thread_id):
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._set_owner_and_report("lock_acquire")
                        return result
                    acquired = self._lock.acquire(blocking=False)
                    if acquired:
                        break
                    self._report("lock_wait")
                    after_sync_retry(thread_id)
                else:
                    self._report("lock_wait")
                    if scheduler._finished or scheduler._error:
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
                        if not scheduler._error and _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                            return False
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._set_owner_and_report("lock_acquire")
                        return result
                    if note_spin is not None:
                        # Flag the untimed spin for the virtual-clock autojump,
                        # then re-probe: an acquire here closes the race where
                        # the lock was released just before we flagged.
                        note_spin(thread_id, self._object_id, True)
                        spin_flagged = True
                        if self._lock.acquire(blocking=False):
                            break
                    scheduler.wait_for_turn(thread_id)
                    if self._lock.acquire(blocking=False):
                        break
        except BaseException:
            if graph is not None:
                graph.remove_waiting(thread_id, self._object_id)
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            raise
        finally:
            if spin_flagged and note_spin is not None:
                note_spin(thread_id, self._object_id, False)

        # Acquired — update graph: remove waiting edge, add holding edge
        if graph is not None:
            graph.remove_waiting(thread_id, self._object_id)
        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
        _record_holding(thread_id, self._object_id)

        self._owner_thread_id = thread_id
        self._report("lock_acquire")
        if after_sync_retry is not None:
            after_sync_retry(thread_id)
        return True

    def release(self) -> None:
        from frontrun._deadlock import get_wait_for_graph

        # Reentrancy guard: skip scheduler interaction during GC __del__
        if _in_dpor_machinery():
            owner = self._owner_thread_id
            self._owner_thread_id = None
            self._lock.release()
            if owner is not None:
                graph = get_wait_for_graph()
                if graph is not None:
                    graph.remove_holding(owner, self._object_id)
            return

        owner = self._owner_thread_id
        self._owner_thread_id = None
        self._lock.release()
        self._report("lock_release")
        _note_spin_release(self._object_id)

        # Remove holding edge
        if owner is not None:
            graph = get_wait_for_graph()
            if graph is not None:
                graph.remove_holding(owner, self._object_id)

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> "CooperativeLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _set_owner_and_report(self, event: str) -> None:
        """Set owner from TLS context and report the event."""
        from frontrun._deadlock import get_wait_for_graph

        ctx = get_context()
        if ctx is not None:
            _, thread_id = ctx
            self._owner_thread_id = thread_id
            graph = get_wait_for_graph()
            if graph is not None:
                graph.add_holding(thread_id, self._object_id)
        self._report(event)

    def _report(self, event: str) -> None:
        if is_sync_suppressed():
            return
        reporter = get_sync_reporter()
        if reporter is not None:
            prev = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            try:
                reporter(event, self._object_id, self)
            finally:
                _scheduler_tls._in_dpor_machinery = prev

    def __repr__(self) -> str:
        return f"<CooperativeLock locked={self.locked()}>"


# ---------------------------------------------------------------------------
# Cooperative RLock
# ---------------------------------------------------------------------------


class CooperativeRLock:
    """A reentrant lock that yields scheduler turns instead of blocking.

    Tracks the owning thread and recursion count.  The same thread can
    acquire multiple times without blocking; other threads spin-yield.

    Like :class:`CooperativeLock`, registers edges in the global
    :class:`WaitForGraph` for instant deadlock cycle detection.
    """

    def __init__(self) -> None:
        self._lock = real_lock()
        self._owner: int | None = None
        self._count = 0
        self._object_id = id(self)
        self._owner_thread_id: int | None = None  # frontrun thread_id
        self._acquired_during_dpor_machinery = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True

        # Reentrancy guard: if we're already inside DPOR machinery (e.g.,
        # GC-triggered __del__ during _process_opcode or _sync_reporter),
        # fall back to real blocking to avoid re-entering the scheduler.
        if _in_dpor_machinery():
            result = self._lock.acquire(blocking=blocking, timeout=timeout if timeout >= 0 else -1)
            if result:
                self._owner = me
                self._count = 1
                self._owner_thread_id = None
                self._acquired_during_dpor_machinery = True
            return result

        if not blocking or timeout == 0:
            if self._lock.acquire(blocking=False):
                self._owner = me
                self._count = 1
                self._set_owner_and_report("lock_acquire")
                self._report("lock_attempt_ok")
                return True
            self._report("lock_attempt_fail")
            return False

        # Fast path
        if self._lock.acquire(blocking=False):
            self._owner = me
            self._count = 1
            self._set_owner_and_report("lock_acquire")
            return True

        # Slow path: spin-yield
        ctx = get_context()
        if ctx is None:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._owner = me
                self._count = 1
                self._set_owner_and_report("lock_acquire")
            return result

        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)

        # Register waiting edge in the wait-for graph; raises SchedulerAbort on
        # cycle.  Skipped (graph is None) for timed acquires — see
        # _timed_acquire_state.
        deadline, graph, clock = _timed_acquire_state(timeout, scheduler, thread_id)
        if graph is not None:
            _check_lock_cycle(graph, thread_id, self._object_id, scheduler)

        note_spin = _spin_note_hook(scheduler) if timeout < 0 else None
        spin_flagged = False
        try:
            while True:
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    return False
                if before_sync_retry is not None:
                    assert after_sync_retry is not None
                    if not before_sync_retry(thread_id):
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._owner = me
                            self._count = 1
                            self._set_owner_and_report("lock_acquire")
                        return result
                    acquired = self._lock.acquire(blocking=False)
                    if acquired:
                        break
                    self._report("lock_wait")
                    after_sync_retry(thread_id)
                else:
                    self._report("lock_wait")
                    if scheduler._finished or scheduler._error:
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
                        if not scheduler._error and _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                            return False
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._owner = me
                            self._count = 1
                            self._set_owner_and_report("lock_acquire")
                        return result
                    if note_spin is not None:
                        # See CooperativeLock.acquire: flag the untimed spin
                        # for the virtual-clock autojump, then re-probe.
                        note_spin(thread_id, self._object_id, True)
                        spin_flagged = True
                        if self._lock.acquire(blocking=False):
                            break
                    scheduler.wait_for_turn(thread_id)
                    if self._lock.acquire(blocking=False):
                        break
        except BaseException:
            if graph is not None:
                graph.remove_waiting(thread_id, self._object_id)
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            raise
        finally:
            if spin_flagged and note_spin is not None:
                note_spin(thread_id, self._object_id, False)

        # Acquired — update graph
        if graph is not None:
            graph.remove_waiting(thread_id, self._object_id)
        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
        _record_holding(thread_id, self._object_id)

        self._owner = me
        self._owner_thread_id = thread_id
        self._count = 1
        self._report("lock_acquire")
        if after_sync_retry is not None:
            after_sync_retry(thread_id)
        return True

    def release(self) -> None:
        from frontrun._deadlock import get_wait_for_graph

        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            owner_tid = self._owner_thread_id
            acquired_during_dpor_machinery = self._acquired_during_dpor_machinery
            self._owner = None
            self._owner_thread_id = None
            self._acquired_during_dpor_machinery = False
            # Reentrancy guard: skip scheduler interaction during GC __del__
            # (same guard as CooperativeLock.release — see defect #7 / #11).
            if _in_dpor_machinery():
                self._lock.release()
                # Still scrub the holding edge for a normally-acquired lock
                # (owner_tid set): a lock released here but left in the
                # wait-for graph can fabricate a deadlock cycle through a lock
                # nobody actually holds.  Mirrors CooperativeLock.release().
                if owner_tid is not None:
                    graph = get_wait_for_graph()
                    if graph is not None:
                        graph.remove_holding(owner_tid, self._object_id)
                return
            self._lock.release()
            if acquired_during_dpor_machinery:
                return
            self._report("lock_release")
            _note_spin_release(self._object_id)

            if owner_tid is not None:
                graph = get_wait_for_graph()
                if graph is not None:
                    graph.remove_holding(owner_tid, self._object_id)

    def __enter__(self) -> "CooperativeRLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return self._owner == threading.get_ident()

    def _set_owner_and_report(self, event: str) -> None:
        """Set frontrun thread_id owner from TLS and report."""
        from frontrun._deadlock import get_wait_for_graph

        ctx = get_context()
        if ctx is not None:
            _, thread_id = ctx
            self._owner_thread_id = thread_id
            graph = get_wait_for_graph()
            if graph is not None:
                graph.add_holding(thread_id, self._object_id)
        self._report(event)

    def _report(self, event: str) -> None:
        if is_sync_suppressed():
            return
        reporter = get_sync_reporter()
        if reporter is not None:
            prev = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            try:
                reporter(event, self._object_id, self)
            finally:
                _scheduler_tls._in_dpor_machinery = prev

    def __repr__(self) -> str:
        return f"<CooperativeRLock owner={self._owner} count={self._count}>"


# ---------------------------------------------------------------------------
# Cooperative Semaphore
# ---------------------------------------------------------------------------


class CooperativeSemaphore:
    """A Semaphore that yields scheduler turns instead of blocking.

    Implemented with a real lock and counter rather than delegating to
    ``threading.Semaphore``, because the real Semaphore's ``__init__``
    references Condition/Lock from ``threading``'s globals which may be
    patched.

    Reports ``lock_acquire``/``lock_release`` sync events to the DPOR
    engine so that it establishes happens-before edges between release
    and subsequent acquire, preventing false-positive race reports on
    Semaphore-protected critical sections.
    """

    _value: int
    _lock: Any

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._lock = real_lock()
        self._object_id = id(self)

    def _report(self, event: str) -> None:
        if is_sync_suppressed():
            return
        reporter = get_sync_reporter()
        if reporter is not None:
            prev = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            try:
                reporter(event, self._object_id, self)
            finally:
                _scheduler_tls._in_dpor_machinery = prev

    def _try_acquire(self) -> bool:
        """Attempt a non-blocking decrement; return True on success."""
        self._lock.acquire()
        try:
            if self._value > 0:
                self._value -= 1
                return True
            return False
        finally:
            self._lock.release()

    def _drain_until(self, deadline: float) -> bool:
        """Spin on ``_try_acquire`` until *deadline*; report on success.

        Must use the *real* sleep: this runs after the scheduler finished, when
        the patched ``time.sleep`` is an instant no-op for managed threads —
        the 1 s drain window would otherwise busy-spin a full CPU core.
        """
        while _real_monotonic() < deadline:
            if self._try_acquire():
                self._report("lock_acquire")
                return True
            _real_time_sleep(0.001)
        return False

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        # Fast path: try to decrement counter
        is_trylock = (not blocking) or timeout == 0
        if self._try_acquire():
            self._report("lock_acquire")
            if is_trylock:
                # Mark as a trylock so DPOR release backtracking explores
                # the ordering where the attempt fails (see CooperativeLock).
                self._report("lock_attempt_ok")
            return True

        if is_trylock:
            # A failed trylock is observable behavior; make it visible to
            # the DPOR engine (see CooperativeLock.acquire).
            self._report("lock_attempt_fail")
            return False

        ctx = get_context()
        if ctx is None:
            # Unmanaged thread: spin with real sleep.  A single loop handles
            # both timeout and no-timeout via a sentinel (math.inf) deadline.
            deadline = _real_monotonic() + timeout if timeout is not None else math.inf
            while True:
                if self._try_acquire():
                    self._report("lock_acquire")
                    return True
                if _real_monotonic() >= deadline:
                    return False
                # Real sleep: the patched sleep is a no-op / clock-advance for
                # the driver thread under clock_scope, which would make this
                # poll a hot spin that also inflates the virtual clock.
                _real_time_sleep(0.001)

        # Spin-yield loop for managed threads.  A timeout is honoured through
        # _timed_acquire_state below: under a virtual clock it registers a
        # schedulable deadline with the scheduler; otherwise the deadline is
        # checked against the wall clock on every probe.  Shutdown drains via
        # ``_finished`` below.
        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)
        deadline, _, clock = (
            _timed_acquire_state(timeout, scheduler, thread_id) if timeout is not None else (None, None, None)
        )
        if before_sync_retry is not None:
            assert after_sync_retry is not None
            try:
                while True:
                    if _timed_acquire_expired(deadline, clock):
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                        return False
                    if scheduler._error:
                        raise SchedulerAbort("scheduler aborted")
                    if scheduler._finished:
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        return self._drain_until(_real_monotonic() + 1.0)
                    if not before_sync_retry(thread_id):
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        return False
                    if self._try_acquire():
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        self._report("lock_acquire")
                        after_sync_retry(thread_id)
                        return True
                    self._report("lock_wait")
                    after_sync_retry(thread_id)
            except BaseException:
                _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                raise

        # Bytecode scheduling relies on re-probing after each scheduler
        # turn; reporting wait without retrying can wedge a waiter forever.
        self._report("lock_wait")
        note_spin = _spin_note_hook(scheduler) if timeout is None else None
        try:
            while True:
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    return False
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if self._try_acquire():
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    self._report("lock_acquire")
                    return True
                if scheduler._finished:
                    if _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                        return False
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    return self._drain_until(_real_monotonic() + 1.0)
                if note_spin is not None:
                    # See CooperativeLock.acquire: flag the spin for the
                    # virtual-clock autojump, then re-probe.
                    note_spin(thread_id, self._object_id, True)
                    if self._try_acquire():
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        self._report("lock_acquire")
                        return True
                scheduler.wait_for_turn(thread_id)
        finally:
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            if note_spin is not None:
                note_spin(thread_id, self._object_id, False)

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        self._value += n
        self._lock.release()
        self._report("lock_release")
        _note_spin_release(self._object_id)

    def __enter__(self) -> "CooperativeSemaphore":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"<CooperativeSemaphore value={self._value}>"


# ---------------------------------------------------------------------------
# Cooperative BoundedSemaphore
# ---------------------------------------------------------------------------


class CooperativeBoundedSemaphore(CooperativeSemaphore):
    """A BoundedSemaphore that yields scheduler turns instead of blocking.

    Like ``CooperativeSemaphore`` but raises ``ValueError`` on
    over-release.
    """

    _initial_value: int

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self._initial_value = value

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        if self._value + n > self._initial_value:
            self._lock.release()
            raise ValueError("Semaphore released too many times")
        self._value += n
        self._lock.release()
        self._report("lock_release")
        _note_spin_release(self._object_id)

    def __repr__(self) -> str:
        return f"<CooperativeBoundedSemaphore value={self._value}/{self._initial_value}>"


# ---------------------------------------------------------------------------
# Cooperative Event
# ---------------------------------------------------------------------------


class CooperativeEvent:
    """An Event that yields scheduler turns instead of blocking on wait().

    Under a DPOR scheduler, an untimed ``wait()`` on an unset event reports
    ``event_wait`` — blocking the thread in the engine until ``set()`` reports
    ``event_set`` — instead of spin-probing.  A spinning waiter looks like
    useful work to the DPOR engine, so exploration branches that schedule the
    waiter before the setter would run it unboundedly until the deadlock
    timeout kills the branch.  Blocking also gives the engine the
    set() → wait()-return happens-before edge, which the spin never reported.
    """

    def __init__(self) -> None:
        self._event = make_real_event()
        self._object_id = id(self)
        # Waiters currently engine-blocked: thread_id → scheduler.  Lets a
        # set() from an *unmanaged* thread (no sync reporter in TLS) unblock
        # them via scheduler.clear_engine_block.
        self._engine_blocked: dict[int, Any] = {}

    def _report(self, event: str) -> bool:
        """Report to the per-thread sync reporter; True if one was called."""
        if _in_dpor_machinery():
            return False
        if is_sync_suppressed():
            return False
        reporter = get_sync_reporter()
        if reporter is None:
            return False
        prev = getattr(_scheduler_tls, "_in_dpor_machinery", False)
        _scheduler_tls._in_dpor_machinery = True
        try:
            reporter(event, self._object_id, self)
        finally:
            _scheduler_tls._in_dpor_machinery = prev
        return True

    @staticmethod
    def _yield_after_state_access() -> None:
        ctx = get_context()
        if ctx is None:
            return
        scheduler, thread_id = ctx
        wait_for_turn = getattr(scheduler, "wait_for_turn", None)
        if wait_for_turn is not None:
            wait_for_turn(thread_id)

    def wait(self, timeout: float | None = None) -> bool:

        if self._event.is_set():
            if self._report("event_read"):
                self._yield_after_state_access()
            return True

        # timeout == 0 is a pure probe (matches threading.Event.wait(0)): the
        # event is not set, so return False now rather than registering a
        # zero-length virtual deadline (a pointless schedulable clock step).
        if timeout == 0:
            return False

        ctx = get_context()
        if ctx is None:
            return self._event.wait(timeout=timeout)

        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)
        if timeout is None and before_sync_retry is not None and get_sync_reporter() is not None:
            # DPOR path: engine-block until set() (see class docstring).
            # Registration must precede the probe so an unmanaged set()
            # racing this window sees us in _engine_blocked (its
            # clear_engine_block is serialized with our block by the
            # engine lock inside the reporter).
            assert after_sync_retry is not None
            self._engine_blocked[thread_id] = scheduler
            try:
                while True:
                    if scheduler._error:
                        raise SchedulerAbort("scheduler aborted")
                    if scheduler._finished:
                        return self._event.wait(timeout=1.0)
                    if not before_sync_retry(thread_id):
                        return self._event.wait(timeout=1.0)
                    if self._event.is_set():
                        # Close the set() → wake happens-before edge.
                        self._report("event_wake")
                        after_sync_retry(thread_id)
                        return True
                    self._report("event_wait")
                    after_sync_retry(thread_id)
            finally:
                self._engine_blocked.pop(thread_id, None)

        # Bytecode/marker schedulers, and timed waits: spin-yield, probing
        # after each turn.  Under a virtual-clock scheduler, timed waits
        # register a virtual deadline so timeout branches are schedulable.
        deadline, clock = _timed_wait_deadline(timeout, scheduler, thread_id)
        note_spin = _spin_hook_for_wait(scheduler, timeout, clock)
        try:
            while not self._event.is_set():
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if scheduler._finished:
                    if _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                        return self._event.is_set()
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    return self._event.wait(timeout=1.0)
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    return False
                if note_spin is not None:
                    # Flag the spin for the virtual-clock autojump, then
                    # re-probe: a set() just before the flag must win.
                    note_spin(thread_id, self._object_id, True)
                    if self._event.is_set():
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        return True
                scheduler.wait_for_turn(thread_id)
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            return True
        finally:
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            if note_spin is not None:
                note_spin(thread_id, self._object_id, False)

    def set(self) -> None:
        if self._report("event_write"):
            self._yield_after_state_access()
        self._event.set()
        if not self._report("event_set"):
            # Unmanaged setter (no sync reporter): engine-blocked waiters must
            # still be unblocked, or they would sit until the deadlock timeout.
            for tid, sched in list(self._engine_blocked.items()):
                clear = getattr(sched, "clear_engine_block", None)
                if clear is not None:
                    clear(tid)
        # Random-strategy spinners waiting on this event must re-probe before
        # counting as hopeless for a virtual-clock autojump.
        _note_spin_release(self._object_id)

    def clear(self) -> None:
        if self._report("event_write"):
            self._yield_after_state_access()
        self._event.clear()

    def is_set(self) -> bool:
        result = self._event.is_set()
        if self._report("event_read"):
            self._yield_after_state_access()
        return result

    def __repr__(self) -> str:
        return f"<CooperativeEvent set={self.is_set()}>"


# ---------------------------------------------------------------------------
# Cooperative Condition
# ---------------------------------------------------------------------------


class CooperativeCondition:
    """A Condition that yields scheduler turns instead of blocking on wait().

    Uses a ticket-based notification system instead of polling a real
    Condition.  Each ``wait()`` takes a sequential ticket; ``notify(n)``
    advances the served counter by exactly ``n``, waking only ``n``
    waiters.  ``notify_all()`` advances by the number of current waiters.

    This avoids both the lost-notification bug (notifications before any
    thread is in ``wait()``) and the broadcast-instead-of-signal bug
    (``notify(1)`` waking all waiters that share the same snapshot).
    """

    def __init__(self, lock: "CooperativeLock | CooperativeRLock | None" = None) -> None:
        if lock is None:
            lock = CooperativeLock()
        self._lock: CooperativeLock | CooperativeRLock = lock
        # Ticket-based notification system.
        # _next_ticket: next ticket to assign to a waiter (incremented in wait())
        # _served: how many tickets have been served (incremented in notify())
        # A waiter with ticket T wakes when T < _served.
        # Both are modified while holding self._lock, so updates are serialized.
        self._next_ticket = 0
        self._served = 0
        self._waiters = 0
        # Tickets abandoned by waiters that left wait() without being served
        # (timeout or SchedulerAbort) while still un-served.  These must NOT
        # absorb a future notify(): a real threading.Condition removes timed-out
        # waiters from the wait queue.  Only ever holds un-served ticket ids
        # (ticket >= _served); once _served advances past a cancelled ticket it
        # is discarded.  All mutations happen while holding self._lock.
        self._cancelled: set[int] = set()
        # Tickets held by unmanaged waiters (no scheduler context) that block
        # in self._real_cond.wait() instead of spinning on the ticket system.
        # notify()/notify_all() wake real_cond only for these tickets, so a
        # notification meant for a managed (ticket-spinning) waiter does not
        # also spuriously wake an unmanaged waiter (over-waking beyond n).
        # All mutations happen while holding self._lock.
        self._real_cond_tickets: set[int] = set()
        # Legacy counter kept for notify_all() and backward compat with
        # any code that reads _notify_count.
        self._notify_count = 0
        self._object_id = id(self)
        # Fallback real condition for non-managed threads (no scheduler)
        self._real_cond = real_condition(real_lock())

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self._lock.acquire(*args, **kwargs)  # type: ignore[arg-type]

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "CooperativeCondition":
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()

    def _ticket_served(self, ticket: int) -> bool:
        """Whether *ticket* has been served (woken) by a notify call.

        A ticket is served when ``_served`` has advanced past it.  Cancelled
        tickets are never "served" (no live waiter owns them).
        """
        return ticket < self._served and ticket not in self._cancelled

    def _cancel_ticket(self, ticket: int) -> None:
        """Mark an un-served ticket as abandoned (caller holds self._lock).

        Recorded so that ``notify``/``notify_all`` skip it instead of letting
        it absorb a notification meant for a live waiter.  If the ticket has
        already been served, this is a no-op (handled by the caller).
        """
        if ticket >= self._served:
            self._cancelled.add(ticket)

    def _advance_served(self, n: int) -> tuple[int, int]:
        """Advance ``_served`` to wake up to *n* live (non-cancelled) tickets.

        Cancelled tickets in the path are skipped for free (they do not
        consume the notify budget).  Returns ``(woken, real_woken)`` where
        ``woken`` is the number of live tickets woken and ``real_woken`` is
        how many of those belong to unmanaged waiters blocked in
        ``real_cond.wait()`` (so the caller can wake exactly that many via
        ``real_cond``).  Caller must hold self._lock.
        """
        woken = 0
        real_woken = 0
        while woken < n and self._served < self._next_ticket:
            ticket = self._served
            self._served += 1
            if ticket in self._cancelled:
                self._cancelled.discard(ticket)
                continue
            woken += 1
            if ticket in self._real_cond_tickets:
                real_woken += 1
        self._notify_count += woken
        return woken, real_woken

    def _release_save(self) -> int:
        """Fully release the underlying lock, returning saved recursion depth.

        Mirrors ``threading.Condition._release_save`` so a reentrant lock held
        multiple times is fully released while waiting (otherwise a notifier on
        another thread could never acquire it).
        """
        lock = self._lock
        if isinstance(lock, CooperativeRLock):
            count = lock._count
            for _ in range(count):
                lock.release()
            return count
        lock.release()
        return 1

    def _acquire_restore(self, saved: int) -> None:
        """Re-acquire the lock to the recursion depth saved by _release_save."""
        lock = self._lock
        lock.acquire()
        if isinstance(lock, CooperativeRLock):
            # acquire() set count to 1; restore the remaining recursion levels.
            for _ in range(saved - 1):
                lock.acquire()

    def wait(self, timeout: float | None = None) -> bool:

        # _waiters and ticket assignment are written while we hold
        # self._lock (the caller must hold it per the Condition API).
        self._waiters += 1
        # Take a ticket BEFORE releasing the lock.  This waiter wakes
        # when my_ticket < self._served (i.e., enough notify() calls
        # have been made to reach this ticket).
        my_ticket = self._next_ticket
        self._next_ticket += 1
        # Determine the wakeup mechanism while we still hold the lock so the
        # ticket's population (managed spin vs. real_cond) is recorded
        # atomically with its assignment.
        ctx = get_context()
        if ctx is None:
            self._real_cond_tickets.add(my_ticket)
        # Fully release a reentrant lock (all recursion levels) so a notifier
        # on another thread can acquire it.  Releasing only one level would
        # leave count >= 1 and guarantee a stall (finding 9c).
        saved = self._release_save()
        served = False
        try:
            if ctx is None:
                # Not in a managed thread — fall back to real condition
                with self._real_cond:
                    served = self._real_cond.wait(timeout=timeout)
                    return served

            scheduler, thread_id = ctx
            if timeout == 0:
                # Pure probe (matches threading.Condition.wait(0)): a freshly
                # taken ticket can only be served if a notify already passed it,
                # so return now without registering a zero-length virtual
                # deadline.  The lock is already released (``_release_save``
                # above); the outer ``finally`` re-acquires it and reconciles the
                # ticket exactly as the spin path would on immediate expiry.
                served = my_ticket < self._served
                return served
            deadline, clock = _timed_wait_deadline(timeout, scheduler, thread_id)
            note_spin = _spin_hook_for_wait(scheduler, timeout, clock)

            # The spin-loop reads of _served below are intentionally
            # done WITHOUT holding self._lock.  This is safe because
            # _served is monotonically increasing: a stale read can
            # only cause one extra spin iteration, never a missed wakeup.

            try:
                while my_ticket >= self._served:
                    if scheduler._error:
                        raise SchedulerAbort("scheduler aborted")
                    if scheduler._finished:
                        if _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                            return my_ticket < self._served
                        # When the scheduler is done, give notifications a
                        # brief window to land (bounded by the remaining
                        # user timeout, if any).  Matches the previous
                        # per-branch behaviour: timeout case slept once for
                        # min(0.01, remaining); no-timeout case polled for up
                        # to 1s.  We unify with a bounded poll loop.
                        now = clock.now() if clock is not None else _real_monotonic()
                        end = _real_monotonic() + min(1.0, max(0.0, deadline - now))
                        # Real sleep, not the patched one: post-_finished the
                        # patched sleep is an instant no-op for managed
                        # threads, which would turn this poll into a hot spin.
                        while my_ticket >= self._served and _real_monotonic() < end:
                            _real_time_sleep(0.001)
                        served = my_ticket < self._served
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        return served
                    if _timed_acquire_expired(deadline, clock):
                        served = False
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                        return False
                    if note_spin is not None:
                        note_spin(thread_id, self._object_id, True)
                        if my_ticket < self._served:
                            break
                    scheduler.wait_for_turn(thread_id)
            finally:
                _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                if note_spin is not None:
                    note_spin(thread_id, self._object_id, False)
            served = True
            return True
        finally:
            # Re-acquire fully (serialises with notify/_waiters bookkeeping).
            self._acquire_restore(saved)
            self._waiters -= 1
            if ctx is None:
                self._real_cond_tickets.discard(my_ticket)
            # Reconcile the ticket: if we are leaving WITHOUT having observed a
            # wakeup, either the ticket is still un-served (cancel it so it does
            # not absorb a future notify), or it was served between our timeout
            # decision and re-acquiring the lock (the notification is wasted —
            # pass it on to another live waiter).
            if not served:
                if my_ticket >= self._served:
                    self._cancel_ticket(my_ticket)
                else:
                    # Already served but we are abandoning the wakeup; hand it
                    # to the next live waiter so no notification is lost.
                    _, real_woken = self._advance_served(1)
                    if real_woken:
                        with self._real_cond:
                            self._real_cond.notify(real_woken)

    def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        result = predicate()
        if result or timeout == 0:
            return result
        if timeout is not None:
            deadline = time.monotonic() + timeout
            while not result:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.wait(timeout=remaining)
                result = predicate()
            return result
        while not result:
            self.wait()
            result = predicate()
        return result

    def _check_owned(self) -> None:
        """Raise RuntimeError if the caller does not hold the underlying lock.

        The standard ``threading.Condition`` contract requires the caller
        to hold the associated lock when calling ``notify()`` or
        ``notify_all()``.  This method enforces that invariant.
        """
        lock = self._lock
        if isinstance(lock, CooperativeRLock):
            if not lock._is_owned():
                raise RuntimeError("cannot notify on un-acquired lock")
        elif isinstance(lock, CooperativeLock):  # type: ignore[unnecessary-isinstance]
            # CooperativeLock — check owner via TLS thread id
            ctx = get_context()
            if ctx is not None:
                _, thread_id = ctx
                if lock._owner_thread_id != thread_id:
                    raise RuntimeError("cannot notify on un-acquired lock")
            elif not lock.locked():
                raise RuntimeError("cannot notify on un-acquired lock")

    def notify(self, n: int = 1) -> None:
        # Enforce the Condition invariant: caller must hold self._lock.
        self._check_owned()
        # The caller holds self._lock, so this increment is serialised
        # with other notify/notify_all calls and with the _waiters/ticket
        # bookkeeping in wait().
        #
        # Advance _served to wake up to n LIVE (non-cancelled) waiters.
        # Cancelled tickets (abandoned by timed-out/aborted waiters) are
        # skipped for free so they do not absorb a notification — matching a
        # real threading.Condition which removes timed-out waiters from the
        # wait queue.  _advance_served returns the number actually woken and,
        # separately, how many of those were unmanaged (real_cond) waiters.
        _, real_woken = self._advance_served(max(0, n))
        _note_spin_release(self._object_id)
        # Wake the real condition ONLY for the unmanaged waiters whose tickets
        # were just served (no scheduler context — they block in
        # _real_cond.wait()).  Waking by the total served count would also wake
        # unmanaged waiters whose tickets were NOT served, over-waking past n.
        if real_woken:
            with self._real_cond:
                self._real_cond.notify(real_woken)

    def notify_all(self) -> None:
        # Enforce the Condition invariant: caller must hold self._lock.
        self._check_owned()
        # Wake every live waiter; cancelled tickets are skipped for free.
        self._advance_served(self._next_ticket - self._served)
        _note_spin_release(self._object_id)
        with self._real_cond:
            self._real_cond.notify_all()


# ---------------------------------------------------------------------------
# Cooperative Queue / LifoQueue / PriorityQueue
# ---------------------------------------------------------------------------


class CooperativeQueue:
    """A Queue that yields scheduler turns instead of blocking on get()/put()."""

    _queue_class = real_queue
    _queue_factory = staticmethod(make_real_queue)
    _queue: Any

    @classmethod
    def __class_getitem__(cls, item: Any) -> type:
        """Support generic subscript syntax (e.g. Queue[T]) for compatibility with psycopg v3."""
        return cls

    def __init__(self, maxsize: int = 0) -> None:
        self._queue = self._queue_factory(maxsize)
        self._object_id = id(self)
        self._engine_blocked_joiners: dict[int, Any] = {}

    def is_set(self) -> bool:
        """Internal event-style probe used by DPOR queue-join reporting."""
        with self._queue.all_tasks_done:
            return self._queue.unfinished_tasks == 0

    def _report_join(self, event: str) -> bool:
        if _in_dpor_machinery() or is_sync_suppressed():
            return False
        reporter = get_sync_reporter()
        if reporter is None:
            return False
        previous = getattr(_scheduler_tls, "_in_dpor_machinery", False)
        _scheduler_tls._in_dpor_machinery = True
        try:
            reporter(event, self._object_id, self)
        finally:
            _scheduler_tls._in_dpor_machinery = previous
        return True

    def get(self, block: bool = True, timeout: float | None = None) -> Any:

        try:
            item = self._queue.get(block=False)
            _note_spin_release(self._object_id)
            return item
        except queue.Empty:
            # timeout == 0 is a pure probe (matches queue.Queue.get(timeout=0)):
            # the queue is empty, so raise now rather than registering a
            # zero-length virtual deadline (a pointless schedulable clock step).
            if not block or timeout == 0:
                raise

        ctx = get_context()
        if ctx is None:
            return self._queue.get(block=True, timeout=timeout)

        scheduler, thread_id = ctx
        deadline, clock = _timed_wait_deadline(timeout, scheduler, thread_id)
        note_spin = _spin_hook_for_wait(scheduler, timeout, clock)
        try:
            while True:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                try:
                    item = self._queue.get(block=False)
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    _note_spin_release(self._object_id)
                    return item
                except queue.Empty:
                    pass
                if scheduler._finished:
                    if _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                        raise queue.Empty
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    return self._queue.get(block=True, timeout=1.0)
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    raise queue.Empty
                if note_spin is not None:
                    note_spin(thread_id, self._object_id, True)
                    try:
                        item = self._queue.get(block=False)
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        _note_spin_release(self._object_id)
                        return item
                    except queue.Empty:
                        pass
                scheduler.wait_for_turn(thread_id)
        finally:
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            if note_spin is not None:
                note_spin(thread_id, self._object_id, False)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:

        try:
            self._queue.put(item, block=False)
            _note_spin_release(self._object_id)
            return
        except queue.Full:
            # timeout == 0 is a pure probe (matches queue.Queue.put(timeout=0)):
            # the queue is full, so raise now rather than registering a
            # zero-length virtual deadline (a pointless schedulable clock step).
            if not block or timeout == 0:
                raise

        ctx = get_context()
        if ctx is None:
            self._queue.put(item, block=True, timeout=timeout)
            return

        scheduler, thread_id = ctx
        deadline, clock = _timed_wait_deadline(timeout, scheduler, thread_id)
        note_spin = _spin_hook_for_wait(scheduler, timeout, clock)
        try:
            while True:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                try:
                    self._queue.put(item, block=False)
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    _note_spin_release(self._object_id)
                    return
                except queue.Full:
                    pass
                if scheduler._finished:
                    if _finish_virtual_timed_wait(scheduler, thread_id, deadline, clock):
                        raise queue.Full
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                    self._queue.put(item, block=True, timeout=1.0)
                    _note_spin_release(self._object_id)
                    return
                if _timed_acquire_expired(deadline, clock):
                    _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=True)
                    raise queue.Full
                if note_spin is not None:
                    note_spin(thread_id, self._object_id, True)
                    try:
                        self._queue.put(item, block=False)
                        _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
                        _note_spin_release(self._object_id)
                        return
                    except queue.Full:
                        pass
                scheduler.wait_for_turn(thread_id)
        finally:
            _timed_acquire_cleanup(scheduler, thread_id, clock, gave_up=False)
            if note_spin is not None:
                note_spin(thread_id, self._object_id, False)

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def get_nowait(self) -> Any:
        item = self._queue.get(block=False)
        _note_spin_release(self._object_id)
        return item

    def put_nowait(self, item: Any) -> None:
        self._queue.put(item, block=False)
        _note_spin_release(self._object_id)

    def task_done(self) -> None:
        self._queue.task_done()
        if self.is_set():
            if not self._report_join("event_set"):
                for thread_id, scheduler in list(self._engine_blocked_joiners.items()):
                    clear = getattr(scheduler, "clear_engine_block", None)
                    if clear is not None:
                        clear(thread_id)
        _note_spin_release(self._object_id)

    def join(self) -> None:
        ctx = get_context()
        if ctx is None:
            self._queue.join()
            return

        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)
        if before_sync_retry is not None and get_sync_reporter() is not None:
            assert after_sync_retry is not None
            self._engine_blocked_joiners[thread_id] = scheduler
            try:
                while True:
                    if scheduler._error:
                        raise SchedulerAbort("scheduler aborted")
                    if scheduler._finished or not before_sync_retry(thread_id):
                        self._queue.join()
                        return
                    if self.is_set():
                        self._report_join("event_wake")
                        after_sync_retry(thread_id)
                        return
                    self._report_join("event_wait")
                    after_sync_retry(thread_id)
            finally:
                self._engine_blocked_joiners.pop(thread_id, None)

        # Random/marker schedulers have no engine-blocking sync protocol; keep
        # their existing cooperative spin-yield behavior.
        note_spin = _spin_hook_for_wait(scheduler, None, None)
        try:
            while True:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                with self._queue.all_tasks_done:
                    if self._queue.unfinished_tasks == 0:
                        return
                if scheduler._finished:
                    # Scheduling is over, so let any already-running consumer
                    # finish through the queue's native condition protocol.
                    self._queue.join()
                    return
                if note_spin is not None:
                    note_spin(thread_id, self._object_id, True)
                    with self._queue.all_tasks_done:
                        if self._queue.unfinished_tasks == 0:
                            return
                scheduler.wait_for_turn(thread_id)
        finally:
            if note_spin is not None:
                note_spin(thread_id, self._object_id, False)


class CooperativeLifoQueue(CooperativeQueue):
    """LifoQueue variant of the cooperative queue."""

    _queue_class = real_lifo_queue
    _queue_factory = staticmethod(make_real_lifo_queue)


class CooperativePriorityQueue(CooperativeQueue):
    """PriorityQueue variant of the cooperative queue."""

    _queue_class = real_priority_queue
    _queue_factory = staticmethod(make_real_priority_queue)


# ---------------------------------------------------------------------------
# Monkey-patching helpers
# ---------------------------------------------------------------------------

_patched = False
# Reference count protects against concurrent patch/unpatch from
# parallel test runners (e.g. pytest-xdist in-process parallelism).
_patch_count = 0
_patch_count_lock = real_lock()


def is_patched() -> bool:
    """Return True if cooperative lock patching is currently active."""
    return _patched


def patch_locks() -> None:
    """Replace threading and queue primitives with cooperative versions.

    Safe to call from multiple concurrent test runners: uses reference
    counting so the first call patches and subsequent calls increment
    the count.  ``unpatch_locks()`` only restores the originals when
    the count drops to zero.
    """
    # Pre-import modules that grab threading.Lock at module level so their
    # internal lock objects are created with the real C-level lock before we
    # monkey-patch threading.Lock with a cooperative version.
    import concurrent.futures.thread  # noqa: F401

    global _patched, _patch_count  # noqa: PLW0603
    with _patch_count_lock:
        _patch_count += 1
        if _patch_count > 1:
            return  # Already patched by another runner
        threading.Lock = CooperativeLock  # type: ignore[assignment]
        threading.RLock = CooperativeRLock  # type: ignore[assignment]
        threading.Semaphore = CooperativeSemaphore  # type: ignore[assignment]
        threading.BoundedSemaphore = CooperativeBoundedSemaphore  # type: ignore[assignment]
        threading.Event = CooperativeEvent  # type: ignore[assignment]
        threading.Condition = CooperativeCondition  # type: ignore[assignment]
        queue.Queue = CooperativeQueue  # type: ignore[assignment]
        queue.LifoQueue = CooperativeLifoQueue  # type: ignore[assignment]
        queue.PriorityQueue = CooperativePriorityQueue  # type: ignore[assignment]
        _patched = True


def unpatch_locks() -> None:
    """Restore the original threading and queue primitives.

    Only actually restores when all paired ``patch_locks()`` calls
    have been balanced by ``unpatch_locks()`` calls.
    """
    global _patched, _patch_count  # noqa: PLW0603
    with _patch_count_lock:
        if _patch_count <= 0:
            return  # Not patched — nothing to do
        _patch_count -= 1
        if _patch_count > 0:
            return  # Still in use by another runner
        threading.Lock = real_lock  # type: ignore[assignment]
        threading.RLock = real_rlock  # type: ignore[assignment]
        threading.Semaphore = real_semaphore  # type: ignore[assignment]
        threading.BoundedSemaphore = real_bounded_semaphore  # type: ignore[assignment]
        threading.Event = real_event  # type: ignore[assignment]
        threading.Condition = real_condition  # type: ignore[assignment]
        queue.Queue = real_queue  # type: ignore[assignment]
        queue.LifoQueue = real_lifo_queue  # type: ignore[assignment]
        queue.PriorityQueue = real_priority_queue  # type: ignore[assignment]
        _patched = False


# ---------------------------------------------------------------------------
# Sleep patching
# ---------------------------------------------------------------------------

_real_time_sleep = time.sleep
_saved_time_sleep = _real_time_sleep
_sleep_patch_count = 0
_sleep_patch_lock = real_lock()
_sleep_patch_owners: dict[int, int] = {}


def _cooperative_sleep(seconds: float) -> None:
    """Replacement for ``time.sleep`` during exploration.

    Without a virtual clock this is a no-op scheduling point: if a
    cooperative scheduler context is active, yields a turn so other threads
    can run.  The actual delay is skipped entirely — sleeping during
    interleaving exploration would make execution extremely slow.

    When the scheduler owns a :class:`~frontrun._virtual_clock.VirtualClock`,
    a positive sleep becomes a *timed block*: the thread registers a virtual
    deadline and blocks until the scheduler advances the clock to it.
    ``time.sleep(0)`` stays a pure yield, matching real Python semantics.
    """
    # Preserve time.sleep's input contract before any virtual/no-delay branch.
    # Calling the pristine function only for invalid values gives us CPython's
    # platform-specific exception types/messages without ever blocking here.
    if seconds < 0 or not math.isfinite(seconds):
        return _real_time_sleep(seconds)

    ctx = get_context()
    if ctx is None:
        # No scheduler turn (e.g. setup()/invariant under clock_scope): if a
        # virtual clock is active for this thread/context, age it by the sleep
        # so TTL-aging setup code isn't frozen.  The driver thread is the only
        # clock user at that moment, so advancing here stays deterministic.
        # An unrelated caller has no virtual clock: retain the behavior that
        # surrounded the outermost frontrun patch scope.
        clock = _active_virtual_clock()
        if clock is None:
            # A direct low-level patch_sleep() owner retains the historical
            # cooperative/no-delay contract. The process-wide shim can also be
            # observed by unrelated background threads; those must keep the
            # surrounding real/third-party sleep behavior.
            with _sleep_patch_lock:
                owned_here = _sleep_patch_owners.get(threading.get_ident(), 0) > 0
                sleep = _saved_time_sleep if _sleep_patch_count > 0 else _real_time_sleep
            if not owned_here:
                sleep(seconds)
        elif seconds > 0:
            clock.advance_to(clock.now() + seconds)
        return
    scheduler, thread_id = ctx
    clock = getattr(scheduler, "virtual_clock", None)
    sleep_until = getattr(scheduler, "sleep_until", None)
    if clock is not None and sleep_until is not None and seconds > 0:
        sleep_until(thread_id, clock.now() + seconds)
        return
    scheduler.wait_for_turn(thread_id)


def patch_sleep() -> None:
    """Replace ``time.sleep`` with the cooperative scheduler hook.

    Reference-counted like :func:`patch_locks` so multiple concurrent
    callers are safe.
    """
    global _saved_time_sleep, _sleep_patch_count  # noqa: PLW0603
    with _sleep_patch_lock:
        _sleep_patch_count += 1
        owner = threading.get_ident()
        _sleep_patch_owners[owner] = _sleep_patch_owners.get(owner, 0) + 1
        if _sleep_patch_count > 1:
            return
        _saved_time_sleep = time.sleep
        time.sleep = _cooperative_sleep  # type: ignore[assignment]


def unpatch_sleep() -> None:
    """Restore the original ``time.sleep``.

    Only restores when all paired ``patch_sleep()`` calls have been
    balanced.
    """
    global _sleep_patch_count  # noqa: PLW0603
    with _sleep_patch_lock:
        if _sleep_patch_count <= 0:
            return
        _sleep_patch_count -= 1
        owner = threading.get_ident()
        owner_count = _sleep_patch_owners.get(owner, 0)
        if owner_count <= 1:
            _sleep_patch_owners.pop(owner, None)
        else:
            _sleep_patch_owners[owner] = owner_count - 1
        if _sleep_patch_count > 0:
            return
        time.sleep = _saved_time_sleep  # type: ignore[assignment]
