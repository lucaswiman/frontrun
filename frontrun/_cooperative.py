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
        from frontrun._deadlock import get_wait_for_graph

        # Reentrancy guard: if we're already inside DPOR machinery (e.g.,
        # _sync_reporter or _process_opcode), GC-triggered __del__ chains
        # must not re-enter the scheduler.  Fall back to real blocking.
        if _in_dpor_machinery():
            result = self._lock.acquire(blocking=blocking, timeout=timeout if timeout >= 0 else -1)
            return result

        if not blocking:
            result = self._lock.acquire(blocking=False)
            if result:
                self._set_owner_and_report("lock_acquire")
            return result

        # Fast path: uncontested
        if self._lock.acquire(blocking=False):
            self._set_owner_and_report("lock_acquire")
            return True

        # Slow path: lock is held – get scheduler context from TLS.
        ctx = get_context()
        if ctx is None:
            # Not in a managed thread — fall back to real blocking
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._set_owner_and_report("lock_acquire")
            return result

        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)

        # Register waiting edge in the wait-for graph; raises SchedulerAbort on cycle
        graph = get_wait_for_graph()
        if graph is not None:
            _check_lock_cycle(graph, thread_id, self._object_id, scheduler)

        try:
            while True:
                if before_sync_retry is not None:
                    assert after_sync_retry is not None
                    if not before_sync_retry(thread_id):
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
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
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._set_owner_and_report("lock_acquire")
                        return result
                    scheduler.wait_for_turn(thread_id)
                    if self._lock.acquire(blocking=False):
                        break
        except BaseException:
            if graph is not None:
                graph.remove_waiting(thread_id, self._object_id)
            raise

        # Acquired — update graph: remove waiting edge, add holding edge
        if graph is not None:
            graph.remove_waiting(thread_id, self._object_id)
            graph.add_holding(thread_id, self._object_id)

        self._owner_thread_id = thread_id
        self._report("lock_acquire")
        if after_sync_retry is not None:
            after_sync_retry(thread_id)
        return True

    def release(self) -> None:
        from frontrun._deadlock import get_wait_for_graph

        # Reentrancy guard: skip scheduler interaction during GC __del__
        if _in_dpor_machinery():
            self._lock.release()
            return

        owner = self._owner_thread_id
        self._owner_thread_id = None
        self._lock.release()
        self._report("lock_release")

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
        from frontrun._deadlock import get_wait_for_graph

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

        if not blocking:
            if self._lock.acquire(blocking=False):
                self._owner = me
                self._count = 1
                self._set_owner_and_report("lock_acquire")
                return True
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

        # Register waiting edge in the wait-for graph; raises SchedulerAbort on cycle
        graph = get_wait_for_graph()
        if graph is not None:
            _check_lock_cycle(graph, thread_id, self._object_id, scheduler)

        try:
            while True:
                if before_sync_retry is not None:
                    assert after_sync_retry is not None
                    if not before_sync_retry(thread_id):
                        if graph is not None:
                            graph.remove_waiting(thread_id, self._object_id)
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
                        result = self._lock.acquire(blocking=blocking, timeout=1.0)
                        if result:
                            self._owner = me
                            self._count = 1
                            self._set_owner_and_report("lock_acquire")
                        return result
                    scheduler.wait_for_turn(thread_id)
                    if self._lock.acquire(blocking=False):
                        break
        except BaseException:
            if graph is not None:
                graph.remove_waiting(thread_id, self._object_id)
            raise

        # Acquired — update graph
        if graph is not None:
            graph.remove_waiting(thread_id, self._object_id)
            graph.add_holding(thread_id, self._object_id)

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
                return
            self._lock.release()
            if acquired_during_dpor_machinery:
                return
            self._report("lock_release")

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
        """Spin on ``_try_acquire`` until *deadline*; report on success."""
        while time.monotonic() < deadline:
            if self._try_acquire():
                self._report("lock_acquire")
                return True
            time.sleep(0.001)
        return False

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        # Fast path: try to decrement counter
        if self._try_acquire():
            self._report("lock_acquire")
            return True

        if not blocking:
            return False

        ctx = get_context()
        if ctx is None:
            # Unmanaged thread: spin with real sleep.  A single loop handles
            # both timeout and no-timeout via a sentinel (math.inf) deadline.
            deadline = time.monotonic() + timeout if timeout is not None else math.inf
            while True:
                if self._try_acquire():
                    self._report("lock_acquire")
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.001)

        # Spin-yield loop for managed threads.  Timeout is intentionally
        # ignored here: DPOR/bytecode schedulers drive progress themselves,
        # and shutdown drains via ``_finished`` below.
        scheduler, thread_id = ctx
        before_sync_retry = getattr(scheduler, "before_sync_retry", None)
        after_sync_retry = getattr(scheduler, "after_sync_retry", None)
        if before_sync_retry is not None:
            assert after_sync_retry is not None
            while True:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if scheduler._finished:
                    return self._drain_until(time.monotonic() + 1.0)
                if not before_sync_retry(thread_id):
                    return False
                if self._try_acquire():
                    self._report("lock_acquire")
                    after_sync_retry(thread_id)
                    return True
                self._report("lock_wait")
                after_sync_retry(thread_id)

        # Bytecode scheduling relies on re-probing after each scheduler
        # turn; reporting wait without retrying can wedge a waiter forever.
        self._report("lock_wait")
        while True:
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            if self._try_acquire():
                self._report("lock_acquire")
                return True
            if scheduler._finished:
                return self._drain_until(time.monotonic() + 1.0)
            scheduler.wait_for_turn(thread_id)

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        self._value += n
        self._lock.release()
        self._report("lock_release")

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

    def __repr__(self) -> str:
        return f"<CooperativeBoundedSemaphore value={self._value}/{self._initial_value}>"


# ---------------------------------------------------------------------------
# Cooperative Event
# ---------------------------------------------------------------------------


class CooperativeEvent:
    """An Event that yields scheduler turns instead of blocking on wait()."""

    def __init__(self) -> None:
        self._event = make_real_event()

    def wait(self, timeout: float | None = None) -> bool:

        if self._event.is_set():
            return True

        ctx = get_context()
        if ctx is None:
            return self._event.wait(timeout=timeout)

        scheduler, thread_id = ctx
        deadline = time.monotonic() + timeout if timeout is not None else math.inf
        while not self._event.is_set():
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            if scheduler._finished:
                return self._event.wait(timeout=1.0)
            if time.monotonic() >= deadline:
                return False
            scheduler.wait_for_turn(thread_id)
        return True

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

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
        # Legacy counter kept for notify_all() and backward compat with
        # any code that reads _notify_count.
        self._notify_count = 0
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

    def wait(self, timeout: float | None = None) -> bool:

        # _waiters and ticket assignment are written while we hold
        # self._lock (the caller must hold it per the Condition API).
        self._waiters += 1
        # Take a ticket BEFORE releasing the lock.  This waiter wakes
        # when my_ticket < self._served (i.e., enough notify() calls
        # have been made to reach this ticket).
        my_ticket = self._next_ticket
        self._next_ticket += 1
        self._lock.release()
        try:
            ctx = get_context()
            if ctx is None:
                # Not in a managed thread — fall back to real condition
                with self._real_cond:
                    return self._real_cond.wait(timeout=timeout)

            scheduler, thread_id = ctx

            # The spin-loop reads of _served below are intentionally
            # done WITHOUT holding self._lock.  This is safe because
            # _served is monotonically increasing: a stale read can
            # only cause one extra spin iteration, never a missed wakeup.

            deadline = time.monotonic() + timeout if timeout is not None else math.inf
            while my_ticket >= self._served:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if scheduler._finished:
                    # When the scheduler is done, give notifications a
                    # brief window to land (bounded by the remaining
                    # user timeout, if any).  Matches the previous
                    # per-branch behaviour: timeout case slept once for
                    # min(0.01, remaining); no-timeout case polled for up
                    # to 1s.  We unify with a bounded poll loop.
                    end = time.monotonic() + min(1.0, max(0.0, deadline - time.monotonic()))
                    while my_ticket >= self._served and time.monotonic() < end:
                        time.sleep(0.001)
                    return my_ticket < self._served
                if time.monotonic() >= deadline:
                    return False
                scheduler.wait_for_turn(thread_id)
            return True
        finally:
            self._lock.acquire()
            # Decrement AFTER re-acquiring the lock so that the write is
            # serialised with notify_all()'s read of _waiters.
            self._waiters -= 1

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
        # Advance _served by exactly n (or fewer if there aren't enough
        # un-served tickets).  Only waiters whose ticket < _served will wake.
        # Use _next_ticket - _served (un-served tickets) instead of _waiters,
        # because _waiters includes already-notified waiters that haven't yet
        # re-acquired the lock to decrement _waiters.
        unserved = self._next_ticket - self._served
        actual = min(max(0, n), unserved)
        self._served += actual
        self._notify_count += actual
        # Also wake the real condition for threads in the non-cooperative
        # path (no scheduler context — they block in _real_cond.wait()).
        with self._real_cond:
            self._real_cond.notify(actual)

    def notify_all(self) -> None:
        # Enforce the Condition invariant: caller must hold self._lock.
        self._check_owned()
        unserved = self._next_ticket - self._served
        if unserved > 0:
            self._served += unserved
            self._notify_count += unserved
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

    def get(self, block: bool = True, timeout: float | None = None) -> Any:

        try:
            return self._queue.get(block=False)
        except queue.Empty:
            if not block:
                raise

        ctx = get_context()
        if ctx is None:
            return self._queue.get(block=True, timeout=timeout)

        scheduler, thread_id = ctx
        deadline = time.monotonic() + timeout if timeout is not None else math.inf
        while True:
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            try:
                return self._queue.get(block=False)
            except queue.Empty:
                pass
            if scheduler._finished:
                return self._queue.get(block=True, timeout=1.0)
            if time.monotonic() >= deadline:
                raise queue.Empty
            scheduler.wait_for_turn(thread_id)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:

        try:
            self._queue.put(item, block=False)
            return
        except queue.Full:
            if not block:
                raise

        ctx = get_context()
        if ctx is None:
            self._queue.put(item, block=True, timeout=timeout)
            return

        scheduler, thread_id = ctx
        deadline = time.monotonic() + timeout if timeout is not None else math.inf
        while True:
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            try:
                self._queue.put(item, block=False)
                return
            except queue.Full:
                pass
            if scheduler._finished:
                self._queue.put(item, block=True, timeout=1.0)
                return
            if time.monotonic() >= deadline:
                raise queue.Full
            scheduler.wait_for_turn(thread_id)

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def get_nowait(self) -> Any:
        return self._queue.get(block=False)

    def put_nowait(self, item: Any) -> None:
        self._queue.put(item, block=False)

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()


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
_sleep_patch_count = 0
_sleep_patch_lock = real_lock()


def _cooperative_sleep(seconds: float) -> None:  # noqa: ARG001
    """No-op replacement for ``time.sleep`` during exploration.

    Acts as a scheduling point: if a cooperative scheduler context is
    active, yields a turn so other threads can run.  The actual delay
    is skipped entirely — sleeping during interleaving exploration would
    make execution extremely slow.
    """
    ctx = get_context()
    if ctx is not None:
        scheduler, thread_id = ctx
        scheduler.wait_for_turn(thread_id)


def patch_sleep() -> None:
    """Replace ``time.sleep`` with a no-op cooperative version.

    Reference-counted like :func:`patch_locks` so multiple concurrent
    callers are safe.
    """
    global _sleep_patch_count  # noqa: PLW0603
    with _sleep_patch_lock:
        _sleep_patch_count += 1
        if _sleep_patch_count > 1:
            return
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
        if _sleep_patch_count > 0:
            return
        time.sleep = _real_time_sleep  # type: ignore[assignment]
