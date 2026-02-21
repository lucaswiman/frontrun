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
waiting edge creates a cycle, a ``TimeoutError`` with a diagnostic message
is raised immediately.  All spin loops also check ``scheduler._error``
eagerly (before each iteration) and bail via
:class:`~frontrun._deadlock.SchedulerAbort` when the scheduler has been
torn down.
"""

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from frontrun import _real_threading as _rt

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

SyncReporter = Callable[[str, int], None]  # (event_name, object_id) -> None


def get_sync_reporter() -> SyncReporter | None:
    """Return the per-thread sync reporter, or ``None``."""
    return getattr(_scheduler_tls, "sync_reporter", None)


def set_sync_reporter(reporter: SyncReporter | None) -> None:
    """Install a per-thread sync reporter (or clear with ``None``)."""
    _scheduler_tls.sync_reporter = reporter


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
        from frontrun._deadlock import SchedulerAbort, format_cycle, get_wait_for_graph

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

        # Register waiting edge in the wait-for graph
        graph = get_wait_for_graph()
        if graph is not None:
            cycle = graph.add_waiting(thread_id, self._object_id)
            if cycle is not None:
                graph.remove_waiting(thread_id, self._object_id)
                msg = f"Lock-ordering deadlock detected: {format_cycle(cycle)}"
                scheduler.report_error(TimeoutError(msg))
                raise SchedulerAbort(msg)

        # Tell the DPOR engine that this thread is waiting for the lock
        # so it can schedule the lock holder instead.
        self._report("lock_wait")

        try:
            # Spin: yield scheduler turns until we can acquire
            while not self._lock.acquire(blocking=False):
                if scheduler._finished or scheduler._error:
                    if graph is not None:
                        graph.remove_waiting(thread_id, self._object_id)
                    result = self._lock.acquire(blocking=blocking, timeout=1.0)
                    if result:
                        self._set_owner_and_report("lock_acquire")
                    return result
                scheduler.wait_for_turn(thread_id)
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
        return True

    def release(self) -> None:
        from frontrun._deadlock import get_wait_for_graph

        owner = self._owner_thread_id
        self._owner_thread_id = None
        self._report("lock_release")
        self._lock.release()

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
        reporter = get_sync_reporter()
        if reporter is not None:
            reporter(event, self._object_id)

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

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        from frontrun._deadlock import SchedulerAbort, format_cycle, get_wait_for_graph

        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True

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

        # Register waiting edge in the wait-for graph
        graph = get_wait_for_graph()
        if graph is not None:
            cycle = graph.add_waiting(thread_id, self._object_id)
            if cycle is not None:
                graph.remove_waiting(thread_id, self._object_id)
                msg = f"Lock-ordering deadlock detected: {format_cycle(cycle)}"
                scheduler.report_error(TimeoutError(msg))
                raise SchedulerAbort(msg)

        # Tell the DPOR engine that this thread is waiting for the lock
        # so it can schedule the lock holder instead.
        self._report("lock_wait")

        try:
            while not self._lock.acquire(blocking=False):
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
        return True

    def release(self) -> None:
        from frontrun._deadlock import get_wait_for_graph

        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            owner_tid = self._owner_thread_id
            self._owner = None
            self._owner_thread_id = None
            self._report("lock_release")
            self._lock.release()

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
        reporter = get_sync_reporter()
        if reporter is not None:
            reporter(event, self._object_id)

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
    """

    _value: int
    _lock: Any

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._lock = real_lock()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        from frontrun._deadlock import SchedulerAbort

        # Fast path: try to decrement counter
        self._lock.acquire()
        if self._value > 0:
            self._value -= 1
            self._lock.release()
            return True
        self._lock.release()

        if not blocking:
            return False

        ctx = get_context()
        if ctx is None:
            # Fall back to spinning with real lock (non-managed thread)
            if timeout is not None:
                deadline = time.monotonic() + timeout
                while True:
                    self._lock.acquire()
                    if self._value > 0:
                        self._value -= 1
                        self._lock.release()
                        return True
                    self._lock.release()
                    if time.monotonic() >= deadline:
                        return False
                    time.sleep(0.001)
            while True:
                self._lock.acquire()
                if self._value > 0:
                    self._value -= 1
                    self._lock.release()
                    return True
                self._lock.release()
                time.sleep(0.001)

        # Spin-yield loop for managed threads
        scheduler, thread_id = ctx
        while True:
            # Aggressive error check (option 6)
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            self._lock.acquire()
            if self._value > 0:
                self._value -= 1
                self._lock.release()
                return True
            self._lock.release()
            if scheduler._finished:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    self._lock.acquire()
                    if self._value > 0:
                        self._value -= 1
                        self._lock.release()
                        return True
                    self._lock.release()
                    time.sleep(0.001)
                return False
            scheduler.wait_for_turn(thread_id)

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        self._value += n
        self._lock.release()

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

    def __repr__(self) -> str:
        return f"<CooperativeBoundedSemaphore value={self._value}/{self._initial_value}>"


# ---------------------------------------------------------------------------
# Cooperative Event
# ---------------------------------------------------------------------------


class CooperativeEvent:
    """An Event that yields scheduler turns instead of blocking on wait()."""

    def __init__(self) -> None:
        self._event = real_event()

    def wait(self, timeout: float | None = None) -> bool:
        from frontrun._deadlock import SchedulerAbort

        if self._event.is_set():
            return True

        ctx = get_context()
        if ctx is None:
            return self._event.wait(timeout=timeout)

        scheduler, thread_id = ctx

        if timeout is not None:
            deadline: float = time.monotonic() + timeout
            while not self._event.is_set():
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if scheduler._finished:
                    return self._event.wait(timeout=1.0)
                if time.monotonic() >= deadline:
                    return self._event.is_set()
                scheduler.wait_for_turn(thread_id)
            return True

        while not self._event.is_set():
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            if scheduler._finished:
                return self._event.wait(timeout=1.0)
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

    Uses a simple notification counter instead of polling a real Condition.
    ``notify()`` increments the counter; ``wait()`` spin-yields until the
    counter advances past its snapshot.  This avoids the lost-notification
    bug that occurs when ``_real_cond.notify()`` fires while no thread is
    blocked in ``_real_cond.wait()``.
    """

    def __init__(self, lock: "CooperativeLock | CooperativeRLock | None" = None) -> None:
        if lock is None:
            lock = CooperativeLock()
        self._lock: CooperativeLock | CooperativeRLock = lock
        # Notification counter — monotonically increasing.  Each notify()
        # bumps it by n; each wait() records a snapshot and spins until
        # the counter exceeds the snapshot.
        self._notify_count = 0
        self._waiters = 0
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
        from frontrun._deadlock import SchedulerAbort

        # _waiters and notify_count_before_wait are written while we hold
        # self._lock (the caller must hold it per the Condition API).
        self._waiters += 1
        # Record the counter BEFORE releasing the lock so that any
        # notify() that fires after we release is visible.
        notify_count_before_wait = self._notify_count
        self._lock.release()
        try:
            ctx = get_context()
            if ctx is None:
                # Not in a managed thread — fall back to real condition
                with self._real_cond:
                    return self._real_cond.wait(timeout=timeout)

            scheduler, thread_id = ctx

            # The spin-loop reads of _notify_count below are intentionally
            # done WITHOUT holding self._lock.  This is safe because
            # _notify_count is monotonically increasing: a stale read can
            # only cause one extra spin iteration, never a missed wakeup.

            if timeout is not None:
                deadline = time.monotonic() + timeout
                while self._notify_count <= notify_count_before_wait:
                    if scheduler._error:
                        raise SchedulerAbort("scheduler aborted")
                    if scheduler._finished:
                        remaining = max(0.0, deadline - time.monotonic())
                        time.sleep(min(0.01, remaining))
                        return self._notify_count > notify_count_before_wait
                    if time.monotonic() >= deadline:
                        return False
                    scheduler.wait_for_turn(thread_id)
                return True

            while self._notify_count <= notify_count_before_wait:
                if scheduler._error:
                    raise SchedulerAbort("scheduler aborted")
                if scheduler._finished:
                    end = time.monotonic() + 1.0
                    while self._notify_count <= notify_count_before_wait and time.monotonic() < end:
                        time.sleep(0.001)
                    return self._notify_count > notify_count_before_wait
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
        # with other notify/notify_all calls and with the _waiters
        # bookkeeping in wait().
        self._notify_count += n
        # Also wake the real condition for threads in the non-cooperative
        # path (no scheduler context — they block in _real_cond.wait()).
        with self._real_cond:
            self._real_cond.notify(n)

    def notify_all(self) -> None:
        # Enforce the Condition invariant: caller must hold self._lock.
        self._check_owned()
        self._notify_count += max(self._waiters, 1)
        with self._real_cond:
            self._real_cond.notify_all()


# ---------------------------------------------------------------------------
# Cooperative Queue / LifoQueue / PriorityQueue
# ---------------------------------------------------------------------------


class CooperativeQueue:
    """A Queue that yields scheduler turns instead of blocking on get()/put()."""

    _queue_class = real_queue
    _queue: Any

    def __init__(self, maxsize: int = 0) -> None:
        self._queue = self._queue_class(maxsize)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        from frontrun._deadlock import SchedulerAbort

        try:
            return self._queue.get(block=False)
        except queue.Empty:
            if not block:
                raise

        ctx = get_context()
        if ctx is None:
            return self._queue.get(block=True, timeout=timeout)

        scheduler, thread_id = ctx

        if timeout is not None:
            deadline: float = time.monotonic() + timeout
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

        while True:
            if scheduler._error:
                raise SchedulerAbort("scheduler aborted")
            try:
                return self._queue.get(block=False)
            except queue.Empty:
                pass
            if scheduler._finished:
                return self._queue.get(block=True, timeout=1.0)
            scheduler.wait_for_turn(thread_id)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        from frontrun._deadlock import SchedulerAbort

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

        if timeout is not None:
            deadline: float = time.monotonic() + timeout
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


class CooperativePriorityQueue(CooperativeQueue):
    """PriorityQueue variant of the cooperative queue."""

    _queue_class = real_priority_queue


# ---------------------------------------------------------------------------
# Monkey-patching helpers
# ---------------------------------------------------------------------------

_patched = False
# Reference count protects against concurrent patch/unpatch from
# parallel test runners (e.g. pytest-xdist in-process parallelism).
_patch_count = 0
_patch_count_lock = real_lock()


def patch_locks() -> None:
    """Replace threading and queue primitives with cooperative versions.

    Safe to call from multiple concurrent test runners: uses reference
    counting so the first call patches and subsequent calls increment
    the count.  ``unpatch_locks()`` only restores the originals when
    the count drops to zero.
    """
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
        _patch_count -= 1
        if _patch_count > 0:
            return  # Still in use by another runner
        _patch_count = max(_patch_count, 0)  # Guard against over-unpatch
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
