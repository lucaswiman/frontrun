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
"""

import queue
import threading
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Save real factories before any patching happens.
# ---------------------------------------------------------------------------
# threading.Lock is a factory function (_thread.allocate_lock), not a type.

real_lock = threading.Lock
real_rlock = threading.RLock
real_semaphore = threading.Semaphore
real_bounded_semaphore = threading.BoundedSemaphore
real_event = threading.Event
real_condition = threading.Condition
real_queue = queue.Queue
real_lifo_queue = queue.LifoQueue
real_priority_queue = queue.PriorityQueue

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
    """

    def __init__(self) -> None:
        self._lock = real_lock()
        self._object_id = id(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking:
            result = self._lock.acquire(blocking=False)
            if result:
                self._report("lock_acquire")
            return result

        # Fast path: uncontested
        if self._lock.acquire(blocking=False):
            self._report("lock_acquire")
            return True

        # Slow path: lock is held – get scheduler context from TLS.
        ctx = get_context()
        if ctx is None:
            # Not in a managed thread — fall back to real blocking
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._report("lock_acquire")
            return result

        scheduler, thread_id = ctx
        # Spin: yield scheduler turns until we can acquire
        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                result = self._lock.acquire(blocking=blocking, timeout=1.0)
                if result:
                    self._report("lock_acquire")
                return result
            scheduler.wait_for_turn(thread_id)

        self._report("lock_acquire")
        return True

    def release(self) -> None:
        self._report("lock_release")
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> "CooperativeLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

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
    """

    def __init__(self) -> None:
        self._lock = real_lock()
        self._owner: int | None = None
        self._count = 0
        self._object_id = id(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True

        if not blocking:
            if self._lock.acquire(blocking=False):
                self._owner = me
                self._count = 1
                self._report("lock_acquire")
                return True
            return False

        # Fast path
        if self._lock.acquire(blocking=False):
            self._owner = me
            self._count = 1
            self._report("lock_acquire")
            return True

        # Slow path: spin-yield
        ctx = get_context()
        if ctx is None:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._owner = me
                self._count = 1
                self._report("lock_acquire")
            return result

        scheduler, thread_id = ctx
        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                result = self._lock.acquire(blocking=blocking, timeout=1.0)
                if result:
                    self._owner = me
                    self._count = 1
                    self._report("lock_acquire")
                return result
            scheduler.wait_for_turn(thread_id)

        self._owner = me
        self._count = 1
        self._report("lock_acquire")
        return True

    def release(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._report("lock_release")
            self._lock.release()

    def __enter__(self) -> "CooperativeRLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return self._owner == threading.get_ident()

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
            self._lock.acquire()
            if self._value > 0:
                self._value -= 1
                self._lock.release()
                return True
            self._lock.release()
            if scheduler._finished or scheduler._error:
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
        if self._event.is_set():
            return True

        ctx = get_context()
        if ctx is None:
            return self._event.wait(timeout=timeout)

        scheduler, thread_id = ctx

        if timeout is not None:
            deadline: float = time.monotonic() + timeout
            while not self._event.is_set():
                if scheduler._finished or scheduler._error:
                    return self._event.wait(timeout=1.0)
                if time.monotonic() >= deadline:
                    return self._event.is_set()
                scheduler.wait_for_turn(thread_id)
            return True

        while not self._event.is_set():
            if scheduler._finished or scheduler._error:
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

    Uses the scheduler's TLS to avoid infinite recursion — the
    ``OpcodeScheduler`` itself uses ``real_condition`` internally.
    """

    def __init__(self, lock: CooperativeLock | None = None) -> None:
        if lock is None:
            lock = CooperativeLock()
        self._lock = lock
        self._real_cond = real_condition(real_lock())
        self._waiters = 0

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
        # Release the user lock, spin-yield, then re-acquire
        self._waiters += 1
        self._lock.release()
        try:
            ctx = get_context()
            if ctx is None:
                with self._real_cond:
                    return self._real_cond.wait(timeout=timeout)

            scheduler, thread_id = ctx

            if timeout is not None:
                deadline = time.monotonic() + timeout
                with self._real_cond:
                    notified = self._real_cond.wait(timeout=0)
                while not notified:
                    if scheduler._finished or scheduler._error:
                        with self._real_cond:
                            return self._real_cond.wait(timeout=1.0)
                    if time.monotonic() >= deadline:
                        return False
                    scheduler.wait_for_turn(thread_id)
                    with self._real_cond:
                        notified = self._real_cond.wait(timeout=0)
                return True

            with self._real_cond:
                notified = self._real_cond.wait(timeout=0)
            while not notified:
                if scheduler._finished or scheduler._error:
                    with self._real_cond:
                        return self._real_cond.wait(timeout=1.0)
                scheduler.wait_for_turn(thread_id)
                with self._real_cond:
                    notified = self._real_cond.wait(timeout=0)
            return True
        finally:
            self._waiters -= 1
            self._lock.acquire()

    def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        result = predicate()
        while not result:
            self.wait(timeout=timeout)
            result = predicate()
            if timeout is not None:
                break
        return result

    def notify(self, n: int = 1) -> None:
        with self._real_cond:
            self._real_cond.notify(n)

    def notify_all(self) -> None:
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
                try:
                    return self._queue.get(block=False)
                except queue.Empty:
                    pass
                if scheduler._finished or scheduler._error:
                    return self._queue.get(block=True, timeout=1.0)
                if time.monotonic() >= deadline:
                    raise queue.Empty
                scheduler.wait_for_turn(thread_id)

        while True:
            try:
                return self._queue.get(block=False)
            except queue.Empty:
                pass
            if scheduler._finished or scheduler._error:
                return self._queue.get(block=True, timeout=1.0)
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

        if timeout is not None:
            deadline: float = time.monotonic() + timeout
            while True:
                try:
                    self._queue.put(item, block=False)
                    return
                except queue.Full:
                    pass
                if scheduler._finished or scheduler._error:
                    self._queue.put(item, block=True, timeout=1.0)
                    return
                if time.monotonic() >= deadline:
                    raise queue.Full
                scheduler.wait_for_turn(thread_id)

        while True:
            try:
                self._queue.put(item, block=False)
                return
            except queue.Full:
                pass
            if scheduler._finished or scheduler._error:
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


def patch_locks() -> None:
    """Replace threading and queue primitives with cooperative versions."""
    global _patched  # noqa: PLW0603
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
    """Restore the original threading and queue primitives."""
    global _patched  # noqa: PLW0603
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
