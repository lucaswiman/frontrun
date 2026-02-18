"""
Bytecode-level deterministic concurrency testing.

Uses sys.settrace with f_trace_opcodes to intercept execution at every
bytecode instruction, enabling fine-grained control over thread interleaving.

This pairs naturally with property-based testing: rather than specifying exact
schedules, generate random interleavings and check that invariants hold (or
that bugs can be found).

The core insight: CPython context switches happen between bytecode instructions.
By controlling which thread gets to execute each instruction, we can explore
the full space of possible interleavings.

Example — find a race condition with random schedule exploration:

    >>> from frontrun.bytecode import explore_interleavings
    >>>
    >>> class Counter:
    ...     def __init__(self):
    ...         self.value = 0
    ...     def increment(self):
    ...         temp = self.value
    ...         self.value = temp + 1
    >>>
    >>> result = explore_interleavings(
    ...     setup=lambda: Counter(),
    ...     threads=[lambda c: c.increment(), lambda c: c.increment()],
    ...     invariant=lambda c: c.value == 2,
    ... )
    >>> assert not result.property_holds  # race condition found!
"""

import os
import queue
import random
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from frontrun.common import InterleavingResult

# Type variable for the shared state passed between setup and thread functions
T = TypeVar("T")

# Directories to never trace into (stdlib, site-packages, threading internals)
_SKIP_DIRS: set[str] = set()
for _p in sys.path:
    if "lib/python" in _p or "site-packages" in _p:
        _SKIP_DIRS.add(_p)
_THREADING_FILE = threading.__file__
_THIS_FILE = os.path.abspath(__file__)


def _should_trace_file(filename: str) -> bool:
    """Check whether a file is user code that should be traced."""
    if filename == _THREADING_FILE or filename == _THIS_FILE:
        return False
    if filename.startswith("<"):
        return False
    for skip_dir in _SKIP_DIRS:
        if filename.startswith(skip_dir):
            return False
    return True


class OpcodeScheduler:
    """Controls thread execution at bytecode instruction granularity.

    The schedule is a list of thread indices. Each entry means "let this
    thread execute one bytecode instruction." When the schedule is
    exhausted, all threads run freely to completion.
    """

    def __init__(self, schedule: list[int], num_threads: int):
        self.schedule = schedule
        self.num_threads = num_threads
        self._index = 0
        self._lock = _real_lock()
        self._condition = threading.Condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()

    def wait_for_turn(self, thread_id: int) -> bool:
        """Block until it's this thread's turn. Returns False when schedule exhausted."""
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False

                if self._index >= len(self.schedule):
                    self._finished = True
                    self._condition.notify_all()
                    return False

                scheduled_tid = self.schedule[self._index]

                if scheduled_tid in self._threads_done:
                    self._index += 1
                    self._condition.notify_all()
                    continue

                if scheduled_tid == thread_id:
                    self._index += 1
                    self._condition.notify_all()
                    return True

                if not self._condition.wait(timeout=5.0):
                    needed = self.schedule[self._index]
                    if needed in self._threads_done:
                        continue
                    self._error = TimeoutError(
                        f"Deadlock: schedule wants thread {needed} at index {self._index}/{len(self.schedule)}"
                    )
                    self._condition.notify_all()
                    return False

    def mark_done(self, thread_id: int):
        """Mark a thread as finished."""
        with self._condition:
            self._threads_done.add(thread_id)
            self._condition.notify_all()

    def report_error(self, error: Exception):
        """Report an error and unblock all threads."""
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    @property
    def had_error(self) -> bool:
        return self._error is not None


class _CooperativeLock:
    """A Lock replacement that yields scheduler turns instead of blocking.

    When acquire() would block (lock is held by another thread), this
    spins with non-blocking attempts, calling scheduler.wait_for_turn()
    between each attempt. This gives the lock-holding thread a chance
    to execute opcodes and release the lock, breaking the deadlock that
    occurs with real Lock.acquire() under opcode-level scheduling.
    """

    def __init__(self):
        self._lock = _real_lock()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking:
            return self._lock.acquire(blocking=False)

        # Fast path: lock is uncontested
        if self._lock.acquire(blocking=False):
            return True

        # Slow path: lock is held. Get our scheduler context from TLS.
        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            # Not in a managed thread — fall back to real blocking
            return self._lock.acquire(blocking=blocking, timeout=timeout)

        # Spin: yield scheduler turns until we can acquire
        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                return self._lock.acquire(blocking=blocking, timeout=1.0)
            scheduler.wait_for_turn(thread_id)

        return True

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __repr__(self):
        return f"<_CooperativeLock locked={self.locked()}>"


# Thread-local storage for the active scheduler context.
# Set by _run_thread so cooperative wrappers can find the scheduler.
_active_scheduler = threading.local()

# Save real factories before any patching.
# threading.Lock is a factory function (_thread.allocate_lock), not a type.
_real_lock = threading.Lock
_real_rlock = threading.RLock
_real_semaphore = threading.Semaphore
_real_bounded_semaphore = threading.BoundedSemaphore
_real_event = threading.Event
_real_condition = threading.Condition
_real_queue = queue.Queue
_real_lifo_queue = queue.LifoQueue
_real_priority_queue = queue.PriorityQueue


class _CooperativeRLock:
    """A reentrant lock that yields scheduler turns instead of blocking.

    Tracks the owning thread and recursion count. The same thread can
    acquire multiple times without blocking; other threads spin-yield.
    """

    def __init__(self):
        self._lock = _real_lock()
        self._owner = None
        self._count = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True

        if not blocking:
            if self._lock.acquire(blocking=False):
                self._owner = me
                self._count = 1
                return True
            return False

        # Fast path
        if self._lock.acquire(blocking=False):
            self._owner = me
            self._count = 1
            return True

        # Slow path: spin-yield
        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._owner = me
                self._count = 1
            return result

        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                result = self._lock.acquire(blocking=blocking, timeout=1.0)
                if result:
                    self._owner = me
                    self._count = 1
                return result
            scheduler.wait_for_turn(thread_id)

        self._owner = me
        self._count = 1
        return True

    def release(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _is_owned(self):
        return self._owner == threading.get_ident()

    def __repr__(self):
        owner = self._owner
        return f"<_CooperativeRLock owner={owner} count={self._count}>"


class _CooperativeSemaphore:
    """A Semaphore that yields scheduler turns instead of blocking.

    Implemented with a real lock and counter rather than delegating to
    threading.Semaphore, because the real Semaphore's __init__ references
    Condition/Lock from threading's globals which may be patched.
    """

    _value: int
    _lock: Any

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._lock = _real_lock()

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

        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            # Fall back to spinning with real lock (non-managed thread)
            if timeout is not None:
                import time

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
                import time

                time.sleep(0.001)

        # Spin-yield loop for managed threads
        while True:
            self._lock.acquire()
            if self._value > 0:
                self._value -= 1
                self._lock.release()
                return True
            self._lock.release()
            if scheduler._finished or scheduler._error:
                # Schedule exhausted — spin without scheduler to let
                # threads complete naturally (same pattern as _CooperativeLock)
                import time

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

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __repr__(self):
        return f"<_CooperativeSemaphore value={self._value}>"


class _CooperativeBoundedSemaphore(_CooperativeSemaphore):
    """A BoundedSemaphore that yields scheduler turns instead of blocking.

    Like _CooperativeSemaphore but raises ValueError on over-release.
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

    def __repr__(self):
        return f"<_CooperativeBoundedSemaphore value={self._value}/{self._initial_value}>"


class _CooperativeEvent:
    """An Event that yields scheduler turns instead of blocking on wait()."""

    def __init__(self):
        self._event = _real_event()

    def wait(self, timeout: float | None = None) -> bool:
        if self._event.is_set():
            return True

        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            return self._event.wait(timeout=timeout)

        if timeout is not None:
            import time

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

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def is_set(self):
        return self._event.is_set()

    def __repr__(self):
        return f"<_CooperativeEvent set={self.is_set()}>"


class _CooperativeCondition:
    """A Condition that yields scheduler turns instead of blocking on wait().

    Uses the scheduler's TLS to avoid infinite recursion — the OpcodeScheduler
    itself uses _real_condition internally.
    """

    def __init__(self, lock: _CooperativeLock | None = None) -> None:
        if lock is None:
            lock = _CooperativeLock()
        self._lock = lock
        self._real_cond = _real_condition(_real_lock())
        # Track waiters so notify can wake them
        self._waiters = 0

    def acquire(self, *args: Any, **kwargs: Any) -> bool:  # type: ignore[no-untyped-def]
        return self._lock.acquire(*args, **kwargs)  # type: ignore[arg-type]

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()

    def wait(self, timeout: float | None = None) -> bool:
        # Release the user lock, spin-yield, then re-acquire
        self._waiters += 1
        self._lock.release()
        try:
            scheduler = getattr(_active_scheduler, "scheduler", None)
            thread_id = getattr(_active_scheduler, "thread_id", None)
            if scheduler is None:
                with self._real_cond:
                    return self._real_cond.wait(timeout=timeout)

            if timeout is not None:
                import time

                deadline = time.monotonic() + timeout
                # Spin-yield checking if we've been notified
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


class _CooperativeQueue:
    """A Queue that yields scheduler turns instead of blocking on get()/put()."""

    _queue_class = _real_queue
    _queue: Any  # type: ignore[assignment]

    def __init__(self, maxsize: int = 0) -> None:
        self._queue = self._queue_class(maxsize)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        try:
            return self._queue.get(block=False)
        except queue.Empty:
            if not block:
                raise

        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            return self._queue.get(block=True, timeout=timeout)

        if timeout is not None:
            import time

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

        scheduler = getattr(_active_scheduler, "scheduler", None)
        thread_id = getattr(_active_scheduler, "thread_id", None)
        if scheduler is None:
            self._queue.put(item, block=True, timeout=timeout)
            return

        if timeout is not None:
            import time

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

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def get_nowait(self) -> Any:
        return self._queue.get(block=False)

    def put_nowait(self, item: Any) -> None:
        self._queue.put(item, block=False)

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()


class _CooperativeLifoQueue(_CooperativeQueue):
    """LifoQueue variant of the cooperative queue."""

    _queue_class = _real_lifo_queue


class _CooperativePriorityQueue(_CooperativeQueue):
    """PriorityQueue variant of the cooperative queue."""

    _queue_class = _real_priority_queue


class BytecodeShuffler:
    """Run concurrent functions with bytecode-level interleaving control.

    Sets up per-thread trace functions that intercept every bytecode
    instruction in user code and defer to the OpcodeScheduler.

    When cooperative_locks=True (default), replaces threading and queue
    primitives (Lock, RLock, Semaphore, BoundedSemaphore, Event,
    Condition, Queue, LifoQueue, PriorityQueue) with cooperative
    versions that yield scheduler turns instead of blocking in C. This
    prevents the deadlock that otherwise occurs when one thread holds a
    primitive and the scheduler gives a turn to another thread that
    tries to acquire it.
    """

    def __init__(self, scheduler: OpcodeScheduler, cooperative_locks: bool = True):
        self.scheduler = scheduler
        self.cooperative_locks = cooperative_locks
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False

    def _patch_locks(self):
        """Replace threading and queue primitives with cooperative versions."""
        if not self.cooperative_locks:
            return
        threading.Lock = _CooperativeLock
        threading.RLock = _CooperativeRLock
        threading.Semaphore = _CooperativeSemaphore
        threading.BoundedSemaphore = _CooperativeBoundedSemaphore
        threading.Event = _CooperativeEvent
        threading.Condition = _CooperativeCondition
        queue.Queue = _CooperativeQueue
        queue.LifoQueue = _CooperativeLifoQueue
        queue.PriorityQueue = _CooperativePriorityQueue
        self._lock_patched = True

    def _unpatch_locks(self):
        """Restore the original threading and queue primitives."""
        if self._lock_patched:
            threading.Lock = _real_lock
            threading.RLock = _real_rlock
            threading.Semaphore = _real_semaphore
            threading.BoundedSemaphore = _real_bounded_semaphore
            threading.Event = _real_event
            threading.Condition = _real_condition
            queue.Queue = _real_queue
            queue.LifoQueue = _real_lifo_queue
            queue.PriorityQueue = _real_priority_queue
            self._lock_patched = False

    def _make_trace(self, thread_id: int) -> Callable[[Any, str, Any], Any]:  # type: ignore[return-value]
        """Create a sys.settrace function for the given thread."""
        scheduler = self.scheduler

        def trace(frame: Any, event: str, arg: Any) -> Any:  # type: ignore[name-defined]
            if scheduler._finished or scheduler._error:
                return None

            if event == "call":
                if _should_trace_file(frame.f_code.co_filename):
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == "opcode":
                scheduler.wait_for_turn(thread_id)
                return trace

            return trace

        return trace

    def _run_thread(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Thread entry point. Installs tracing, runs func, cleans up."""
        try:
            # Store scheduler context in TLS for cooperative wrappers
            _active_scheduler.scheduler = self.scheduler
            _active_scheduler.thread_id = thread_id

            trace_fn = self._make_trace(thread_id)
            sys.settrace(trace_fn)
            func(*args, **kwargs)
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            sys.settrace(None)
            _active_scheduler.scheduler = None
            _active_scheduler.thread_id = None
            self.scheduler.mark_done(thread_id)

    def run(
        self,
        funcs: list[Callable[..., None]],
        args: list[tuple[Any, ...]] | None = None,
        kwargs: list[dict[str, Any]] | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Run functions concurrently with controlled interleaving.

        Args:
            funcs: One callable per thread.
            args: Per-thread positional args.
            kwargs: Per-thread keyword args.
            timeout: Max wait time per thread.
        """
        if args is None:
            args = [() for _ in funcs]
        if kwargs is None:
            kwargs = [{} for _ in funcs]

        for i, (func, a, kw) in enumerate(zip(funcs, args, kwargs)):
            t = threading.Thread(
                target=self._run_thread,
                args=(i, func, a, kw),
                name=f"frontrun-{i}",
                daemon=True,
            )
            self.threads.append(t)

        for t in self.threads:
            t.start()

        for t in self.threads:
            t.join(timeout=timeout)

        if self.errors:
            raise list(self.errors.values())[0]


@contextmanager
def controlled_interleaving(schedule: list[int], num_threads: int = 2):
    """Context manager for running code under a specific interleaving.

    Args:
        schedule: List of thread indices controlling opcode execution order.
        num_threads: Number of threads.

    Yields:
        BytecodeShuffler runner.

    Example:
        >>> with controlled_interleaving([0, 1, 0, 1], num_threads=2) as runner:
        ...     runner.run([func1, func2])
    """
    scheduler = OpcodeScheduler(schedule, num_threads)
    runner = BytecodeShuffler(scheduler)
    yield runner


# ---------------------------------------------------------------------------
# Property-based testing
# ---------------------------------------------------------------------------


def run_with_schedule(
    schedule: list[int],
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    timeout: float = 5.0,
    cooperative_locks: bool = True,
) -> T:
    """Run one interleaving and return the state object.

    Args:
        schedule: Opcode-level schedule (list of thread indices).
        setup: Returns fresh shared state.
        threads: Callables that each receive the state as their argument.
        timeout: Max seconds.
        cooperative_locks: Replace threading/queue primitives with
            scheduler-aware versions that prevent deadlocks (default True).

    Returns:
        The state object after execution.
    """
    scheduler = OpcodeScheduler(schedule, len(threads))
    runner = BytecodeShuffler(scheduler, cooperative_locks=cooperative_locks)

    # Patch locks BEFORE setup() so any locks created there are cooperative
    runner._patch_locks()
    try:
        state = setup()

        def make_thread_func(thread_func: Callable[[T], None], thread_state: T) -> Callable[[], None]:
            def thread_wrapper() -> None:
                thread_func(thread_state)

            return thread_wrapper

        funcs: list[Callable[[], None]] = [make_thread_func(t, state) for t in threads]
        try:
            runner.run(funcs, timeout=timeout)
        except TimeoutError:
            pass
    finally:
        runner._unpatch_locks()
    return state


def explore_interleavings(
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    max_attempts: int = 200,
    max_ops: int = 300,
    timeout_per_run: float = 5.0,
    seed: int | None = None,
) -> InterleavingResult:
    """Search for interleavings that violate an invariant.

    Generates random opcode-level schedules and tests whether the invariant
    holds under each one. If a violation is found, returns immediately with
    the counterexample schedule.

    This is the bytecode-level analogue of property-based testing: instead
    of generating random *inputs*, we generate random *interleavings* and
    check that the result satisfies an invariant.

    Args:
        setup: Returns fresh shared state for each attempt.
        threads: Callables that each receive the state as their argument.
        invariant: Predicate on the state. Returns True if the property holds.
        max_attempts: How many random interleavings to try.
        max_ops: Maximum schedule length per attempt.
        timeout_per_run: Timeout for each individual run.
        seed: Optional RNG seed for reproducibility.

    Returns:
        InterleavingResult with the outcome.
    """
    rng = random.Random(seed)
    num_threads = len(threads)
    result = InterleavingResult(property_holds=True, num_explored=0)

    for _ in range(max_attempts):
        length = rng.randint(1, max_ops)
        schedule = [rng.randint(0, num_threads - 1) for _ in range(length)]

        state = run_with_schedule(schedule, setup, threads, timeout=timeout_per_run)
        result.num_explored += 1

        if not invariant(state):
            result.property_holds = False
            result.counterexample = schedule
            return result

    return result


def schedule_strategy(num_threads: int, max_ops: int = 300):
    """Hypothesis strategy for generating opcode schedules.

    For use with hypothesis @given decorator in your own tests:

        >>> from hypothesis import given
        >>> from frontrun.bytecode import schedule_strategy, run_with_schedule
        >>>
        >>> @given(schedule=schedule_strategy(2))
        ... def test_my_invariant(schedule):
        ...     state = run_with_schedule(schedule, setup, threads)
        ...     assert state.value == expected

    Note: hypothesis expects deterministic tests. Bytecode-level interleaving
    is deterministic for a given schedule, but hypothesis's shrinking may
    still interact oddly with threading. Consider using
    settings(phases=[Phase.generate]) to skip shrinking if needed.
    """
    from hypothesis import strategies as st

    return st.lists(
        st.integers(min_value=0, max_value=num_threads - 1),
        min_size=1,
        max_size=max_ops,
    )
