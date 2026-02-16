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

    >>> from interlace.bytecode import explore_interleavings
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
import random
import sys
import threading
from contextlib import contextmanager
from typing import Callable, List, Optional, Dict, Any, Set

from interlace.common import InterleavingResult


# Directories to never trace into (stdlib, site-packages, threading internals)
_SKIP_DIRS: Set[str] = set()
for _p in sys.path:
    if 'lib/python' in _p or 'site-packages' in _p:
        _SKIP_DIRS.add(_p)
_THREADING_FILE = threading.__file__
_THIS_FILE = os.path.abspath(__file__)


def _should_trace_file(filename: str) -> bool:
    """Check whether a file is user code that should be traced."""
    if filename == _THREADING_FILE or filename == _THIS_FILE:
        return False
    if filename.startswith('<'):
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

    def __init__(self, schedule: List[int], num_threads: int):
        self.schedule = schedule
        self.num_threads = num_threads
        self._index = 0
        self._lock = _real_lock()
        self._condition = threading.Condition(self._lock)
        self._finished = False
        self._error: Optional[Exception] = None
        self._threads_done: Set[int] = set()

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
                        f"Deadlock: schedule wants thread {needed} "
                        f"at index {self._index}/{len(self.schedule)}"
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

    def acquire(self, blocking=True, timeout=-1):
        if not blocking:
            return self._lock.acquire(blocking=False)

        # Fast path: lock is uncontested
        if self._lock.acquire(blocking=False):
            return True

        # Slow path: lock is held. Get our scheduler context from TLS.
        scheduler = _active_scheduler.scheduler
        thread_id = _active_scheduler.thread_id
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

    def __exit__(self, *args):
        self.release()

    def __repr__(self):
        return f"<_CooperativeLock locked={self.locked()}>"


# Thread-local storage for the active scheduler context.
# Set by _run_thread so _CooperativeLock can find the scheduler.
_active_scheduler = threading.local()

# Save the real Lock factory before any patching.
# threading.Lock is a factory function (_thread.allocate_lock), not a type.
_real_lock = threading.Lock


class BytecodeInterlace:
    """Run concurrent functions with bytecode-level interleaving control.

    Sets up per-thread trace functions that intercept every bytecode
    instruction in user code and defer to the OpcodeScheduler.

    When cooperative_locks=True (default), replaces threading.Lock with
    a cooperative version that yields scheduler turns instead of blocking
    in C. This prevents the deadlock that otherwise occurs when one
    thread holds a lock and the scheduler gives a turn to another thread
    that tries to acquire it.
    """

    def __init__(self, scheduler: OpcodeScheduler, cooperative_locks: bool = True):
        self.scheduler = scheduler
        self.cooperative_locks = cooperative_locks
        self.threads: List[threading.Thread] = []
        self.errors: Dict[int, Exception] = {}
        self._lock_patched = False

    def _patch_locks(self):
        """Replace threading.Lock with _CooperativeLock."""
        if not self.cooperative_locks:
            return
        threading.Lock = _CooperativeLock
        self._lock_patched = True

    def _unpatch_locks(self):
        """Restore the original threading.Lock."""
        if self._lock_patched:
            threading.Lock = _real_lock
            self._lock_patched = False

    def _make_trace(self, thread_id: int):
        """Create a sys.settrace function for the given thread."""
        scheduler = self.scheduler

        def trace(frame, event, arg):
            if scheduler._finished or scheduler._error:
                return None

            if event == 'call':
                if _should_trace_file(frame.f_code.co_filename):
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == 'opcode':
                scheduler.wait_for_turn(thread_id)
                return trace

            return trace

        return trace

    def _run_thread(self, thread_id: int, func: Callable, args: tuple, kwargs: dict):
        """Thread entry point. Installs tracing, runs func, cleans up."""
        try:
            # Store scheduler context in TLS for _CooperativeLock
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
        funcs: List[Callable],
        args: Optional[List[tuple]] = None,
        kwargs: Optional[List[dict]] = None,
        timeout: float = 10.0,
    ):
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
                name=f"interlace-{i}",
            )
            self.threads.append(t)

        for t in self.threads:
            t.start()

        for t in self.threads:
            t.join(timeout=timeout)

        if self.errors:
            raise list(self.errors.values())[0]


@contextmanager
def controlled_interleaving(schedule: List[int], num_threads: int = 2):
    """Context manager for running code under a specific interleaving.

    Args:
        schedule: List of thread indices controlling opcode execution order.
        num_threads: Number of threads.

    Yields:
        BytecodeInterlace runner.

    Example:
        >>> with controlled_interleaving([0, 1, 0, 1], num_threads=2) as runner:
        ...     runner.run([func1, func2])
    """
    scheduler = OpcodeScheduler(schedule, num_threads)
    runner = BytecodeInterlace(scheduler)
    yield runner


# ---------------------------------------------------------------------------
# Property-based testing
# ---------------------------------------------------------------------------


def run_with_schedule(
    schedule: List[int],
    setup: Callable,
    threads: List[Callable],
    timeout: float = 5.0,
    cooperative_locks: bool = True,
) -> Any:
    """Run one interleaving and return the state object.

    Args:
        schedule: Opcode-level schedule (list of thread indices).
        setup: Returns fresh shared state.
        threads: Callables that each receive the state as their argument.
        timeout: Max seconds.
        cooperative_locks: Replace threading.Lock with a scheduler-aware
            version that prevents deadlocks (default True).

    Returns:
        The state object after execution.
    """
    scheduler = OpcodeScheduler(schedule, len(threads))
    runner = BytecodeInterlace(scheduler, cooperative_locks=cooperative_locks)

    # Patch locks BEFORE setup() so any locks created there are cooperative
    runner._patch_locks()
    try:
        state = setup()
        funcs = [lambda s=state, t=t: t(s) for t in threads]
        runner.run(funcs, timeout=timeout)
    except TimeoutError:
        pass
    finally:
        runner._unpatch_locks()
    return state


def explore_interleavings(
    setup: Callable,
    threads: List[Callable],
    invariant: Callable[[Any], bool],
    max_attempts: int = 200,
    max_ops: int = 300,
    timeout_per_run: float = 5.0,
    seed: Optional[int] = None,
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
        >>> from interlace.bytecode import schedule_strategy, run_with_schedule
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
