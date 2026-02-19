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
import random
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from frontrun._cooperative import (
    clear_context,
    patch_locks,
    real_lock,
    set_context,
    unpatch_locks,
)
from frontrun.common import InterleavingResult

# Type variable for the shared state passed between setup and thread functions
T = TypeVar("T")

_PY_VERSION = sys.version_info[:2]
# sys.monitoring (PEP 669) is available since 3.12 and is required for
# free-threaded builds (3.13t/3.14t) where sys.settrace + f_trace_opcodes
# has a known crash bug (CPython #118415).
_USE_SYS_MONITORING = _PY_VERSION >= (3, 12)

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
        self._lock = real_lock()
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

    # sys.monitoring tool ID (use OPTIMIZER_ID to avoid conflict with DPOR's PROFILER_ID)
    _TOOL_ID: int | None = None

    def __init__(self, scheduler: OpcodeScheduler, cooperative_locks: bool = True):
        self.scheduler = scheduler
        self.cooperative_locks = cooperative_locks
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False
        self._monitoring_active = False

    def _patch_locks(self):
        """Replace threading and queue primitives with cooperative versions."""
        if not self.cooperative_locks:
            return
        patch_locks()
        self._lock_patched = True

    def _unpatch_locks(self):
        """Restore the original threading and queue primitives."""
        if self._lock_patched:
            unpatch_locks()
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

    # --- sys.monitoring backend (3.12+) ---

    def _setup_monitoring(self) -> None:
        """Set up sys.monitoring INSTRUCTION events for opcode-level scheduling."""
        if not _USE_SYS_MONITORING:
            return

        mon = sys.monitoring
        tool_id = mon.OPTIMIZER_ID  # type: ignore[attr-defined]
        BytecodeShuffler._TOOL_ID = tool_id

        mon.use_tool_id(tool_id, "frontrun-bytecode")  # type: ignore[attr-defined]
        mon.set_events(tool_id, mon.events.PY_START | mon.events.INSTRUCTION)  # type: ignore[attr-defined]

        scheduler = self.scheduler

        def handle_py_start(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return mon.DISABLE  # type: ignore[attr-defined]
            if not _should_trace_file(code.co_filename):
                return mon.DISABLE  # type: ignore[attr-defined]
            return None

        def handle_instruction(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return None
            if not _should_trace_file(code.co_filename):
                return None

            from frontrun._cooperative import _scheduler_tls

            thread_id = getattr(_scheduler_tls, "thread_id", None)
            if thread_id is None:
                return None

            scheduler.wait_for_turn(thread_id)
            return None

        mon.register_callback(tool_id, mon.events.PY_START, handle_py_start)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)  # type: ignore[attr-defined]
        self._monitoring_active = True

    def _teardown_monitoring(self) -> None:
        """Remove sys.monitoring callbacks and free the tool ID."""
        if not self._monitoring_active:
            return
        mon = sys.monitoring
        tool_id = BytecodeShuffler._TOOL_ID
        if tool_id is not None:
            mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
            mon.free_tool_id(tool_id)  # type: ignore[attr-defined]
        self._monitoring_active = False

    # --- Thread entry points ---

    def _run_thread_settrace(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Thread entry using sys.settrace (3.10-3.11)."""
        try:
            set_context(self.scheduler, thread_id)

            trace_fn = self._make_trace(thread_id)
            sys.settrace(trace_fn)
            func(*args, **kwargs)
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            sys.settrace(None)
            clear_context()
            self.scheduler.mark_done(thread_id)

    def _run_thread_monitoring(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Thread entry using sys.monitoring (3.12+)."""
        try:
            set_context(self.scheduler, thread_id)

            func(*args, **kwargs)
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            clear_context()
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

        use_monitoring = _USE_SYS_MONITORING
        if use_monitoring:
            self._setup_monitoring()
            run_thread = self._run_thread_monitoring
        else:
            run_thread = self._run_thread_settrace

        try:
            for i, (func, a, kw) in enumerate(zip(funcs, args, kwargs)):
                t = threading.Thread(
                    target=run_thread,
                    args=(i, func, a, kw),
                    name=f"frontrun-{i}",
                    daemon=True,
                )
                self.threads.append(t)

            for t in self.threads:
                t.start()

            for t in self.threads:
                t.join(timeout=timeout)
        finally:
            if use_monitoring:
                self._teardown_monitoring()

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
