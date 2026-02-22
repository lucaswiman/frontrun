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

import random
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from frontrun._cooperative import (
    clear_context,
    patch_locks,
    real_condition,
    real_lock,
    set_context,
    unpatch_locks,
)
from frontrun._deadlock import SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._io_detection import (
    patch_io,
    set_io_reporter,
    unpatch_io,
)
from frontrun._trace_format import TraceRecorder, format_trace
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun.cli import require_active as _require_frontrun_env
from frontrun.common import InterleavingResult

# Type variable for the shared state passed between setup and thread functions
T = TypeVar("T")

_PY_VERSION = sys.version_info[:2]
# sys.monitoring (PEP 669) is available since 3.12 and is required for
# free-threaded builds (3.13t/3.14t) where sys.settrace + f_trace_opcodes
# has a known crash bug (CPython #118415).
_USE_SYS_MONITORING = _PY_VERSION >= (3, 12)


class OpcodeScheduler:
    """Controls thread execution at bytecode instruction granularity.

    The schedule is a list of thread indices. Each entry means "let this
    thread execute one bytecode instruction."

    When the explicit schedule is exhausted, the scheduler dynamically
    extends it with round-robin entries so that threads remain under
    deterministic scheduler control instead of falling back to real
    (non-deterministic) concurrency.  A hard cap (``max_ops``) limits
    the total number of scheduler steps to prevent infinite runs.

    Deadlock detection uses a configurable fallback ``condition.wait``
    timeout (default 5 s) for threads stuck in C extensions or other
    unmanaged blocking calls.  When cooperative locks are enabled, the
    :class:`~frontrun._deadlock.WaitForGraph` provides instant
    lock-ordering cycle detection.
    """

    def __init__(
        self,
        schedule: list[int],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        max_ops: int = 0,
        trace_recorder: TraceRecorder | None = None,
    ):
        self.schedule = list(schedule)  # mutable copy for dynamic extension
        self.num_threads = num_threads
        self.deadlock_timeout = deadlock_timeout
        self._max_ops = max_ops if max_ops > 0 else len(schedule) * 10 + 10000
        self._index = 0
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self.trace_recorder = trace_recorder

    def _extend_schedule(self) -> bool:
        """Append a round-robin round of all active threads.

        Returns True if the schedule was extended, False if all threads
        are done or the max_ops cap was reached.
        """
        if self._index >= self._max_ops:
            return False
        active = [t for t in range(self.num_threads) if t not in self._threads_done]
        if not active:
            return False
        self.schedule.extend(active)
        return True

    def wait_for_turn(self, thread_id: int) -> bool:
        """Block until it's this thread's turn. Returns False when done."""
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False

                if self._index >= len(self.schedule):
                    if not self._extend_schedule():
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

                if not self._condition.wait(timeout=self.deadlock_timeout):
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

    Replaces threading and queue primitives (Lock, RLock, Semaphore,
    BoundedSemaphore, Event, Condition, Queue, LifoQueue, PriorityQueue)
    with cooperative versions that yield scheduler turns instead of
    blocking in C. This prevents the deadlock that otherwise occurs when
    one thread holds a primitive and the scheduler gives a turn to
    another thread that tries to acquire it.
    """

    # sys.monitoring tool ID (use OPTIMIZER_ID to avoid conflict with DPOR's PROFILER_ID)
    _TOOL_ID: int | None = None

    def __init__(self, scheduler: OpcodeScheduler, detect_io: bool = True):
        self.scheduler = scheduler
        self.detect_io = detect_io
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False
        self._io_patched = False
        self._monitoring_active = False

    def _patch_locks(self):
        """Replace threading and queue primitives with cooperative versions."""
        install_wait_for_graph()
        patch_locks()
        self._lock_patched = True

    def _unpatch_locks(self):
        """Restore the original threading and queue primitives."""
        if self._lock_patched:
            unpatch_locks()

            uninstall_wait_for_graph()
            self._lock_patched = False

    def _patch_io(self):
        """Replace socket and open with traced versions."""
        if not self.detect_io:
            return
        patch_io()
        self._io_patched = True

    def _unpatch_io(self):
        """Restore original socket and open implementations."""
        if self._io_patched:
            unpatch_io()
            self._io_patched = False

    def _make_trace(self, thread_id: int) -> Callable[[Any, str, Any], Any]:  # type: ignore[return-value]
        """Create a sys.settrace function for the given thread."""
        scheduler = self.scheduler
        recorder = scheduler.trace_recorder

        def trace(frame: Any, event: str, arg: Any) -> Any:  # type: ignore[name-defined]
            if scheduler._finished or scheduler._error:
                return None

            if event == "call":
                if _should_trace_file(frame.f_code.co_filename):
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == "opcode":
                if recorder is not None:
                    recorder.record_from_opcode(thread_id, frame)
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
            # Only use mon.DISABLE for code that should *never* be traced
            # (stdlib, site-packages, frontrun internals).  Do NOT disable
            # for transient conditions like scheduler._finished — DISABLE
            # permanently removes INSTRUCTION events from the code object,
            # corrupting monitoring state for subsequent iterations and
            # tests that share the same tool ID.
            if not _should_trace_file(code.co_filename):
                return mon.DISABLE  # type: ignore[attr-defined]
            return None

        recorder = scheduler.trace_recorder

        def handle_instruction(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return None
            if not _should_trace_file(code.co_filename):
                return None

            from frontrun._cooperative import _scheduler_tls

            thread_id = getattr(_scheduler_tls, "thread_id", None)
            if thread_id is None:
                return None
            # Guard against zombie threads from a previous runner
            if getattr(_scheduler_tls, "scheduler", None) is not scheduler:
                return None

            if recorder is not None:
                frame = sys._getframe(1)
                recorder.record_from_opcode(thread_id, frame)

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

    def _setup_io_reporter(self, thread_id: int) -> None:
        """Install IO reporter that forces a scheduling point on IO events."""
        if not self.detect_io:
            return
        scheduler = self.scheduler

        def _io_reporter(resource_id: str, kind: str) -> None:
            # Force a scheduling point around IO operations so the random
            # exploration can try different orderings of IO vs other threads.
            if not scheduler._finished and not scheduler._error:
                scheduler.wait_for_turn(thread_id)

        set_io_reporter(_io_reporter)

    def _teardown_io_reporter(self) -> None:
        """Remove the IO reporter for the current thread."""
        if self.detect_io:
            set_io_reporter(None)

    def _run_thread_settrace(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Thread entry using sys.settrace (3.10-3.11)."""
        try:
            set_context(self.scheduler, thread_id)
            self._setup_io_reporter(thread_id)

            trace_fn = self._make_trace(thread_id)
            sys.settrace(trace_fn)
            func(*args, **kwargs)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            sys.settrace(None)
            self._teardown_io_reporter()
            clear_context()
            self.scheduler.mark_done(thread_id)

    def _run_thread_monitoring(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Thread entry using sys.monitoring (3.12+)."""
        try:
            set_context(self.scheduler, thread_id)
            self._setup_io_reporter(thread_id)

            func(*args, **kwargs)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            self._teardown_io_reporter()
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
            timeout: Max total wait time for all threads (global deadline).
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

            deadline = time.monotonic() + timeout
            for t in self.threads:
                remaining = max(0, deadline - time.monotonic())
                t.join(timeout=remaining)
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
    detect_io: bool = True,
    debug: bool = False,
    deadlock_timeout: float = 5.0,
    trace_recorder: TraceRecorder | None = None,
) -> T:
    """Run one interleaving and return the state object.

    Args:
        schedule: Opcode-level schedule (list of thread indices).
        setup: Returns fresh shared state.
        threads: Callables that each receive the state as their argument.
        timeout: Max seconds.
        detect_io: Automatically detect socket/file I/O and treat them
            as scheduling points (default True).
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).
        trace_recorder: Optional recorder for capturing trace events.
            When provided, records shared-state accesses for later
            formatting into human-readable explanations.

    Returns:
        The state object after execution.
    """
    scheduler = OpcodeScheduler(
        schedule, len(threads), deadlock_timeout=deadlock_timeout, trace_recorder=trace_recorder
    )
    runner = BytecodeShuffler(scheduler, detect_io=detect_io)

    # Patch locks BEFORE setup() so any locks created there are cooperative
    runner._patch_locks()
    runner._patch_io()
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
            if debug:
                print(f"Timed out with {timeout=} on {schedule=}", flush=True)
    finally:
        runner._unpatch_io()
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
    debug: bool = False,
    detect_io: bool = True,
    deadlock_timeout: float = 5.0,
    reproduce_on_failure: int = 10,
) -> InterleavingResult:
    """Search for interleavings that violate an invariant.

    .. note::

       When running under **pytest**, this function requires the
       ``frontrun`` CLI wrapper (``frontrun pytest ...``) or the
       ``--frontrun-patch-locks`` flag.  Without it, the test is
       automatically skipped.

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
        detect_io: Automatically detect socket/file I/O and treat them
            as scheduling points (default True).
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).
        reproduce_on_failure: When a counterexample is found, replay the
            same schedule this many times to measure reproducibility
            (default 10).  Set to 0 to skip reproduction testing.

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct execution orderings were observed,
        providing a lower bound on exploration coverage.
    """
    _require_frontrun_env("explore_interleavings")
    rng = random.Random(seed)
    num_threads = len(threads)
    result = InterleavingResult(property_holds=True, num_explored=0)
    seen_schedule_hashes: set[int] = set()

    for _ in range(max_attempts):
        num_rounds = rng.randint(1, max(1, max_ops // num_threads))
        schedule: list[int] = []
        for _ in range(num_rounds):
            round_perm = list(range(num_threads))
            rng.shuffle(round_perm)
            schedule.extend(round_perm)

        if debug:
            print(f"Running with {schedule=} {threads=}", flush=True)
        recorder = TraceRecorder()
        state = run_with_schedule(
            schedule,
            setup,
            threads,
            timeout=timeout_per_run,
            detect_io=detect_io,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=recorder,
        )
        result.num_explored += 1
        seen_schedule_hashes.add(hash(tuple(schedule)))

        if not invariant(state):
            result.property_holds = False
            result.counterexample = schedule
            result.unique_interleavings = len(seen_schedule_hashes)

            # Replay the counterexample to measure reproducibility
            if reproduce_on_failure > 0:
                successes = 0
                for _ in range(reproduce_on_failure):
                    try:
                        replay_state = run_with_schedule(
                            schedule,
                            setup,
                            threads,
                            timeout=timeout_per_run,
                            detect_io=detect_io,
                            deadlock_timeout=deadlock_timeout,
                        )
                        if not invariant(replay_state):
                            successes += 1
                    except Exception:
                        pass  # timeout / crash during replay — not a reproduction
                result.reproduction_attempts = reproduce_on_failure
                result.reproduction_successes = successes

            result.explanation = format_trace(
                recorder.events,
                num_threads=num_threads,
                num_explored=result.num_explored,
                reproduction_attempts=result.reproduction_attempts,
                reproduction_successes=result.reproduction_successes,
            )

            return result

    result.unique_interleavings = len(seen_schedule_hashes)
    return result


def schedule_strategy(num_threads: int, max_ops: int = 300):
    """Hypothesis strategy for generating fair opcode schedules.

    Generates schedules as a sequence of rounds, where each round is a
    random permutation of all thread indices.  This guarantees every thread
    gets exactly the same number of scheduling slots, preventing starvation
    (e.g. a schedule that gives 99 % of steps to one thread).

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

    max_rounds = max(1, max_ops // num_threads)
    threads = list(range(num_threads))

    @st.composite
    def _fair_schedule(draw: st.DrawFn) -> list[int]:
        num_rounds = draw(st.integers(min_value=1, max_value=max_rounds))
        schedule: list[int] = []
        for _ in range(num_rounds):
            schedule.extend(draw(st.permutations(threads)))
        return schedule

    return _fair_schedule()
