"""
Bytecode-level deterministic concurrency testing.

Uses sys.settrace with f_trace_opcodes to intercept execution at every
bytecode instruction, enabling fine-grained control over thread interleaving.

This pairs naturally with property-based testing: rather than specifying exact
schedules, generate random interleavings and check that invariants hold (or
that bugs can be found).

The core insight: CPython context switches happen between bytecode instructions.
By controlling which thread gets to execute each instruction, we can explore a
broad range of possible interleavings.  Schedules use variable-length per-thread
bursts (see :mod:`frontrun._random_schedules`) so that two threads can drift
more than one opcode apart — this reaches races that require one thread to run
several opcodes inside a narrow window of another.  Random exploration is a
*sampler*, not an exhaustive enumeration (use the DPOR strategy for systematic
coverage).

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
    >>> assert result.property_holds, result.explanation  # fails — lost update!
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
    patch_sleep,
    real_condition,
    real_lock,
    set_context,
    unpatch_locks,
    unpatch_sleep,
)
from frontrun._deadlock import DeadlockError, SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._io_detection import (
    patch_io,
    set_dpor_scheduler,
    set_dpor_thread_id,
    set_io_reporter,
    unpatch_io,
)
from frontrun._opcode_observer import (
    OpcodeTraceHandle,
    install_thread_opcode_trace,
    start_opcode_trace,
    stop_opcode_trace,
    uninstall_thread_opcode_trace,
)
from frontrun._random_schedules import (
    _DEFAULT_MAX_BURST,
    fair_schedule_strategy,
    random_round_robin_schedule,
)
from frontrun._sql_cursor import patch_sql, unpatch_sql
from frontrun._sql_insert_tracker import check_uncaptured_inserts, clear_insert_tracker
from frontrun._threaded_runner import PatchScope, notify_scheduler_timeout, run_thread_group
from frontrun._trace_format import TraceRecorder, build_call_chain, format_trace
from frontrun._tracing import TraceFilter as _TraceFilter
from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun.cli import require_active as _require_frontrun_env
from frontrun.common import (
    DEPRECATION_MESSAGES,
    InterleavingResult,
    check_invariant,
    check_serializability_violation,
    deprecate,
)

# Type variable for the shared state passed between setup and thread functions
T = TypeVar("T")


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
        # Deterministic RNG for dynamic schedule extension.  Seeded from the
        # initial schedule so the extension is reproducible for a given run
        # (and replay re-uses the already-recorded self.schedule list anyway).
        self._extend_rng = random.Random(hash(tuple(schedule)) & 0xFFFFFFFF)
        self._index = 0
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self.trace_recorder = trace_recorder

    def _extend_schedule(self) -> bool:
        """Append a round of all active threads with variable-length bursts.

        Each active thread gets a burst of consecutive slots (rather than a
        single slot) so that, like the initial random schedule, dynamically
        extended schedules can express relative opcode drift > 1 between
        threads.  Returns True if the schedule was extended, False if all
        threads are done or the max_ops cap was reached.
        """
        if self._index >= self._max_ops:
            return False
        active = [t for t in range(self.num_threads) if t not in self._threads_done]
        if not active:
            return False
        # Only add entries up to the max_ops cap to prevent overshoot.
        remaining = self._max_ops - len(self.schedule)
        if remaining <= 0:
            return False
        self._extend_rng.shuffle(active)
        appended: list[int] = []
        for actor in active:
            burst = self._extend_rng.randint(1, _DEFAULT_MAX_BURST)
            appended.extend([actor] * burst)
        self.schedule.extend(appended[:remaining])
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

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Compatibility with DporScheduler for SQL scheduling points.

        ``_intercept_execute`` in ``_sql_cursor.py`` calls
        ``scheduler.report_and_wait(None, thread_id)`` to create a
        scheduling point at each SQL statement.  During replay, the
        OpcodeScheduler is registered as the dpor scheduler, so this
        method is called instead of DporScheduler's version.
        """
        return self.wait_for_turn(thread_id)

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

    def __init__(self, scheduler: OpcodeScheduler, detect_io: bool = True):
        self.scheduler = scheduler
        self.detect_io = detect_io
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False
        self._io_patched = False
        self._sleep_patched = False
        self._opcode_handle: OpcodeTraceHandle | None = None

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

    def _patch_sleep(self):
        """Replace time.sleep with a cooperative no-op."""
        patch_sleep()
        self._sleep_patched = True

    def _unpatch_sleep(self):
        """Restore original time.sleep."""
        if self._sleep_patched:
            unpatch_sleep()
            self._sleep_patched = False

    def _patch_io(self):
        """Replace socket and open with traced versions."""
        if not self.detect_io:
            return
        patch_io()
        patch_sql()
        self._io_patched = True

    def _unpatch_io(self):
        """Restore original socket and open implementations."""
        if self._io_patched:
            unpatch_sql()
            unpatch_io()
            self._io_patched = False

    def patch_scope(self, *, patch_sleep: bool = True) -> PatchScope:
        scope = PatchScope()
        scope.add(self._patch_locks, self._unpatch_locks)
        scope.add(self._patch_io, self._unpatch_io)
        scope.add(self._patch_sleep, self._unpatch_sleep, enabled=patch_sleep)
        return scope

    def _start_opcode_trace(self) -> None:
        """Construct opcode tracing via the tracer-backend.

        On sys.monitoring (3.12+) this also installs the global tool;
        per-thread activation is a no-op.  On sys.settrace (3.10-3.11) this
        only constructs the trace callback; per-thread activation happens
        in :meth:`_thread_runtime` via ``install_thread_opcode_trace``.

        Uses the OPTIMIZER tool ID slot (DPOR uses PROFILER) and skips
        PY_RETURN registration since BytecodeShuffler does not maintain a
        shadow stack.
        """
        scheduler = self.scheduler
        recorder = scheduler.trace_recorder
        from frontrun._cooperative import _scheduler_tls

        def _get_tid() -> int | None:
            tid = getattr(_scheduler_tls, "thread_id", None)
            if tid is None:
                return None
            # Guard against zombie threads from a previous runner instance
            # (sys.monitoring is global, so old threads can still trip the
            # callback after this scheduler's run ends).
            if getattr(_scheduler_tls, "scheduler", None) is not scheduler:
                return None
            return tid  # type: ignore[no-any-return]

        def _on_opcode(code: Any, offset: int, frame: Any, tid: int) -> bool:
            if recorder is not None:
                recorder.record_from_opcode(tid, frame)
            scheduler.wait_for_turn(tid)
            # Returning False suppresses the PEP 667 LocalsToFast workaround
            # in make_settrace_callback; BytecodeShuffler historically did
            # not apply it.
            return False

        self._opcode_handle = start_opcode_trace(
            get_thread_id=_get_tid,
            on_opcode=_on_opcode,
            is_active=lambda: not (scheduler._finished or scheduler._error),
            tool_name="frontrun-bytecode",
            tool_kind="optimizer",
            monitor_returns=False,
        )

    def _stop_opcode_trace(self) -> None:
        """Tear down opcode tracing started by :meth:`_start_opcode_trace`."""
        handle = self._opcode_handle
        if handle is None:
            return
        stop_opcode_trace(handle)
        self._opcode_handle = None

    # --- Thread entry points ---

    def _setup_io_reporter(self, thread_id: int) -> None:
        """Install IO reporter that forces a scheduling point on IO events."""
        if not self.detect_io:
            return
        scheduler = self.scheduler
        recorder = scheduler.trace_recorder

        def _io_reporter(resource_id: str, kind: str) -> None:
            # Force a scheduling point around IO operations so the random
            # exploration can try different orderings of IO vs other threads.
            if not scheduler._finished and not scheduler._error:
                scheduler.wait_for_turn(thread_id)
            # Record I/O event in the trace for human-readable output
            if recorder is not None:
                _frame = sys._getframe(1)
                while _frame is not None and not _should_trace_file(_frame.f_code.co_filename):
                    _frame = _frame.f_back
                if _frame is not None:
                    chain = build_call_chain(_frame, filter_fn=_should_trace_file)
                    recorder.record(
                        thread_id=thread_id,
                        frame=_frame,
                        opcode="IO",
                        access_type=kind,
                        attr_name=resource_id,
                        call_chain=chain,
                    )

        set_io_reporter(_io_reporter)

    def _teardown_io_reporter(self) -> None:
        """Remove the IO reporter for the current thread."""
        if self.detect_io:
            set_io_reporter(None)

    @contextmanager
    def _thread_runtime(self, thread_id: int):
        set_context(self.scheduler, thread_id)
        self._setup_io_reporter(thread_id)
        # Register OpcodeScheduler as the dpor scheduler so that
        # _intercept_execute (SQL cursor patching) can create scheduling
        # points during replay, matching the schedule generated by DPOR.
        set_dpor_scheduler(self.scheduler)
        set_dpor_thread_id(thread_id)
        # On sys.settrace (3.10-3.11) this activates the per-thread tracer;
        # on sys.monitoring (3.12+) it is a no-op (the tool was registered
        # globally in run()).
        handle = self._opcode_handle
        if handle is not None:
            install_thread_opcode_trace(handle)
        try:
            yield
        finally:
            if handle is not None:
                uninstall_thread_opcode_trace(handle)
            self._teardown_io_reporter()
            set_dpor_scheduler(None)
            set_dpor_thread_id(None)
            clear_context()
            self.scheduler.mark_done(thread_id)

    def _run_thread(
        self, thread_id: int, func: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        try:
            with self._thread_runtime(thread_id):
                func(*args, **kwargs)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)

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

        self._start_opcode_trace()
        run_thread = self._run_thread

        thread_args = [(a, kw) for a, kw in zip(args, kwargs)]

        def make_thread_target(
            thread_id: int,
            func: Callable[..., None],
            packed_args: tuple[Any, ...],
        ) -> Callable[[], None]:
            a, kw = packed_args

            def target() -> None:
                run_thread(thread_id, func, a, kw)

            return target

        def on_timeout(alive: list[threading.Thread]) -> None:
            notify_scheduler_timeout(self.scheduler, alive)

        run_thread_group(
            funcs=funcs,
            args=thread_args,
            make_thread_target=make_thread_target,
            name_prefix="frontrun",
            timeout=timeout,
            thread_store=self.threads,
            teardown=self._stop_opcode_trace,
            on_timeout=on_timeout,
        )

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
    patch_sleep: bool = True,
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
    with runner.patch_scope(patch_sleep=patch_sleep):
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
        # Re-raise DeadlockError so callers (e.g. reproduction logic) can
        # detect that a deadlock occurred during replay.
        if isinstance(scheduler._error, DeadlockError):
            raise scheduler._error
    return state


def explore_random(
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
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    trace_packages: list[str] | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[T], Any] | bool = False,
    error_on_any_race: bool = False,
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
        total_timeout: Maximum total time in seconds for the entire
            exploration (default None = unlimited).  When exceeded, returns
            results gathered so far.
        warn_nondeterministic_sql: If True (default), raise
            :class:`~frontrun.common.NondeterministicSQLError` when SQL
            INSERT statements are detected but ``lastrowid`` capture
            failed (e.g. psycopg2 without RETURNING).  Set to False to
            suppress.  When capture succeeds, INSERTs use stable
            indexical resource IDs automatically.
        trace_packages: List of package name patterns (fnmatch syntax) to
            trace in addition to user code.  By default, code in
            site-packages is skipped.  Use this to include specific
            installed packages, e.g. ``["django_*", "mylib.*"]``.

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct execution orderings were observed,
        providing a lower bound on exploration coverage.
    """
    _require_frontrun_env("explore_random")
    if error_on_any_race:
        raise ValueError("error_on_any_race requires DPOR (use explore_dpor instead)")
    if trace_packages is not None:
        _set_active_trace_filter(_TraceFilter(trace_packages))
    try:
        from frontrun._dpor_core import compute_serializable_baseline_sync

        serial_valid_states, serial_hash_fn = compute_serializable_baseline_sync(setup, threads, serializable_invariant)

        rng = random.Random(seed)
        num_threads = len(threads)
        result = InterleavingResult(property_holds=True, num_explored=0)
        seen_schedule_hashes: set[int] = set()
        total_deadline = time.monotonic() + total_timeout if total_timeout is not None else None

        for _ in range(max_attempts):
            if total_deadline is not None and time.monotonic() > total_deadline:
                break
            schedule = random_round_robin_schedule(rng, num_threads, max_ops)

            clear_insert_tracker()
            if debug:
                print(f"Running with {schedule=} {threads=}", flush=True)
            recorder = TraceRecorder()
            try:
                state = run_with_schedule(
                    schedule,
                    setup,
                    threads,
                    timeout=timeout_per_run,
                    detect_io=detect_io,
                    debug=debug,
                    deadlock_timeout=deadlock_timeout,
                    trace_recorder=recorder,
                    patch_sleep=patch_sleep,
                )
            except DeadlockError as dl_err:
                result.num_explored += 1
                seen_schedule_hashes.add(hash(tuple(schedule)))
                result.property_holds = False
                result.counterexample = schedule
                result.unique_interleavings = len(seen_schedule_hashes)
                result.explanation = (
                    f"Deadlock detected after {result.num_explored} interleaving(s).\n\n{dl_err.cycle_description}"
                )
                return result
            result.num_explored += 1
            seen_schedule_hashes.add(hash(tuple(schedule)))

            if warn_nondeterministic_sql:
                check_uncaptured_inserts()

            # --- serializable_invariant check ---
            if serial_valid_states is not None:
                explanation = check_serializability_violation(
                    state, serial_valid_states, serial_hash_fn, result.num_explored
                )
                if explanation is not None:
                    result.property_holds = False
                    result.counterexample = schedule
                    result.unique_interleavings = len(seen_schedule_hashes)
                    result.explanation = explanation
                    return result

            invariant_failed, assertion_msg = check_invariant(invariant, state)
            if invariant_failed:
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
                                patch_sleep=patch_sleep,
                            )
                            replay_failed, _ = check_invariant(invariant, replay_state)
                            if replay_failed:
                                successes += 1
                        except Exception:
                            pass  # timeout / crash during replay — not a reproduction
                    result.reproduction_attempts = reproduce_on_failure
                    result.reproduction_successes = successes

                trace_explanation = format_trace(
                    recorder.events,
                    num_threads=num_threads,
                    num_explored=result.num_explored,
                    reproduction_attempts=result.reproduction_attempts,
                    reproduction_successes=result.reproduction_successes,
                )
                if assertion_msg:
                    result.explanation = f"AssertionError: {assertion_msg}\n\n{trace_explanation}"
                else:
                    result.explanation = trace_explanation

                return result

        result.unique_interleavings = len(seen_schedule_hashes)
        return result
    finally:
        _set_active_trace_filter(None)


explore_interleavings = deprecate(explore_random, DEPRECATION_MESSAGES["explore_interleavings"])
explore_interleavings.__doc__ = (
    "Deprecated alias for :func:`explore_random`.\n\n"
    ".. deprecated:: 0.5\n"
    "    ``explore_interleavings`` will be removed in 0.6. Use\n"
    "    :func:`frontrun.explore` with ``strategy='random'`` instead."
)


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
    return fair_schedule_strategy(num_threads, max_ops)
