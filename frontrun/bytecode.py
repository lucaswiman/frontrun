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

    >>> import frontrun
    >>>
    >>> class Counter:
    ...     def __init__(self):
    ...         self.value = 0
    ...     def increment(self):
    ...         temp = self.value
    ...         self.value = temp + 1
    >>>
    >>> result = frontrun.explore_random(
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

from frontrun._certificate import PassEvidence, certify_pass
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
from frontrun._dpor_core import VirtualClockPort, noop_on_wake
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
    burst_round,
    fair_schedule_strategy,
    random_round_robin_schedule,
)
from frontrun._sql_cursor import patch_sql, unpatch_sql
from frontrun._sql_insert_tracker import clear_insert_tracker, ensure_no_uncaptured_inserts
from frontrun._threaded_runner import PatchScope, notify_scheduler_timeout, run_thread_group
from frontrun._trace_format import TraceRecorder, build_call_chain, format_trace
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun._tracing import trace_filter_scope
from frontrun._virtual_clock import (
    ClockConfig,
    ClockMode,
    VirtualClock,
    clock_scope,
    warn_if_captured_time_reference,
)
from frontrun.cli import require_active as _require_frontrun_env
from frontrun.common import (
    InterleavingResult,
    _call_sync_setup,
    check_invariant,
    record_serializability_violation,
    validate_random_exploration_params,
)

# Type variable for the shared state passed between setup and thread functions
T = TypeVar("T")


class _WorkerExecutionError(Exception):
    """Internal wrapper distinguishing worker crashes from setup failures."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class _MaxOpsExhaustedError(TimeoutError):
    """The scheduler cap was reached before the worker trace completed."""


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
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_diagnostics: bool = False,
    ):
        self.schedule = list(schedule)  # mutable copy for dynamic extension
        self.num_threads = num_threads
        self.deadlock_timeout = deadlock_timeout
        # Virtual clock (ideas/virtual_clock.md).  Random exploration has no
        # DPOR engine, so the clock advance rules live directly here:
        # - a schedule entry for a sleeping thread is skipped ("virtual") or
        #   advances the clock to that thread's deadline ("explored" — the
        #   random sampler's "maybe advance time" branch);
        # - when every live thread is deadline-blocked, the clock autojumps
        #   to the earliest deadline (see sleep_until).
        self.virtual_clock = virtual_clock
        self.clock_mode = clock_mode
        self._clock_diagnostics = clock_diagnostics
        self._max_ops = max_ops if max_ops > 0 else len(schedule) * 10 + 10000
        self._max_ops_exhausted = False
        # Deterministic RNG for dynamic schedule extension.  Seeded from the
        # initial schedule so the extension is reproducible for a given run
        # (and replay re-uses the already-recorded self.schedule list anyway).
        self._extend_rng = random.Random(hash(tuple(schedule)) & 0xFFFFFFFF)
        self._index = 0
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        # Shared virtual-clock protocol.  Random exploration has no DPOR engine,
        # so the block/unblock/sync callbacks default to no-ops; the port still
        # owns the single DeadlineCoordinator and the blocking-spin flags.
        # ``_deadlines`` / ``_spin_waiters`` alias the port's state so the rest
        # of the scheduler is unchanged.  The spin flags matter because an
        # untimed spinner otherwise blocks the "every live thread is
        # deadline-blocked" autojump forever (see sleep_until); entries clear on
        # release/set so a freshly-unblocked spinner re-probes before counting
        # as hopeless.
        self._clock_port = VirtualClockPort(condition=self._condition)
        self._deadlines = self._clock_port.coordinator
        self._spin_waiters = self._clock_port.spin_waiters
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
            if any(t not in self._threads_done for t in range(self.num_threads)):
                self._max_ops_exhausted = True
            return False
        active = [t for t in range(self.num_threads) if t not in self._threads_done]
        if not active:
            return False
        # Only add entries up to the max_ops cap to prevent overshoot.
        remaining = self._max_ops - len(self.schedule)
        if remaining <= 0:
            if active:
                self._max_ops_exhausted = True
            return False
        appended = burst_round(self._extend_rng, active)
        self.schedule.extend(appended[:remaining])
        return True

    def _blocks_clock_progress(self, thread_id: int) -> bool:
        return self._clock_port.blocks_clock_progress(thread_id)

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

                if self.virtual_clock is not None and self._deadlines.is_sleeping(scheduled_tid):
                    if self.clock_mode == "explored":
                        # "Maybe advance": the random schedule picked a
                        # sleeping thread, so let time pass toward its
                        # deadline; the woken thread then consumes this entry.
                        # Each speculative hop is clamped to the *earliest*
                        # pending deadline so an earlier timed wait fires at
                        # its own clock value, never at a later sleeper's
                        # target.  A clamped hop cannot wake the scheduled
                        # sleeper, so it also consumes the sleeper's contiguous
                        # run of entries: leaving them would re-advance to the
                        # next deadline under this same lock hold, before the
                        # just-woken waiter could be granted a single turn to
                        # observe its own timeout.
                        sleep_deadline = self._deadlines.sleep_deadline(scheduled_tid)
                        if sleep_deadline is None:
                            # A perpetual sleep is blocked but cannot advance
                            # virtual time. Skip this unusable schedule slot.
                            self._index += 1
                            self._condition.notify_all()
                            continue
                        next_deadline = self._deadlines.next_deadline()
                        assert next_deadline is not None  # sleep_deadline is pending
                        self._advance_clock_to(min(next_deadline, sleep_deadline))
                        if next_deadline < sleep_deadline:
                            # A clamped hop can't wake the scheduled sleeper, so
                            # consume its contiguous run of entries (see above).
                            while self._index < len(self.schedule) and self.schedule[self._index] == scheduled_tid:
                                self._index += 1
                    else:
                        # Autojump semantics: a sleeping thread cannot run
                        # before the clock advances; skip its slot.
                        self._index += 1
                    self._condition.notify_all()
                    continue

                if (
                    self.virtual_clock is not None
                    and scheduled_tid == thread_id
                    and self._deadlines.in_timed_wait(thread_id)
                    and all(
                        self._blocks_clock_progress(t)
                        for t in range(self.num_threads)
                        if t != thread_id and t not in self._threads_done
                    )
                ):
                    # This thread spins on a timed lock acquire and every
                    # other live thread is deadline-blocked: nothing can
                    # release the lock before time passes, so advance to the
                    # earliest pending deadline (which may be our own).
                    deadline = self._deadlines.next_deadline()
                    if deadline is not None:
                        self._advance_clock_to(deadline)

                if scheduled_tid == thread_id:
                    self._index += 1
                    self._condition.notify_all()
                    return True

                if not self._condition.wait(timeout=self.deadlock_timeout):
                    # Re-check terminal conditions before indexing: another
                    # thread may have finished the run or advanced past the end
                    # of the schedule while we were blocked in wait() (9a).
                    if self._finished or self._error:
                        return False
                    if self._index >= len(self.schedule):
                        continue
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
            self._deadlines.cancel(thread_id)
            self._spin_waiters.pop(thread_id, None)
            self._condition.notify_all()

    # -- Virtual clock support (see class docstring fields) ---------------

    def _advance_clock_to(self, target: float) -> None:
        """Jump the clock to *target* and wake every due deadline.

        Caller must hold ``self._condition``.  The shared port drops each due
        deadline from the coordinator and pops the due actor's spin flag.  That
        spin-flag pop is load-bearing: a virtual timed wait registers a
        "timeout" deadline AND a ``_spin_waiters`` flag (see
        ``_cooperative._spin_hook_for_wait``); its deadline firing means
        "re-probe before being counted as blocked again", so without clearing
        the flag the autojump loop in ``sleep_until`` would keep seeing it as
        blocked and advance straight to the next deadline, making the wait
        observe more elapsed virtual time than its own timeout.
        """
        clock = self.virtual_clock
        if clock is None:
            return
        self._clock_port.advance_clock_to(clock, target, noop_on_wake)

    def sleep_until(self, thread_id: int, deadline: float | None = None, *, duration: float | None = None) -> None:
        """Block *thread_id* until the virtual clock reaches *deadline*.

        With ``duration=`` the deadline is computed under ``_condition`` (the
        lock all clock advances hold), so a concurrent explored-mode advance
        cannot land between the caller's ``now()`` read and the registration
        and instantly expire the sleep.

        Wakes when another thread's scheduling advances the clock past the
        deadline, or — when every live thread is deadline-blocked — by
        autojumping to the earliest pending deadline directly.
        """
        with self._condition:
            if deadline is None:
                if duration is None or self.virtual_clock is None:
                    raise TypeError("sleep_until needs either deadline= or duration= (with a virtual clock)")
                deadline = self.virtual_clock.now() + duration
            self._deadlines.add_sleep(thread_id, deadline, wake_id=None)
            self._condition.notify_all()
            try:
                while self._deadlines.is_sleeping(thread_id):
                    if self._error:
                        return
                    if self._finished:
                        # Schedule/op budget exhausted (wait_for_turn hit the
                        # max_ops cap): no more turns are granted, but the
                        # sleep must still resolve virtually.  Advance to this
                        # sleeper's deadline — firing earlier due deadlines in
                        # order — instead of returning with the clock frozen,
                        # which would silently truncate the sleep and report a
                        # phantom TTL/timeout counterexample.
                        sleep_deadline = self._deadlines.sleep_deadline(thread_id)
                        if sleep_deadline is None:
                            self._error = TimeoutError(
                                f"Deadlock: thread {thread_id} is blocked on a perpetual virtual sleep"
                            )
                            self._condition.notify_all()
                            return
                        self._advance_clock_to(sleep_deadline)
                        self._condition.notify_all()
                        return
                    alive = [t for t in range(self.num_threads) if t not in self._threads_done]
                    blocked = [t for t in alive if self._blocks_clock_progress(t)]
                    if alive and len(blocked) == len(alive):
                        # Every live thread is asleep, in a timed wait, or
                        # spinning on a resource nothing can release before
                        # time passes: only the clock can move.
                        next_deadline = self._deadlines.next_deadline()
                        if next_deadline is not None:
                            self._advance_clock_to(next_deadline)
                            self._condition.notify_all()
                            continue
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._finished or self._error:
                            continue  # dispatch via the loop-top checks
                        self._error = TimeoutError(
                            f"Deadlock: thread {thread_id} sleeping until t={deadline} was never woken"
                        )
                        self._condition.notify_all()
                        return
            finally:
                self._deadlines.cancel_sleep(thread_id)

    def advance_clock_after_finish(self, deadline: float) -> None:
        """Advance the clock to a timed wait's deadline after schedule exhaustion.

        Called by cooperative timed waits (via
        ``_cooperative._finish_virtual_timed_wait``) once ``_finished`` is set:
        no more turns are granted, but a registered virtual timeout must still
        elapse on the virtual clock rather than degrading to a real wait with
        the clock frozen.  Fires earlier due deadlines in order (see
        :meth:`_advance_clock_to`).
        """
        with self._condition:
            self._advance_clock_to(deadline)
            self._condition.notify_all()

    def note_blocking_spin(self, thread_id: int, resource_id: int, waiting: bool, *, timed_wait: bool = False) -> None:
        """Flag *thread_id* as spinning on an untimed cooperative wait.

        Cooperative primitives set the flag after a failed probe and clear it
        once they acquire (or give up); release/set of the resource clears it
        via :meth:`note_spin_release` so the spinner re-probes before being
        counted as blocked by the autojump check in :meth:`sleep_until`.
        ``timed_wait=True`` flags are refused once their deadline has fired
        (see :meth:`VirtualClockPort.note_blocking_spin`).
        """
        self._clock_port.note_blocking_spin(thread_id, resource_id, waiting, timed_wait=timed_wait)

    def note_spin_release(self, resource_id: int) -> None:
        """Clear spin flags for *resource_id* (it may now be acquirable)."""
        self._clock_port.note_spin_release(resource_id)

    def add_timed_wait(
        self,
        thread_id: int,
        deadline: float | None = None,
        *,
        timeout: float | None = None,
        resource: object | None = None,
    ) -> float:
        """Register a virtual deadline for a timed lock acquire.

        With ``timeout=`` the deadline is computed under the scheduler's
        serialising lock (see ``VirtualClockPort.add_timed_wait``); returns it.
        """
        return self._clock_port.add_timed_wait(thread_id, deadline, timeout=timeout, clock=self.virtual_clock)

    def remove_timed_wait(self, thread_id: int) -> None:
        """Deregister a timed-acquire deadline (acquired or gave up)."""
        self._clock_port.remove_timed_wait(thread_id)

    def clear_engine_block(self, thread_id: int) -> None:
        """No-op: the random scheduler has no engine-level blocked state."""

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> list[str]:
        """No-op DPOR-compat stub: random exploration models no SQL row locks.

        ``_acquire_pending_row_locks`` calls this when the OpcodeScheduler is
        registered as the DPOR context (in-transaction SQL under
        ``explore_random``).  Returning an empty list (rather than ``None``)
        records nothing as held, which is correct because nothing is modeled.
        Mirrors ``async_shuffler.OpcodeShuffler.acquire_row_locks`` and keeps
        the signature compatible with ``DporScheduler.acquire_row_locks``.
        """
        return []

    def release_row_locks(self, thread_id: int, resources: object = None) -> None:
        """No-op DPOR-compat stub: random exploration models no SQL row locks.

        Called by ``_release_dpor_row_locks`` (including the SQL error handler).
        Mirrors ``async_shuffler.OpcodeShuffler.release_row_locks``.
        """

    def give_up_timed_wait(self, thread_id: int) -> None:
        """Deregister a timed-acquire deadline on give-up.

        Mirror of :meth:`DporScheduler.give_up_timed_wait`.  The random
        scheduler has no engine-level blocked state (the port's ``unblock`` /
        ``on_give_up`` callbacks are no-ops), so this only drops the deadline;
        there is no false-positive window to close here.
        """
        self._clock_port.give_up_timed_wait(thread_id)

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
        self.errors: dict[int, BaseException] = {}
        self.worker_originated_errors: dict[int, BaseException] = {}
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
        """Replace time.sleep with the cooperative scheduler hook."""
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
        # The time.* patch is owned by the caller's clock_scope, held once
        # across setup/run/invariant, rather than churned here per phase.
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
            if scheduler._clock_diagnostics:
                warn_if_captured_time_reference(frame)
            if recorder is not None:
                recorder.record_from_opcode(tid, frame)
            scheduler.wait_for_turn(tid)
            # Signal "yielded" so make_settrace_callback applies the CPython
            # 3.10-3.11 LocalsToFast workaround (re-reads frame.f_locals to
            # refresh the snapshot).  wait_for_turn can block this thread while
            # another runs, and if that other thread mutates a shared closure
            # cell, the stale f_locals snapshot would otherwise be written back
            # over the new value when this frame resumes — a lost update that
            # shows up as a false positive on lock-protected closures (only on
            # the settrace path; 3.12+ monitoring ignores this return value).
            return True

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
        except BaseException as e:  # noqa: BLE001
            if getattr(self.scheduler, "_error", None) is not e:
                self.worker_originated_errors[thread_id] = e
            self.errors[thread_id] = e
            if isinstance(e, Exception):
                self.scheduler.report_error(e)
            else:
                # Worker BaseExceptions do not cross the thread boundary on
                # their own. Wake peers with an ordinary scheduler error;
                # run() re-raises the original object on the driver thread.
                self.scheduler.report_error(RuntimeError(f"worker {thread_id} terminated with {type(e).__name__}"))

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

        thread_args = [(a, kw) for a, kw in zip(args, kwargs, strict=True)]

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
    clock: ClockMode = "real",
    _virtual_clock: VirtualClock | None = None,
    clock_diagnostics: bool = False,
    _max_ops: int | None = None,
    _worker_errors_as_findings: bool = False,
    _recorded_schedule: list[int] | None = None,
    _workers_entered: list[bool] | None = None,
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
        clock: ``"real"`` (default), ``"virtual"`` (autojump virtual clock),
            or ``"explored"`` (schedule entries landing on a sleeping thread
            advance the clock — the random "maybe advance time" branch).
        _virtual_clock: Internal — the clock instance to drive, so callers
            (``explore_random``) can evaluate invariants against it.

    Returns:
        The state object after execution.
    """
    clock_config = ClockConfig(mode=clock, diagnostics=clock_diagnostics).validate(patch_sleep=patch_sleep)
    clock = clock_config.mode
    if _virtual_clock is not None and clock == "real":
        raise ValueError("_virtual_clock requires clock='virtual' or clock='explored'")
    virtual_clock = _virtual_clock if _virtual_clock is not None else clock_config.new_clock()
    scheduler = OpcodeScheduler(
        schedule,
        len(threads),
        deadlock_timeout=deadlock_timeout,
        trace_recorder=trace_recorder,
        virtual_clock=virtual_clock,
        clock_mode=clock,
        clock_diagnostics=clock_diagnostics,
        max_ops=_max_ops or 0,
    )
    runner = BytecodeShuffler(scheduler, detect_io=detect_io)

    # The clock_scope owns the time.* patch for the whole run (setup + workers);
    # patch_locks BEFORE setup() so any locks created there are cooperative.
    with clock_scope(virtual_clock), runner.patch_scope(patch_sleep=patch_sleep):
        state = _call_sync_setup(setup)

        from frontrun.common import _reject_deferred_sync_result

        def make_thread_func(idx: int, thread_func: Callable[[T], None], thread_state: T) -> Callable[[], None]:
            def thread_wrapper() -> None:
                if _workers_entered is not None:
                    # Pass-certificate evidence: this worker's body was entered.
                    _workers_entered[idx] = True
                result = thread_func(thread_state)
                _reject_deferred_sync_result(result, thread_func)

            return thread_wrapper

        funcs: list[Callable[[], None]] = [make_thread_func(i, t, state) for i, t in enumerate(threads)]
        timed_out = False
        try:
            runner.run(funcs, timeout=timeout)
        except TimeoutError as exc:
            if runner.worker_originated_errors:
                if _worker_errors_as_findings:
                    raise _WorkerExecutionError(exc) from exc
                raise
            if debug:
                print(f"Timed out with {timeout=} on {schedule=}", flush=True)
            timed_out = True
        except DeadlockError:
            raise
        except Exception as exc:
            if scheduler._max_ops_exhausted:
                raise _MaxOpsExhaustedError("run_with_schedule exhausted max_ops before workers completed") from exc
            if _worker_errors_as_findings and runner.errors:
                raise _WorkerExecutionError(exc) from exc
            raise
        finally:
            if _recorded_schedule is not None:
                _recorded_schedule[:] = scheduler.schedule
        # Re-raise DeadlockError so callers (e.g. reproduction logic) can
        # detect that a deadlock occurred during replay.
        if isinstance(scheduler._error, DeadlockError):
            raise scheduler._error
        # Surface a timeout (from runner.run or a scheduler-recorded
        # TimeoutError) instead of returning a state that timed-out daemon
        # threads may still be mutating.  Evaluating an invariant on such a
        # half-finished racing state is meaningless (finding 9d).  Callers in
        # exploration loops catch this and skip the schedule as inconclusive.
        if scheduler._max_ops_exhausted:
            raise _MaxOpsExhaustedError("run_with_schedule exhausted max_ops before workers completed")
        if timed_out or isinstance(scheduler._error, TimeoutError):
            raise TimeoutError(f"run_with_schedule timed out after {timeout}s; worker threads did not complete")
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
    clock: ClockMode = "real",
    clock_diagnostics: bool = False,
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
        max_ops: Maximum randomly sampled schedule-prefix length per attempt.
            If workers need more turns to finish, the scheduler extends the
            prefix deterministically and records those turns in any returned
            counterexample.
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
        patch_sleep: If True (default), ``time.sleep`` yields to the
            scheduler instead of blocking.  Required for ``clock != "real"``.
        serializable_invariant: Check serializability against sequential
            runs.  Cannot be combined with a virtual clock.
        error_on_any_race: Not supported here — requires the DPOR strategy.
        clock: ``"real"`` (default), ``"virtual"`` (autojump virtual clock:
            time reads are virtual, sleeps cost zero wall time and jump the
            clock when nothing else can run), or ``"explored"`` (schedule
            entries landing on a sleeping thread advance the clock, so the
            random sampler also explores early timer firings).  See
            :doc:`/virtual_clock`.
        clock_diagnostics: With a virtual clock, warn when traced worker frames
            hold captured real ``time.*`` clock-read functions.

    Returns:
        InterleavingResult with the outcome.  The ``unique_interleavings``
        field reports how many distinct execution orderings were observed,
        providing a lower bound on exploration coverage.
    """
    _require_frontrun_env("explore_random")
    clock_config = validate_random_exploration_params(
        max_attempts=max_attempts,
        total_timeout=total_timeout,
        error_on_any_race=error_on_any_race,
        clock=clock,
        clock_diagnostics=clock_diagnostics,
        patch_sleep=patch_sleep,
        serializable_invariant=serializable_invariant,
    )
    clock = clock_config.mode
    with trace_filter_scope(trace_packages):
        from frontrun._dpor_core import compute_serializable_baseline_sync

        serial_valid_states, serial_hash_fn = compute_serializable_baseline_sync(setup, threads, serializable_invariant)

        rng = random.Random(seed)
        num_threads = len(threads)
        # Verdict-less accumulator: the pass verdict is only stamped by
        # certify_pass() at the end, from evidence gathered below.
        result = InterleavingResult(property_holds=None, num_explored=0)
        workers_entered = [False] * num_threads
        seen_schedule_hashes: set[int] = set()
        total_deadline = time.monotonic() + total_timeout if total_timeout is not None else None

        for _attempt in range(max_attempts):
            if total_deadline is not None and time.monotonic() > total_deadline:
                # A pre-expired budget explores nothing: the final result is
                # then inconclusive, never a vacuous pass.
                break
            schedule = random_round_robin_schedule(rng, num_threads, max_ops)

            clear_insert_tracker()
            if debug:
                print(f"Running with {schedule=} {threads=}", flush=True)
            recorder = TraceRecorder()
            attempt_clock = clock_config.new_clock()
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
                    clock=clock,
                    _virtual_clock=attempt_clock,
                    clock_diagnostics=clock_diagnostics,
                    _worker_errors_as_findings=True,
                    _recorded_schedule=schedule,
                    _workers_entered=workers_entered,
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
            except _MaxOpsExhaustedError:
                result.num_explored += 1
                seen_schedule_hashes.add(hash(tuple(schedule)))
                result.property_holds = None
                result.unique_interleavings = len(seen_schedule_hashes)
                result.inconclusive_reason = (
                    f"Random exploration exhausted max_ops={max_ops} on attempt {result.num_explored}. "
                    "The completed run is inconclusive because scheduling control ended before the worker trace; "
                    "increase max_ops to claim a passing exploration."
                )
                result.explanation = result.inconclusive_reason
                return result
            except TimeoutError:
                # Python threads cannot be killed safely. The partial state is
                # inconclusive, and survivors may keep mutating globals, so no
                # later attempt in this exploration is trustworthy.
                if debug:
                    print(f"Aborting after timed-out schedule: {schedule}", flush=True)
                result.num_explored += 1
                seen_schedule_hashes.add(hash(tuple(schedule)))
                result.property_holds = None
                result.unique_interleavings = len(seen_schedule_hashes)
                result.inconclusive_reason = (
                    f"Random exploration timed out before workers completed on attempt {result.num_explored}. "
                    "The search is inconclusive because Python threads cannot be killed safely; increase "
                    "timeout_per_run/deadlock_timeout or remove unmanaged blocking from explored workers."
                )
                result.explanation = result.inconclusive_reason
                return result
            except _WorkerExecutionError as worker_err:
                result.num_explored += 1
                seen_schedule_hashes.add(hash(tuple(schedule)))
                result.property_holds = False
                result.counterexample = schedule
                result.unique_interleavings = len(seen_schedule_hashes)
                cause = worker_err.cause
                result.explanation = f"Worker crash in execution {result.num_explored}: {type(cause).__name__}: {cause}"
                return result
            result.num_explored += 1
            seen_schedule_hashes.add(hash(tuple(schedule)))

            if warn_nondeterministic_sql:
                ensure_no_uncaptured_inserts()

            # --- serializable_invariant check ---
            if record_serializability_violation(
                result,
                state=state,
                serial_valid_states=serial_valid_states,
                serial_hash_fn=serial_hash_fn,
                schedule=schedule,
                unique_interleavings=len(seen_schedule_hashes),
            ):
                return result

            with clock_scope(attempt_clock):
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
                            replay_clock = clock_config.new_clock()
                            replay_state = run_with_schedule(
                                schedule,
                                setup,
                                threads,
                                timeout=timeout_per_run,
                                detect_io=detect_io,
                                deadlock_timeout=deadlock_timeout,
                                patch_sleep=patch_sleep,
                                clock=clock,
                                _virtual_clock=replay_clock,
                                clock_diagnostics=clock_diagnostics,
                                # Surface user worker crashes as a distinct wrapper
                                # so the catch below can absorb them without also
                                # swallowing frontrun-internal replay-engine bugs.
                                _worker_errors_as_findings=True,
                            )
                            with clock_scope(replay_clock):
                                replay_failed, _ = check_invariant(invariant, replay_state)
                            if replay_failed:
                                successes += 1
                        except (_WorkerExecutionError, DeadlockError, TimeoutError):
                            # user worker crash / deadlock / timeout during replay
                            # — not a reproduction.  ``_MaxOpsExhaustedError`` is a
                            # ``TimeoutError`` subclass, so it is covered here too.
                            # Frontrun-internal errors are intentionally NOT caught.
                            pass
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
        # Every failure path above returned early, so no failure was found:
        # certify (or honestly refuse to certify) the pass.  Zero executions
        # can only mean the total_timeout budget was already spent.
        return certify_pass(
            result=result,
            evidence=PassEvidence(
                executions=result.num_explored,
                workers_executed=workers_entered,
                vacuous_reason=(
                    f"total_timeout={total_timeout!r}s elapsed before any interleaving completed; "
                    "increase total_timeout or reduce the workload"
                ),
            ),
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
