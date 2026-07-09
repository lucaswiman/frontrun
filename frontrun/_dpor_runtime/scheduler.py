# ruff: noqa: F403, F405
# pyright: reportUnusedClass=false

from __future__ import annotations

import weakref

from frontrun._dpor_core import ReplayEngine as _ReplayEngine
from frontrun._dpor_core import ReplayExecution as _ReplayExecution
from frontrun._dpor_core import (
    RowLockRegistry,
    VirtualClockPort,
    advance_replay_index,
    apply_lock_blocked_override,
    can_autojump,
    extend_replay_schedule,
    format_exact_deadlock_desc,
    noop_on_wake,
    report_clock_sleep_wake,
    retire_actor_if_done,
    sync_clock_actor,
    wake_sync_id,
)
from frontrun._opcode_observer import anchor_label as _anchor_label
from frontrun._virtual_clock import VirtualClock, WakeEvent, real_monotonic

from ._shared import *
from ._shared import _dpor_tls, _get_instructions, _process_opcode
from .preload_bridge import _PreloadBridge

_SENTINEL = object()

# How long every live thread must remain blocked with no pending virtual-clock
# deadline before an exact deadlock is confirmed.  A short confirm window
# tolerates the transient "all blocked" states that occur while threads hand off
# the scheduler turn, without waiting out the full wall-clock deadlock_timeout.
_EXACT_DEADLOCK_CONFIRM_SECONDS = 0.1


class DporScheduler:
    """Controls thread execution at opcode granularity, driven by the DPOR engine.

    Unlike the random OpcodeScheduler in bytecode.py, this scheduler gets
    its scheduling decisions from the Rust DPOR engine.

    Deadlock detection uses a fallback timeout plus instant lock-ordering
    cycle detection via the :class:`~frontrun._deadlock.WaitForGraph`.
    """

    def __init__(
        self,
        engine: PyDporEngine,
        execution: PyExecution,
        num_threads: int,
        engine_lock: threading.Lock | None = None,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        preload_bridge: _PreloadBridge | None = None,
        detect_io: bool = False,
        stable_ids: StableObjectIds | None = None,
        switch_point_collector: list[Any] | None = None,
        track_dunder_dict_accesses: bool = False,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_actor_id: int | None = None,
        clock_diagnostics: bool = False,
    ) -> None:
        self.engine = engine
        self.execution = execution
        self.num_threads = num_threads
        # Virtual clock (ideas/virtual_clock.md).  When active, the engine was
        # constructed with one extra thread — the *clock actor* — whose only
        # transition is "advance the clock to the next deadline and wake its
        # sleepers".  In "virtual" (autojump) mode the actor is enabled only
        # when no real thread is runnable; in "explored" mode it is enabled
        # whenever a deadline is pending, so the engine explores clock-step
        # orderings like any other interleaving choice.
        self.virtual_clock = virtual_clock
        self._clock_mode = clock_mode
        self._clock_actor_id = clock_actor_id
        self._clock_diagnostics = clock_diagnostics
        # Replay only: clock-actor schedule entries reached before any
        # deadline was registered (schedule drift).  The owed advance is
        # performed at the next deadline registration instead of being lost
        # (losing it costs a full deadlock_timeout per reproduction attempt).
        self._pending_clock_advances = 0
        self._exact_deadlock_candidate_at: float | None = None
        self.deadlock_timeout = deadlock_timeout
        self.trace_recorder = trace_recorder
        self._preload_bridge = preload_bridge
        self._detect_io = detect_io
        self._stable_ids = stable_ids if stable_ids is not None else StableObjectIds()
        self._switch_point_collector = switch_point_collector
        self._track_dunder_dict_accesses = track_dunder_dict_accesses
        self._step_event_collector: dict[int, Any] | None = {} if switch_point_collector is not None else None
        self._lock_event_collector: list[Any] | None = [] if switch_point_collector is not None else None
        # Captured at the moment the first error fires (schedule_trace length then).
        # Steps at/after this index are teardown artifacts and should not be rendered.
        self._deadlock_at: int | None = None
        # On free-threaded Python, PyO3 &mut self borrows are non-blocking
        # (try-or-panic).  A single engine_lock serialises ALL calls to the
        # engine and execution objects across worker threads, the sync
        # reporter, and the main frontrun.explore loop.
        self._engine_lock: threading.Lock = engine_lock if engine_lock is not None else real_lock()
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        # Shared virtual-clock protocol (single source of deadline membership +
        # blocking-spin flags).  DPOR wires the engine block/unblock + clock-actor
        # sync into it; the callbacks run under the engine_lock the port holds.
        # ``_deadlines`` / ``_spin_waiters`` alias the port's state so the rest of
        # the scheduler (sleep_until, _schedule_next, mark_done) is unchanged.
        self._clock_port = VirtualClockPort(
            condition=self._condition,
            engine_lock=self._engine_lock,
            block=self._port_engine_block,
            unblock=self._port_engine_unblock,
            sync=self._sync_clock_actor_locked,
            on_give_up=self._scrub_lock_waiters,
            on_added=self._perform_owed_clock_advance,
        )
        self._deadlines = self._clock_port.coordinator
        self._spin_waiters = self._clock_port.spin_waiters
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self._current_thread: int | None = None
        # Weak references: a baseline thread that exits and is GC'd drops out,
        # so a new external thread that reuses its id() cannot be misclassified
        # as baseline in _has_live_external_threads (id reuse would otherwise
        # wrongly re-enable exact-deadlock detection).
        self._baseline_thread_keys: weakref.WeakSet[threading.Thread] = weakref.WeakSet(
            t for t in threading.enumerate() if t.is_alive()
        )
        self._worker_thread_keys: set[int] = set()

        # Shadow stacks are per-thread (each thread only accesses its own),
        # stored in thread-local storage. This avoids cross-thread access
        # entirely, which is critical for free-threaded builds.
        # Format: _dpor_tls._shadow_stacks = {frame_id: ShadowStack}

        # Tracks which threads are waiting for which locks (lock_id → {thread_ids}).
        # Used to block threads in the DPOR execution when they're spinning
        # on a cooperative lock, and unblock them when the lock is released.
        self._lock_waiters: dict[int, set[int]] = {}
        # Per-thread deferred I/O buffers. Lists are shared with thread-local
        # storage so other threads can flush deferred I/O when they reach a
        # real competing I/O boundary.
        self._pending_io_by_thread: dict[int, list[tuple[int, str, bool]]] = {}
        # Per-thread lock nesting depth mirrored from TLS for cross-thread
        # deferred-I/O flush decisions.
        self._lock_depth_by_thread: dict[int, int] = {}

        # IO trace: records (thread_id, resource_id) in IO execution order.
        # Populated by after_io() under the condition lock.  See defect #16.
        self._io_trace: list[tuple[int, str]] = []
        # Access trace for access-anchored replay (defect #20): records
        # (thread_id, label, kind) for every anchorable shared-memory access,
        # in execution order.  ``_access_key_labels`` maps engine object keys
        # to those labels so that, on failure, the racing objects reported by
        # ``engine.pending_races()`` can be resolved to run-stable labels.
        self._access_trace: list[tuple[int, str, str]] = []
        self._access_key_labels: dict[int, str] = {}
        # Explicit Python-level I/O boundary currently in progress. While set,
        # the owning thread keeps the scheduler turn until after_io() runs.
        self._active_io_thread: int | None = None
        self._next_thread_after_io: int | None = None
        # Like the I/O path, cooperative lock/semaphore retries need to keep
        # the scheduler turn until the real non-blocking probe completes.
        # Otherwise free-threaded builds turn the probe into an OS-level race
        # between multiple awakened waiters.
        self._active_sync_thread: int | None = None
        self._next_thread_after_sync: int | None = None

        # Maps iterator id → original container object. When GET_ITER creates
        # an iterator from a mutable container, we record the mapping so that
        # FOR_ITER can report reads on the underlying container.
        self._iter_to_container: dict[int, Any] = {}

        # Row-lock registry: resource_id → thread_id holding the lock.
        # SELECT FOR UPDATE is exclusive — only one thread can hold it at a time.
        # State and _row_lock_int_id() live in the shared RowLockRegistry;
        # alias dicts into this namespace so the rest of the class is unchanged.
        self._row_lock_registry = RowLockRegistry()
        self._active_row_locks: dict[str, int] = self._row_lock_registry._active_row_locks
        # Reverse index: thread_id → set of resource_ids held by that thread.
        # Avoids O(n) scan in _release_row_locks_unlocked.
        self._task_row_locks: dict[int, set[str]] = self._row_lock_registry._task_row_locks
        # For backward compatibility, keep a _thread_row_locks alias pointing
        # to the same dict (was renamed to _task_row_locks to match async).
        self._thread_row_locks: dict[int, set[str]] = self._row_lock_registry._task_row_locks

        # Stable integer IDs for row-lock resources (for WaitForGraph nodes).
        # Managed by self._row_lock_registry._row_lock_int_id(); no direct access needed.
        self._row_lock_ids: dict[str, int] = self._row_lock_registry._row_lock_ids

        # Threads currently blocked waiting for a DPOR row lock.
        # Maps blocked_thread_id → holder_thread_id.  Used by
        # _schedule_next() to skip blocked threads and schedule their
        # holders instead, preventing the scheduler from cycling
        # indefinitely between a blocked thread and its holder.
        self._row_lock_blocked: dict[int, int] = {}

        # Last path_id snapshot from _schedule_next, used to attribute
        # lock events to the correct scheduling step on free-threaded Python.
        self._last_scheduled_path_id: int | None = None

        # The clock actor starts blocked: it only becomes runnable when a
        # deadline is pending (explored) or when everything else idles
        # (autojump; see _schedule_next).
        if self._clock_actor_id is not None:
            with self._engine_lock:
                self.execution.block_thread(self._clock_actor_id)

        # Request the first scheduling decision
        self._current_thread = self._schedule_next()

    # ------------------------------------------------------------------
    # Virtual clock
    # ------------------------------------------------------------------

    def _has_pending_deadlines(self) -> bool:
        return self._deadlines.has_pending()

    def _condition_wait_timeout(self) -> float:
        if self.virtual_clock is None or self._exact_deadlock_candidate_at is None:
            return self.deadlock_timeout
        remaining = _EXACT_DEADLOCK_CONFIRM_SECONDS - (real_monotonic() - self._exact_deadlock_candidate_at)
        return max(0.001, min(self.deadlock_timeout, remaining))

    def _reschedule_done_current_unlocked(self) -> bool:
        """Advance immediately when the current thread has already finished."""
        if self._current_thread not in self._threads_done:
            return False
        next_thread = self._schedule_next()
        self._current_thread = next_thread
        if next_thread is None and len(self._threads_done) >= self.num_threads:
            self._finished = True
        self._condition.notify_all()
        return True

    def register_worker_thread(self) -> None:
        key = id(threading.current_thread())
        with self._condition:
            self._worker_thread_keys.add(key)

    def unregister_worker_thread(self) -> None:
        key = id(threading.current_thread())
        with self._condition:
            self._worker_thread_keys.discard(key)

    def _has_live_external_threads(self) -> bool:
        # WeakSet auto-drops dead baseline threads, so resolving ids here never
        # subtracts a stale id that a new external thread might have reused.
        baseline_ids = {id(t) for t in self._baseline_thread_keys}
        current = {id(t) for t in threading.enumerate() if t.is_alive()}
        external = current - baseline_ids - self._worker_thread_keys
        return bool(external)

    def _sync_clock_actor_locked(self) -> None:
        """Keep the clock actor's enabledness in step with pending deadlines.

        Caller must hold ``_engine_lock``.  In "explored" mode the actor is
        runnable whenever a deadline is pending; in autojump mode it stays
        blocked (``_schedule_next`` enables it transiently when idle).
        """
        sync_clock_actor(self.execution, self._clock_actor_id, self._clock_mode, self._has_pending_deadlines())

    def _advance_virtual_clock_locked(self) -> None:
        """Perform one clock-actor step: jump to the earliest deadline.

        Caller must hold ``_engine_lock``.  Wakes every thread whose deadline
        is reached (equal deadlines wake in deterministic (deadline, thread id)
        order), reporting a wake happens-before edge for each sleeper.
        """
        clock = self.virtual_clock
        if clock is None:
            return
        # The shared port pops each due actor's spin flag; the on-wake callback
        # closes the engine/HB side.  A clock-actor pick can arrive after all
        # deadlines were canceled, in which case this is a no-op and the
        # trailing sync re-blocks the actor.  Replay accounting for no-op clock
        # actor entries is tracked in the virtual-clock hardening roadmap.
        self._clock_port.advance_clock_to(clock, None, self._on_clock_wake)
        self._sync_clock_actor_locked()

    def _port_engine_block(self, thread_id: int) -> None:
        """Mark *thread_id* engine-blocked (port callback; caller holds ``_engine_lock``)."""
        self.execution.block_thread(thread_id)

    def _port_engine_unblock(self, thread_id: int) -> None:
        """Clear *thread_id*'s engine block (port callback; caller holds ``_engine_lock``)."""
        self.execution.unblock_thread(thread_id)

    def _on_clock_wake(self, event: WakeEvent) -> None:
        """Per-due-deadline engine/HB side of a clock advance (caller holds ``_engine_lock``)."""
        tid = event.actor_id
        if event.kind == "sleep":
            report_clock_sleep_wake(
                self.engine.report_sync,
                self.execution,
                self._clock_actor_id,
                event,
                self._last_scheduled_path_id,
            )
        elif event.kind == "timeout":
            # Timed-wait wakes deliberately carry no happens-before edge: the
            # waiter re-reports lock_wait (re-blocking itself) before it can
            # observe expiry, and the give-up path ends in clear_engine_block.
            self.execution.unblock_thread(tid)

    def _scrub_lock_waiters(self, thread_id: int) -> None:
        """Remove *thread_id* from every lock-waiter set (give-up cleanup).

        A thread waits on at most one resource at a time, so scrubbing it from
        every waiter set is equivalent to knowing the lock id.
        """
        for waiters in self._lock_waiters.values():
            waiters.discard(thread_id)

    def _perform_owed_clock_advance(self) -> None:
        """Perform a replay-owed clock advance after a deadline registration."""
        if self._pending_clock_advances > 0:
            with self._condition:
                if self._pending_clock_advances > 0:
                    self._pending_clock_advances -= 1
                    self._replay_advance_clock_to()
                    self._condition.notify_all()

    def add_timed_wait(self, thread_id: int, deadline: float) -> None:
        """Register a virtual deadline for a timed lock acquire."""
        self._clock_port.add_timed_wait(thread_id, deadline)

    def remove_timed_wait(self, thread_id: int) -> None:
        """Deregister a timed-acquire deadline (acquired or gave up)."""
        self._clock_port.remove_timed_wait(thread_id)

    def clear_engine_block(self, thread_id: int) -> None:
        """Unblock *thread_id* after a timed acquire gives up.

        The waiter was marked blocked by its last ``lock_wait`` sync event;
        without this the engine would never schedule it again.
        """
        with self._condition:
            with self._engine_lock:
                self.execution.unblock_thread(thread_id)
            # A thread waits on at most one resource at a time, so scrubbing
            # it from every waiter set is equivalent to knowing the lock id.
            for waiters in self._lock_waiters.values():
                waiters.discard(thread_id)
            self._condition.notify_all()

    def give_up_timed_wait(self, thread_id: int) -> None:
        """Atomically unblock a timed-acquire waiter and drop its deadline.

        Delegates to the shared port, which unblocks *before* dropping the
        deadline under a single lock hold (``on_give_up`` = ``_scrub_lock_waiters``
        runs the lock-waiter cleanup).  Doing the two separately opened a window
        in which the waiter was engine-blocked with no pending deadline, arming a
        spurious exact-deadlock ``DeadlockError``.
        """
        self._clock_port.give_up_timed_wait(thread_id)

    def note_blocking_spin(self, thread_id: int, resource_id: int, waiting: bool, *, timed_wait: bool = False) -> None:
        """Mark cooperative Condition/Queue polling as engine-blocked."""
        self._clock_port.note_blocking_spin(thread_id, resource_id, waiting, timed_wait=timed_wait)

    def note_spin_release(self, resource_id: int) -> None:
        """Wake spin waiters for a cooperative resource that changed state."""
        self._clock_port.note_spin_release(resource_id)

    def sleep_until(self, thread_id: int, deadline: float) -> None:
        """Block *thread_id* until the virtual clock reaches *deadline*.

        The thread registers its deadline, is marked blocked in the engine,
        and releases the scheduler turn.  It resumes only after (a) a clock
        actor step advanced the clock past the deadline and (b) the engine
        scheduled it again.  On resume it reports the ``lock_acquire`` half
        of the wake happens-before edge.
        """
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            prev_machinery = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            try:
                if self._finished or self._error:
                    return
                with self._engine_lock:
                    self._deadlines.add_sleep(thread_id, deadline, wake_sync_id(thread_id))
                    self.execution.block_thread(thread_id)
                    self._sync_clock_actor_locked()
                if self._pending_clock_advances > 0:
                    # Replay owed us an actor step that arrived before this
                    # registration (drift): perform it now.
                    self._pending_clock_advances -= 1
                    self._replay_advance_clock_to()
                if self._current_thread == thread_id:
                    next_thread = self._schedule_next()
                    self._current_thread = next_thread
                    if next_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                self._condition.notify_all()

                def _abort_sleep() -> None:
                    with self._engine_lock:
                        self._deadlines.cancel_sleep(thread_id)
                        self.execution.unblock_thread(thread_id)
                        self._sync_clock_actor_locked()

                # Phase 1: wait for the clock advance that clears our sleep
                # deadline (and unblocks us in the engine).
                while self._deadlines.is_sleeping(thread_id):
                    if self._finished or self._error:
                        _abort_sleep()
                        return
                    if self._replay_sleep_self_wake(thread_id):
                        continue
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._reschedule_done_current_unlocked():
                            continue
                        self._error = TimeoutError(
                            f"DPOR sleep deadlock: thread {thread_id} sleeping until t={deadline}, "
                            f"current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        _abort_sleep()
                        return
                # Phase 2: woken — wait until the engine schedules us again.
                while self._current_thread != thread_id:
                    if self._finished or self._error:
                        return
                    if self._reschedule_done_current_unlocked():
                        continue
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        self._error = TimeoutError(
                            f"DPOR sleep-wake deadlock: thread {thread_id} woke at t={deadline} "
                            f"but was never rescheduled; current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return
                # Close the wake happens-before edge (clock advance → resume).
                report_sync = getattr(self.engine, "report_sync", None)
                if report_sync is not None:
                    with self._engine_lock:
                        report_sync(self.execution, thread_id, "lock_acquire", wake_sync_id(thread_id), None)
            finally:
                _scheduler_tls._in_dpor_machinery = prev_machinery

    def _try_autojump_locked(self) -> bool:
        """Enable the clock actor when everything is blocked but timers pend.

        Caller holds ``_engine_lock``.  Returns True (``_schedule_next`` should
        re-poll) when the autojump actor was enabled: with no runnable thread and
        a pending deadline, advancing the clock is the only schedulable
        transition, which is exactly when it *must* happen.
        """
        if can_autojump(self.virtual_clock, self._clock_actor_id, self._has_pending_deadlines()):
            self.execution.unblock_thread(self._clock_actor_id)
            self._exact_deadlock_candidate_at = None
            return True
        return False

    def _check_exact_deadlock_locked(self) -> None:
        """Record an exact deadlock when it is confirmed (caller holds ``_engine_lock``).

        Every live thread is blocked and no deadline is pending, so no transition
        can ever become enabled.  Confirmed only after a short window (tolerating
        the transient all-blocked states during turn hand-off) and only when no
        live external thread could still unblock a waiter — then reported
        immediately instead of via the wall-clock fallback timeout.
        """
        if not (
            self.virtual_clock is not None
            and self._error is None
            and not self._finished
            and len(self._threads_done) < self.num_threads
        ):
            return
        if self._has_live_external_threads():
            self._exact_deadlock_candidate_at = None
            return
        if self._exact_deadlock_candidate_at is None:
            self._exact_deadlock_candidate_at = real_monotonic()
            return
        if real_monotonic() - self._exact_deadlock_candidate_at < _EXACT_DEADLOCK_CONFIRM_SECONDS:
            return
        desc = format_exact_deadlock_desc(
            noun="threads",
            sleepers=self._deadlines.sleeping_actors(),
            spin_waiters=sorted(self._spin_waiters),
            done=sorted(self._threads_done),
        )
        self._error = DeadlockError(f"Deadlock detected by virtual clock: {desc}", desc)

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which thread to run next.

        If the engine selects a thread that is blocked on a DPOR row lock,
        override the decision and schedule the lock holder instead.  This
        prevents the scheduler from cycling between a blocked thread and
        its holder (defect #6).

        Clock-actor steps are handled inline: when the engine schedules the
        actor, the clock advances to the earliest deadline and the loop asks
        the engine again.  In autojump mode the actor is enabled only when no
        real thread is runnable, which is exactly when the clock *must*
        advance for anything to happen.

        Also snapshots ``engine.path_position`` under the engine lock so
        that ``report_and_wait`` can attribute subsequent lock events to
        the correct scheduling step (see ``_last_scheduled_path_id``).
        """
        with self._engine_lock:
            while True:
                runnable = self.execution.runnable_threads()
                if not runnable:
                    if self._try_autojump_locked():
                        continue
                    self._last_scheduled_path_id = None
                    self._check_exact_deadlock_locked()
                    return None

                self._exact_deadlock_candidate_at = None
                scheduled = self.engine.schedule(self.execution)
                # Snapshot path position under engine_lock. On free-threaded
                # Python, another thread may call schedule() concurrently
                # after we release the lock, advancing path.pos.  The saved
                # position ensures _sync_reporter attributes lock events to
                # the correct step.
                _pp = getattr(self.engine, "path_position", None)
                self._last_scheduled_path_id = _pp - 1 if _pp is not None else None
                if scheduled is not None and scheduled == self._clock_actor_id:
                    self._advance_virtual_clock_locked()
                    continue
                # Shared with the async scheduler: redirect to the lock holder when
                # the engine picks a row-lock-blocked thread (defect #6), or drop a
                # stale entry whose holder has finished.
                return apply_lock_blocked_override(scheduled, self._row_lock_blocked, self._threads_done)

    def wait_for_turn(self, thread_id: int) -> bool:
        """Block until it's this thread's turn. Returns False when done."""
        return self._report_and_wait(None, thread_id)

    def before_sync_retry(self, thread_id: int) -> bool:
        """Wait for a turn and keep it through one external sync probe."""
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return False
                    if self._reschedule_done_current_unlocked():
                        continue

                    if self._active_sync_thread is not None and self._active_sync_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        self._flush_other_pending_io_for_current_io_unlocked(thread_id)
                        self._flush_pending_io_for_unlocked(thread_id)
                        next_thread = self._schedule_next()
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        self._active_sync_thread = thread_id
                        self._next_thread_after_sync = next_thread
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return True

                    if not self._condition.wait(timeout=self._condition_wait_timeout()):
                        if self.virtual_clock is not None and self._current_thread is None:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        if self._reschedule_done_current_unlocked():
                            continue
                        self._error = TimeoutError(
                            f"DPOR sync deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return False
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_sync_retry(self, thread_id: int) -> None:
        with self._condition:
            if self._active_sync_thread == thread_id:
                self._active_sync_thread = None
                self._current_thread = self._next_thread_after_sync
                self._next_thread_after_sync = None
                if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
                self._condition.notify_all()

    def _all_other_live_threads_blocked_by_current(self, thread_id: int) -> bool:
        from frontrun._deadlock import get_wait_for_graph

        graph = get_wait_for_graph()
        if graph is None:
            return False
        live_other_threads = {
            tid for tid in range(self.num_threads) if tid != thread_id and tid not in self._threads_done
        }
        if not live_other_threads:
            return True
        blocked_threads = graph.reverse_reachable_threads_from(thread_id)
        return live_other_threads.issubset(blocked_threads)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Report accesses for an opcode and wait for this thread's turn.

        Combines ``_process_opcode`` and the wait-for-turn logic under a
        single lock acquisition so that ``engine.report_access()`` and
        ``engine.schedule()`` can never be called concurrently.  This is
        critical on free-threaded Python (3.13t/3.14t) where there is no
        GIL to serialise PyO3 ``&mut self`` borrows.
        """
        return self._report_and_wait(frame, thread_id)

    def _flush_pending_io_for_unlocked(self, flush_thread_id: int, *, allow_inside_lock: bool = False) -> None:
        pending_io = self._pending_io_by_thread.get(flush_thread_id)
        if not pending_io:
            return
        inside_lock = self._lock_depth_by_thread.get(flush_thread_id, 0) > 0
        if not allow_inside_lock and inside_lock:
            # Even though we defer most events, synced events
            # (Python-level Redis/SQL) can be flushed now — they use
            # dpor_vv which respects lock HB, so recording them at
            # the in-lock path_id is correct and avoids position
            # misattribution from deferred flushing.
            engine = self.engine
            execution = self.execution
            remaining: list[tuple[int, str, bool]] = []
            for obj_key, io_kind, synced in pending_io:
                if synced:
                    with self._engine_lock:
                        engine.report_synced_io_access(execution, flush_thread_id, obj_key, io_kind)
                else:
                    remaining.append((obj_key, io_kind, synced))
            pending_io.clear()
            pending_io.extend(remaining)
            return
        engine = self.engine
        execution = self.execution
        for obj_key, io_kind, synced in pending_io:
            with self._engine_lock:
                if synced:
                    engine.report_synced_io_access(execution, flush_thread_id, obj_key, io_kind)
                else:
                    engine.report_io_access(execution, flush_thread_id, obj_key, io_kind)
        pending_io.clear()

    def _flush_other_pending_io_for_current_io_unlocked(self, thread_id: int) -> None:
        current_pending = self._pending_io_by_thread.get(thread_id)
        if not current_pending:
            return
        for other_thread_id, pending_io in list(self._pending_io_by_thread.items()):
            if other_thread_id == thread_id or not pending_io:
                continue
            # Another thread reached a real I/O boundary, so any
            # deferred I/O from this thread is now part of a real
            # race window and must become visible to the engine.
            self._flush_pending_io_for_unlocked(other_thread_id, allow_inside_lock=True)

    def _report_and_wait(self, frame: Any | None, thread_id: int) -> bool:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            # Set reentrancy guard so GC-triggered __del__ (e.g.,
            # redis.Redis.__del__) won't re-enter the scheduler.
            # See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:
                # Merge LD_PRELOAD I/O events (C-level send/recv from e.g.
                # psycopg2) into the thread's pending_io list.  The preload
                # bridge buffers events from the pipe reader thread, keyed by
                # DPOR thread ID.
                _pending_io: list[tuple[int, str, bool]] | None = getattr(_dpor_tls, "pending_io", None)
                if self._preload_bridge is not None:
                    _preload_events = self._preload_bridge.drain(thread_id)
                    # Drop events for known SQL/Redis endpoints that raced past
                    # the listener() check due to async pipe delivery.
                    # Also drop socket events from permanently-suppressed SQL
                    # threads (belt-and-suspenders for connect-time race).
                    _is_sql_tid = is_tid_suppressed(threading.get_native_id())
                    if _preload_events:
                        _preload_events = [
                            ev
                            for ev in _preload_events
                            if not is_sql_endpoint_suppressed(ev[2])
                            and not (_is_sql_tid and ev[2].startswith("socket:"))
                        ]
                    if _preload_events:
                        # Record into trace for human-readable output.  These events
                        # come from C extensions (e.g. libpq) with no Python frame.
                        _recorder = self.trace_recorder
                        if _recorder is not None:
                            for _, _kind, _resource_id, _detail, _call_chain in _preload_events:
                                _recorder.record_io(
                                    thread_id,
                                    _resource_id,
                                    _kind,
                                    call_chain=_call_chain,
                                    detail=_detail,
                                )
                        # LD_PRELOAD events inside a Python lock should respect lock HB
                        # (synced=True uses dpor_vv).  Events outside locks use io_vv
                        # for TOCTOU detection.
                        _inside_lock = self._lock_depth_by_thread.get(thread_id, 0) > 0
                        _io_pairs = [(_key, _kind, _inside_lock) for _key, _kind, _, _, _ in _preload_events]
                        if _pending_io is not None:
                            _pending_io.extend(_io_pairs)
                        else:
                            _dpor_tls.pending_io = _io_pairs
                            _pending_io = _io_pairs
                # NOTE: We intentionally do NOT skip scheduling inside explicit
                # SQL transactions.  SQL atomicity is handled separately by
                # _tx_buffer in _sql_cursor.py (SQL events are buffered during
                # BEGIN...COMMIT and flushed atomically at COMMIT).  Non-SQL
                # shared state (Python objects) modified inside a transaction
                # body must still be interleaved by DPOR to find races.

                while True:
                    if self._finished or self._error:
                        return False
                    if self._reschedule_done_current_unlocked():
                        continue
                    if self._current_thread == thread_id:
                        current_pending = self._pending_io_by_thread.get(thread_id)
                        if (
                            frame is None
                            and current_pending
                            and self._lock_depth_by_thread.get(thread_id, 0) > 0
                            and self._all_other_live_threads_blocked_by_current(thread_id)
                        ):
                            # Flush synced IO events even when skipping the
                            # scheduling point — they use dpor_vv and should
                            # be recorded at the correct in-lock position.
                            self._flush_pending_io_for_unlocked(thread_id)
                            return True
                        self._flush_other_pending_io_for_current_io_unlocked(thread_id)
                        # Flush deferred I/O only once this thread actually owns
                        # the current DPOR step. On free-threaded Python a thread
                        # can reach report_and_wait while another thread still owns
                        # the step; flushing earlier stamps the access onto the
                        # wrong path_id and can hide the wakeup tree insertion point.
                        self._flush_pending_io_for_unlocked(thread_id)
                        # Process opcode accesses only when it's our turn.
                        # Deferring this until the thread is scheduled ensures
                        # that accesses are recorded at the correct path_id
                        # (after any intervening operations by other threads).
                        # Without this, a preempted thread's accesses land at the
                        # preemption branch where the other thread is Active,
                        # making wakeup tree insertions at that position impossible.
                        # Save frame info before _process_opcode clears it,
                        # in case we need to record a switch/step point.
                        _switch_frame = frame
                        # Snapshot shadow stack before _process_opcode pops values.
                        # For STORE_ATTR: stack is [..., value, obj] — we want value (TOS1).
                        # For LOAD_ATTR: stack is [..., obj] — value will be on TOS after.
                        _pre_opcode_stack = None
                        if self._step_event_collector is not None and frame is not None:
                            _pre_stacks = getattr(_dpor_tls, "_shadow_stacks", None)
                            if _pre_stacks:
                                _pre_shadow = _pre_stacks.get(id(frame))
                                if _pre_shadow and _pre_shadow.stack:
                                    _pre_opcode_stack = list(_pre_shadow.stack[-3:])  # last 3 elements
                        if frame is not None:
                            _process_opcode(frame, self, thread_id)
                            frame = None  # only process once
                        # Record step event for the report
                        if self._switch_point_collector is not None and _switch_frame is not None:
                            self._capture_step_event(_switch_frame, thread_id, _pre_opcode_stack)
                        # It's our turn. After executing one opcode, schedule next.
                        next_thread = self._schedule_next()
                        # _schedule_next saves the path position in
                        # self._last_scheduled_path_id (under engine_lock).
                        # Copy it to TLS so _sync_reporter can attribute lock
                        # events to this thread's scheduling step, not a later
                        # step advanced by another thread on free-threaded Python.
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        # Record switch point if thread changes and collector is active
                        if (
                            self._switch_point_collector is not None
                            and next_thread is not None
                            and next_thread != thread_id
                            and _switch_frame is not None
                        ):
                            self._capture_switch_point(_switch_frame, thread_id, next_thread)
                        self._current_thread = next_thread
                        if next_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                        self._condition.notify_all()
                        return True

                    # Wait for our turn (fallback timeout for C-blocked threads)
                    if not self._condition.wait(timeout=self._condition_wait_timeout()):
                        if self._reschedule_done_current_unlocked():
                            continue
                        if self.virtual_clock is not None and self._current_thread is None:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return False
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def before_io(self, thread_id: int, resource_id: str) -> None:
        """Enter an explicit Python-level I/O boundary.

        Unlike wait_for_turn(), this does not release the scheduler turn
        immediately. The current thread keeps running until after_io()
        records completion and hands off to the precomputed next thread.
        """
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return

                    if self._active_io_thread is not None and self._active_io_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        current_pending = self._pending_io_by_thread.get(thread_id)
                        if (
                            current_pending
                            and self._lock_depth_by_thread.get(thread_id, 0) > 0
                            and self._all_other_live_threads_blocked_by_current(thread_id)
                        ):
                            # Keep ownership of the turn and avoid forcing a
                            # preemption inside a lock-held deadlock-avoidance path.
                            self._flush_pending_io_for_unlocked(thread_id)
                            self._active_io_thread = thread_id
                            self._next_thread_after_io = thread_id
                            self._condition.notify_all()
                            return

                        self._flush_other_pending_io_for_current_io_unlocked(thread_id)
                        self._flush_pending_io_for_unlocked(thread_id)
                        next_thread = self._schedule_next()
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        self._active_io_thread = thread_id
                        self._next_thread_after_io = next_thread
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._reschedule_done_current_unlocked():
                            continue
                        if self.virtual_clock is not None and self._schedule_idle_current_unlocked():
                            # Idle under a virtual clock (e.g. after a timed-wait
                            # give-up): reschedule instead of stalling.
                            continue
                        self._error = TimeoutError(
                            f"DPOR I/O deadlock before {resource_id}: waiting for thread {thread_id}, "
                            f"current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_io(self, thread_id: int, resource_id: str) -> None:
        """Called immediately after an IO command completes.

        During exploration, records the IO event and releases the turn to
        the next thread chosen at before_io().
        """
        with self._condition:
            self._io_trace.append((thread_id, resource_id))
            if self._active_io_thread == thread_id:
                self._active_io_thread = None
                self._current_thread = self._next_thread_after_io
                self._next_thread_after_io = None
                if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
                self._condition.notify_all()

    def mark_done(self, thread_id: int) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            # Set the reentrancy guard for this condition-holding critical
            # section: mark_done runs at thread teardown, exactly when GC
            # __del__ chains (e.g. redis.Redis.__del__) fire.  Without it, a
            # __del__ that releases a cooperative lock would call back into the
            # scheduler and try to re-acquire the non-reentrant self._condition
            # on this thread → self-deadlock.  See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:
                self._threads_done.add(thread_id)
                with self._engine_lock:
                    self.execution.finish_thread(thread_id)
                    # Drop any stale virtual-clock deadlines (safety net) and,
                    # once every real thread finished, retire the clock actor
                    # so the engine sees the execution as complete.
                    if self.virtual_clock is not None:
                        self._deadlines.cancel(thread_id)
                        self._spin_waiters.pop(thread_id, None)
                        self._sync_clock_actor_locked()
                    retire_actor_if_done(
                        self.execution, self._clock_actor_id, len(self._threads_done), self.num_threads
                    )
                # Release any row locks the thread may still hold (safety net).
                # _release_row_locks_unlocked avoids re-acquiring self._condition.
                self._release_row_locks_unlocked(thread_id)
                # Clean up stale row-lock-blocked entry (safety net).
                self._row_lock_blocked.pop(thread_id, None)
                # If the done thread was the current one, schedule next
                if self._current_thread == thread_id:
                    next_thread = self._schedule_next()
                    self._current_thread = next_thread
                    if next_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                self._condition.notify_all()
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def on_traced_access(self, thread_id: int, obj: Any, name: Any, kind: str, key: int) -> None:
        """Access-sink callback: record an anchorable access (exploration).

        Overridden by :class:`_ReplayDporScheduler` to *gate* accesses to
        racing objects on the recorded order instead (defect #20).
        """
        label = _anchor_label(obj, name)
        if label is None:
            return
        self._access_trace.append((thread_id, label, kind))
        if key not in self._access_key_labels:
            self._access_key_labels[key] = label

    def racing_access_schedule(self, raced_object_keys: set[int]) -> list[tuple[int, str, str]] | None:
        """Anchor schedule for access-anchored replay (defect #20).

        Filters the recorded access trace to the labels of the objects in
        *raced_object_keys* and collapses consecutive duplicate entries
        (one opcode can report the same access at several scheduling
        points; the replay gate re-matches such runs against the last
        consumed anchor).  Returns ``None`` when nothing anchorable raced.
        """
        watched = {self._access_key_labels[k] for k in raced_object_keys if k in self._access_key_labels}
        if not watched:
            return None
        anchors: list[tuple[int, str, str]] = []
        for entry in self._access_trace:
            if entry[1] in watched and (not anchors or anchors[-1] != entry):
                anchors.append(entry)
        return anchors or None

    def report_error(self, error: Exception) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def _capture_switch_point(self, frame: Any, from_thread: int, to_thread: int) -> None:
        """Capture a SwitchPoint when the scheduler switches threads."""
        from frontrun._report import SwitchPoint, _safe_repr

        code = frame.f_code
        lineno = frame.f_lineno
        instr = _get_instructions(code).get(frame.f_lasti)
        opcode = instr.opname if instr else ""
        source_line = linecache.getline(code.co_filename, lineno).strip()

        # Snapshot shadow stack top 5
        shadow_top5: list[str] = []
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks:
            shadow = stacks.get(id(frame))
            if shadow and shadow.stack:
                shadow_top5.extend(_safe_repr(shadow.stack[-(i + 1)]) for i in range(min(5, len(shadow.stack))))

        # Get access info from the most recent trace event
        access_type: str | None = None
        attr_name: str | None = None
        obj_type_name: str | None = None
        if self.trace_recorder and self.trace_recorder.events:
            last_ev = self.trace_recorder.events[-1]
            if last_ev.thread_id == from_thread:
                access_type = last_ev.access_type
                attr_name = last_ev.attr_name
                obj_type_name = last_ev.obj_type_name

        # schedule_trace length gives current position (after schedule() appended)
        with self._engine_lock:
            schedule_len = len(self.execution.schedule_trace)
        schedule_index = schedule_len - 1  # index of the just-scheduled step

        sp = SwitchPoint(
            schedule_index=schedule_index,
            from_thread=from_thread,
            to_thread=to_thread,
            filename=code.co_filename,
            lineno=lineno,
            function_name=code.co_name,
            opcode=opcode,
            source_line=source_line,
            shadow_stack_top5=shadow_top5,
            access_type=access_type,
            attr_name=attr_name,
            obj_type_name=obj_type_name,
        )
        self._switch_point_collector.append(sp)  # type: ignore[union-attr]

    def _capture_step_event(self, frame: Any, thread_id: int, pre_opcode_stack: list[Any] | None = None) -> None:
        """Capture a StepEvent keyed by schedule index (path_id)."""
        from frontrun._report import StepEvent, _safe_repr

        if self._step_event_collector is None:
            return
        code = frame.f_code
        lineno = frame.f_lineno
        instr = _get_instructions(code).get(frame.f_lasti)
        opcode = instr.opname if instr else ""
        source_line = linecache.getline(code.co_filename, lineno).strip()

        # Get access info from the most recent trace event
        access_type: str | None = None
        attr_name: str | None = None
        obj_type_name: str | None = None
        if self.trace_recorder and self.trace_recorder.events:
            last_ev = self.trace_recorder.events[-1]
            if last_ev.thread_id == thread_id:
                access_type = last_ev.access_type
                attr_name = last_ev.attr_name
                obj_type_name = last_ev.obj_type_name

        # Capture the value involved in the access.
        # The trace callback fires *before* the instruction executes, so for
        # LOAD_ATTR we read the attribute that's about to be loaded, and for
        # STORE_ATTR we show the current (pre-store) value.
        # We read from the actual Python objects rather than the shadow stack,
        # which only tracks object identity (its elements are often None).
        value_repr: str | None = None
        try:
            if attr_name and obj_type_name and opcode in ("LOAD_ATTR", "STORE_ATTR") and instr:
                attr = instr.argval
                if attr:
                    # Find the object in frame locals — look for instances of the right type
                    for local_val in frame.f_locals.values():
                        if type(local_val).__name__ == obj_type_name:
                            val = getattr(local_val, attr, _SENTINEL)
                            if val is not _SENTINEL and not callable(val):
                                value_repr = _safe_repr(val)
                                break
            elif opcode in ("LOAD_GLOBAL", "STORE_GLOBAL") and instr:
                name = instr.argval
                if name and name in frame.f_globals:
                    val = frame.f_globals[name]
                    if not callable(val):
                        value_repr = _safe_repr(val)
        except Exception:
            pass

        # Key by schedule index (= len(schedule_trace) - 1, the most recently
        # scheduled step). This aligns with path_id used in race detection.
        with self._engine_lock:
            schedule_idx = len(self.execution.schedule_trace) - 1

        self._step_event_collector[schedule_idx] = StepEvent(
            thread_id=thread_id,
            filename=code.co_filename,
            lineno=lineno,
            function_name=code.co_name,
            opcode=opcode,
            source_line=source_line,
            access_type=access_type,
            attr_name=attr_name,
            obj_type_name=obj_type_name,
            value_repr=value_repr,
        )

    @staticmethod
    def get_shadow_stack(frame_id: int) -> ShadowStack:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is None:
            stacks = {}
            _dpor_tls._shadow_stacks = stacks
        if frame_id not in stacks:
            stacks[frame_id] = ShadowStack()
        return stacks[frame_id]

    @staticmethod
    def remove_shadow_stack(frame_id: int) -> None:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is not None:
            stacks.pop(frame_id, None)

    def _row_lock_int_id(self, res_id: str) -> int:
        """Return a stable monotonic integer ID for *res_id* (allocated on first call)."""
        return self._row_lock_registry._row_lock_int_id(res_id)

    # -- Replay-side virtual clock helpers (used by the replay subclasses) --

    def _replay_advance_clock_to(self, target: float | None = None) -> None:
        """Advance the clock during replay: wake due deadlines, no engine.

        Caller must hold ``self._condition``.  With ``target=None`` jumps to
        the earliest pending deadline (mirroring an exploration actor step).
        """
        clock = self.virtual_clock
        if clock is None:
            return
        # Shared core drops due entries and spin flags; replay has no engine, so
        # there is no per-event wake work.
        self._clock_port.advance_clock_to(clock, target, noop_on_wake)

    def _replay_sleep_self_wake(self, thread_id: int) -> bool:
        """Replay-only escape from ``sleep_until`` phase 1 (base: no-op).

        During exploration the clock advance must stay an engine choice, so
        the base class never self-wakes.  ``_ReplayDporScheduler`` overrides
        this: when the positional walk is suspended (access-gate waiters) or
        points at this very sleeper, only a clock advance can move the run
        forward — without it, each reproduction attempt burns a
        ``deadlock_timeout``.
        """
        return False

    def _wake_scheduled_sleeper(self) -> bool:
        """Advance the clock when replay schedules a *sleeping* thread.

        Safety net for schedule drift: when the positional/IO-anchored replay
        points at a thread blocked in ``sleep_until``, the only way forward is
        for time to pass — jump to that thread's deadline.  Caller must hold
        ``self._condition``.

        Timed lock acquires (``timeout``-kind deadlines) are deliberately
        excluded — ``sleep_deadline`` consults only ``sleep``-kind entries.  A
        thread registers a timed wait for the whole duration of a contended
        ``acquire(timeout=...)``, but — unlike a sleeper — it is not genuinely
        stuck: ``ReplayExecution.block_thread`` is a no-op, so the waiter keeps
        spinning through recorded probe entries and acquires the lock if the
        recorded run did.  Advancing to its deadline here would force-expire a
        wait that never timed out, flipping the acquire to its timeout branch
        and dragging every earlier deadline due.  Recorded timeout expiries are
        already replayed via the clock-actor entry handling in
        ``_ReplayDporScheduler._schedule_next`` plus the owed-advance
        bookkeeping.  (The async twin only checks sleepers as well; see
        ``async_dpor.py::should_proceed``.)
        """
        cur = self._current_thread
        if cur is None or self.virtual_clock is None:
            return False
        deadline = self._deadlines.sleep_deadline(cur)
        if deadline is None:
            return False
        self._replay_advance_clock_to(deadline)
        self._condition.notify_all()
        return True

    def _engine_block_thread(self, thread_id: int) -> None:
        """Mark *thread_id* blocked in the DPOR engine (finding 1).

        Held under ``_engine_lock`` to serialise PyO3 ``&mut self`` borrows,
        mirroring the cooperative-lock ``lock_wait`` path in ``runner.py``."""
        with self._engine_lock:
            self.execution.block_thread(thread_id)

    def _engine_unblock_thread(self, thread_id: int) -> None:
        """Clear the engine-blocked flag for *thread_id* (finding 1)."""
        with self._engine_lock:
            self.execution.unblock_thread(thread_id)

    def _schedule_idle_current_unlocked(self) -> bool:
        if self._current_thread is not None:
            return False
        next_thread = self._schedule_next()
        self._current_thread = next_thread
        if next_thread is None and len(self._threads_done) >= self.num_threads:
            self._finished = True
        self._condition.notify_all()
        return True

    def _wait_for_row_lock_turn_unlocked(self, thread_id: int) -> bool:
        """Wait until an unblocked row-lock waiter has a scheduler turn."""
        while True:
            if self._finished or self._error:
                return False
            if self._current_thread == thread_id:
                return True
            if self._reschedule_done_current_unlocked():
                continue
            if self._schedule_idle_current_unlocked():
                continue
            if not self._condition.wait(timeout=self._condition_wait_timeout()):
                if self._reschedule_done_current_unlocked():
                    continue
                if self._schedule_idle_current_unlocked():
                    continue
                self._error = TimeoutError(
                    "DPOR row-lock waiter was unblocked but not rescheduled: "
                    f"waiting for thread {thread_id}, current is {self._current_thread}"
                )
                self._condition.notify_all()
                return False

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> list[str]:
        """Block until all *resource_ids* can be held by *thread_id*.

        If another thread holds a conflicting lock, waits on the condition
        variable.  On timeout, aborts the scheduled transition rather than
        letting an unmodeled database call proceed.

        When a WaitForGraph is installed, registers waiting/holding edges for
        instant cycle-based deadlock detection.
        """
        from frontrun._deadlock import DeadlockError, SchedulerAbort, format_cycle, get_wait_for_graph

        graph = get_wait_for_graph()
        acquired: list[str] = []
        with self._condition:
            for res_id in resource_ids:
                lock_int_id = self._row_lock_int_id(res_id)
                already_held = False
                while True:
                    holder = self._active_row_locks.get(res_id)
                    if holder is None:
                        break
                    if holder == thread_id:
                        already_held = True
                        break
                    # Another thread holds this row lock — check for cycle first
                    if graph is not None:
                        cycle = graph.add_waiting(thread_id, lock_int_id, kind="row_lock")
                        if cycle is not None:
                            graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                            desc = format_cycle(cycle, self._row_lock_registry.id_to_resource())
                            err = DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                            if self._error is None:
                                self._error = err
                            self._condition.notify_all()
                            raise SchedulerAbort(str(err))
                    # Register this thread as row-lock-blocked so that
                    # _schedule_next() skips it and schedules the holder
                    # instead of cycling (defect #6).
                    self._row_lock_blocked[thread_id] = holder
                    # Tell the DPOR engine this thread is blocked so the engine
                    # never *schedules* it while it waits (finding 1). Without
                    # this, engine.schedule() could commit a step labelled with
                    # this thread while the holder actually runs, corrupting the
                    # engine's per-thread bookkeeping (notdep, sleep set
                    # propagation, preemption counts, and the recorded schedule).
                    self._engine_block_thread(thread_id)
                    if self._active_sync_thread == thread_id:
                        self._active_sync_thread = None
                        self._next_thread_after_sync = None
                    # Yield scheduling to the holder so it can run and
                    # either release the lock or block on one of ours
                    # (triggering WaitForGraph cycle detection).
                    if self._current_thread == thread_id:
                        self._current_thread = holder
                        self._condition.notify_all()
                    # Wait for the holder to release
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        self._row_lock_blocked.pop(thread_id, None)
                        self._engine_unblock_thread(thread_id)
                        if graph is not None:
                            graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                        if self._finished or self._error:
                            return acquired
                        err = TimeoutError(
                            f"DPOR row-lock wait timed out: thread {thread_id} waiting for {res_id!r} "
                            f"held by thread {holder}"
                        )
                        if self._error is None:
                            self._error = err
                        self._condition.notify_all()
                        raise SchedulerAbort(str(err))
                    self._row_lock_blocked.pop(thread_id, None)
                    self._engine_unblock_thread(thread_id)
                    if graph is not None:
                        graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                    if self._finished or self._error:
                        return acquired
                    if not self._wait_for_row_lock_turn_unlocked(thread_id):
                        return acquired
                if already_held:
                    acquired.append(res_id)
                    continue
                # Record ownership and notify graph — shared logic via registry.
                self._row_lock_registry.record_acquire(thread_id, res_id, graph)
                acquired.append(res_id)
                # Report row-lock acquire to the DPOR engine so vector clocks
                # reflect the serialization from database row locking.
                _elock = getattr(self, "_engine_lock", None)
                if _elock is not None:
                    _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                    with _elock:
                        self.engine.report_sync(self.execution, thread_id, "lock_acquire", lock_int_id, _saved_path_id)
        return acquired

    def _release_row_locks_unlocked(self, thread_id: int) -> bool:
        """Remove row locks for *thread_id*. Caller must hold ``self._condition``."""
        from frontrun._deadlock import get_wait_for_graph

        graph = get_wait_for_graph()
        released = self._row_lock_registry.pop_all(thread_id, graph)
        if not released:
            return False
        # Report each release to the DPOR engine so vector clocks
        # reflect the serialization from database row locking.
        _elock = getattr(self, "_engine_lock", None)
        if _elock is not None:
            _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
            for _res_id, lid in released:
                with _elock:
                    self.engine.report_sync(self.execution, thread_id, "lock_release", lid, _saved_path_id)
        return True

    def release_row_locks(self, thread_id: int) -> None:
        """Release all row locks held by *thread_id* (called on COMMIT/ROLLBACK)."""
        with self._condition:
            if self._release_row_locks_unlocked(thread_id):
                self._condition.notify_all()


class _ReplayDporScheduler(DporScheduler):
    """Replay a fixed DPOR schedule using the DPOR runner and SQL row-lock logic.

    When *access_schedule* is provided, accesses to the racing objects are
    additionally gated on the recorded access order (access-anchored replay,
    defect #20).  The positional bytecode schedule drifts when the code under
    test executes a run-varying number of traced opcodes between scheduling
    points (e.g. a real ``gpg`` subprocess between a racing write and read);
    the access anchors re-enforce the orderings that actually matter — the
    conflicting accesses themselves.  On anchor-stream mismatch the gate
    disables itself after ``deadlock_timeout`` and replay degrades to the
    positional-only behavior.
    """

    def __init__(
        self,
        schedule: list[int],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        detect_io: bool = False,
        access_schedule: list[tuple[int, str, str]] | None = None,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_actor_id: int | None = None,
    ) -> None:
        self._replay_schedule = list(schedule)
        self._replay_index = 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        self._access_anchors = list(access_schedule) if access_schedule else []
        self._anchor_enabled = bool(self._access_anchors)
        self._watched_labels = {label for _, label, _ in self._access_anchors}
        self._anchor_index = 0
        self._last_anchor: tuple[int, str, str] | None = None
        self._gate_waiters = 0
        super().__init__(
            _ReplayEngine(),  # type: ignore[arg-type]
            _ReplayExecution(),  # type: ignore[arg-type]
            num_threads,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
            virtual_clock=virtual_clock,
            clock_mode=clock_mode,
            clock_actor_id=clock_actor_id,
        )

    def on_traced_access(self, thread_id: int, obj: Any, name: Any, kind: str, key: int) -> None:
        """Gate accesses to racing objects on the recorded anchor order."""
        if not self._anchor_enabled:
            return
        label = _anchor_label(obj, name)
        if label is None or label not in self._watched_labels:
            return
        self._gate_access((thread_id, label, kind))

    def _gate_access(self, entry: tuple[int, str, str]) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _prev_machinery = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            waiting = False
            try:
                while True:
                    if self._finished or self._error or not self._anchor_enabled:
                        return
                    idx = self._anchor_index
                    # Skip anchors owned by threads that already finished
                    # (their remaining recorded accesses can never happen).
                    while idx < len(self._access_anchors) and self._access_anchors[idx][0] in self._threads_done:
                        idx += 1
                    self._anchor_index = idx
                    if idx >= len(self._access_anchors):
                        return
                    if self._access_anchors[idx] == entry:
                        self._anchor_index = idx + 1
                        self._last_anchor = entry
                        self._condition.notify_all()
                        return
                    if self._last_anchor == entry:
                        # Continuation of the anchor just consumed (one opcode
                        # can report the same access at several scheduling
                        # points) — let it through without consuming.
                        return
                    # Not this access's turn: wait for anchor progress.  While
                    # any thread gate-waits, _wait_for_turn suspends positional
                    # gating so the anchor-owning thread can reach its access.
                    if not waiting:
                        waiting = True
                        self._gate_waiters += 1
                        self._condition.notify_all()
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        # Anchor stream desynchronised (the replay run took a
                        # different path).  Give up on anchors; degrade to
                        # positional-only replay rather than deadlocking.
                        self._anchor_enabled = False
                        self._condition.notify_all()
                        return
            finally:
                if waiting:
                    self._gate_waiters -= 1
                    self._condition.notify_all()
                _scheduler_tls._in_dpor_machinery = _prev_machinery

    def _extend_schedule(self) -> bool:
        return extend_replay_schedule(
            self._replay_schedule,
            self._replay_index,
            self._replay_max_ops,
            self.num_threads,
            self._threads_done,
        )

    def _replay_sleep_self_wake(self, thread_id: int) -> bool:
        """Advance the clock to our own deadline when nothing else can.

        Two replay situations leave the sleeper as the only possible mover:
        the positional walk is suspended because other threads wait on access
        gates (they need *our* later writes), or the walk points at this very
        sleeper.  Caller (``sleep_until`` phase 1) holds ``_condition``.
        """
        if self.virtual_clock is None:
            return False
        deadline = self._deadlines.sleep_deadline(thread_id)
        if deadline is None:
            return False
        if self._gate_waiters > 0 or self._current_thread == thread_id:
            self._replay_advance_clock_to(deadline)
            self._condition.notify_all()
            return True
        return False

    def _schedule_next(self) -> int | None:
        while True:
            self._replay_index, next_actor = advance_replay_index(
                self._replay_schedule,
                self._replay_index,
                self._extend_schedule,
                self._threads_done,
            )
            if next_actor is not None and self._clock_actor_id is not None and next_actor == self._clock_actor_id:
                # Recorded clock-actor step: advance to the earliest pending
                # deadline and keep walking the schedule.  If the sleeper has not
                # registered its deadline yet, carry the advance until deadline
                # registration; the current trace format does not distinguish
                # that positional drift from a recorded no-op actor entry.
                if self._deadlines.has_pending():
                    self._replay_advance_clock_to()
                else:
                    # Positional drift: the sleeper has not registered its
                    # deadline yet.  Owe the advance — sleep_until /
                    # add_timed_wait perform it on registration.
                    self._pending_clock_advances += 1
                continue
            return next_actor

    def wait_for_turn(self, thread_id: int) -> bool:
        return self._wait_for_turn(thread_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        # Process the opcode to keep the shadow stack in sync with
        # exploration.  Without this, _call_might_report_access sees an
        # empty shadow stack during replay and skips CALL scheduling
        # points that existed during exploration, desynchronising the
        # schedule and preventing reproduction.
        if frame is not None:
            _process_opcode(frame, self, thread_id)
        return self._wait_for_turn(thread_id)

    def _wait_for_turn(self, thread_id: int) -> bool:
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False

                # Schedule drift safety net: if replay scheduled a thread that
                # is deadline-blocked, time must pass for it to move.
                self._wake_scheduled_sleeper()

                # While any thread is blocked on an access anchor, suspend
                # positional gating: the anchor-owning thread must be able to
                # reach its recorded access regardless of the (drifted)
                # positional schedule.  The anchors carry the ordering that
                # matters; the positional layer resumes once gates clear.
                if self._gate_waiters > 0:
                    return True

                if self._current_thread in self._threads_done:
                    self._current_thread = self._schedule_next()
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                        self._condition.notify_all()
                        return False
                    self._condition.notify_all()
                    continue

                if self._current_thread == thread_id:
                    self._current_thread = self._schedule_next()
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                    self._condition.notify_all()
                    return True

                if not self._condition.wait(timeout=self.deadlock_timeout):
                    if self._current_thread in self._threads_done:
                        self._current_thread = self._schedule_next()
                        if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                        self._condition.notify_all()
                        continue
                    self._error = TimeoutError(
                        f"DPOR replay deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                    )
                    self._condition.notify_all()
                    return False


class _IOAnchoredReplayScheduler(DporScheduler):
    """Replay using only IO boundaries as schedule anchors (defect #16).

    When detect_io=True, every CALL opcode is a scheduling point. If the code
    under test has state-dependent paths (e.g., early returns that skip Redis
    operations), the number of opcode-level scheduling points can change
    between exploration and replay, desynchronising the schedule.

    This scheduler uses a two-phase IO protocol:

    1. ``before_io(tid, resource_id)`` checks the next recorded anchor and
       blocks if it's not this thread's turn.

    2. ``after_io(tid, resource_id)`` (post-IO hook): called from the Redis
       interception layer *after* the command completes, inside the
       scheduler's condition lock.  Atomically records the IO event and
       switches ``current_thread`` to the next IO schedule entry.

    Between IO boundaries, one thread runs exclusively (enforced by opcode-
    level ``report_and_wait(frame, tid)`` blocking on ``current_thread``).
    """

    def __init__(
        self,
        io_schedule: list[tuple[int, str]],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        detect_io: bool = False,
        virtual_clock: VirtualClock | None = None,
        clock_mode: str = "real",
        clock_actor_id: int | None = None,
    ) -> None:
        self._io_schedule = list(io_schedule)
        self._io_index = 0
        super().__init__(
            _ReplayEngine(),  # type: ignore[arg-type]
            _ReplayExecution(),  # type: ignore[arg-type]
            num_threads,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
            virtual_clock=virtual_clock,
            clock_mode=clock_mode,
            clock_actor_id=clock_actor_id,
        )
        # _schedule_next() in super().__init__ consumed an entry.
        # Reset so the first IO scheduling point matches entry 0.
        self._io_index = 0
        # Set initial current_thread to the first IO schedule entry.
        self._current_thread = io_schedule[0][0] if io_schedule else 0
        # Bijective rebinding of run-specific Redis keys (defect: redis-om
        # style ORMs generate a fresh random primary key in every setup()
        # call, so the key embedded in a recorded anchor never reappears
        # in the replay run).  Maps recorded key -> replay key, bound
        # greedily on first sight; see _anchors_match().
        self._anchor_key_bindings: dict[str, str] = {}
        self._anchor_bound_keys: set[str] = set()

    def _anchors_match(self, expected: str, got: str) -> bool:
        """Whether a replay I/O anchor matches the recorded one.

        Exact string equality, with one relaxation: Redis anchors of the
        form ``redis\\x1f<cmd>\\x1f<key>\\x1f<db_scope>`` may differ in the
        *key* field when the key embeds run-specific random values (e.g. a
        redis-om ULID primary key created fresh by each replay's setup()).
        In that case the command and db scope must still match exactly and
        the recorded key is bound to the replay key bijectively — once
        bound, the same recorded key must always map to the same replay key
        (and no other recorded key may map to it), preserving cross-command
        key consistency (a thread's GET and SET of one object stay on one
        object).
        """
        exp_parts = expected.split("\x1f")
        got_parts = got.split("\x1f")
        if expected == got:
            # Record an identity binding for exact redis-anchor matches so
            # a deterministic key (e.g. "counter") can never later become
            # the image of a *different* recorded key.
            if len(exp_parts) == 4 and exp_parts[0] == "redis":
                key = exp_parts[2]
                if key not in self._anchor_key_bindings and key not in self._anchor_bound_keys:
                    self._anchor_key_bindings[key] = key
                    self._anchor_bound_keys.add(key)
            return True
        if len(exp_parts) != 4 or len(got_parts) != 4:
            return False
        if exp_parts[0] != "redis" or got_parts[0] != "redis":
            return False
        if exp_parts[1] != got_parts[1] or exp_parts[3] != got_parts[3]:
            return False  # different command or db scope
        exp_key, got_key = exp_parts[2], got_parts[2]
        bound = self._anchor_key_bindings.get(exp_key)
        if bound is not None:
            return bound == got_key
        if got_key in self._anchor_bound_keys:
            return False  # already the image of a different recorded key
        self._anchor_key_bindings[exp_key] = got_key
        self._anchor_bound_keys.add(got_key)
        return True

    def _schedule_next(self) -> int | None:
        """Override to use IO schedule instead of DPOR engine.

        Skip anchors whose thread has already finished (a state-dependent
        early return — the exact divergence this scheduler exists to tolerate,
        see defect #16). Such a thread will never reach that I/O boundary, so
        advancing ``_io_index`` past it lets replay proceed to the next live
        anchor instead of busy-spinning on the done thread. Anchors of live
        threads are still enforced in order.
        """
        while self._io_index < len(self._io_schedule) and self._io_schedule[self._io_index][0] in self._threads_done:
            self._io_index += 1
        if self._io_index >= len(self._io_schedule):
            active = [t for t in range(self.num_threads) if t not in self._threads_done]
            return active[0] if active else None
        return self._io_schedule[self._io_index][0]

    def _replay_sleep_self_wake(self, thread_id: int) -> bool:
        if self.virtual_clock is None:
            return False
        deadline = self._deadlines.sleep_deadline(thread_id)
        if deadline is None:
            return False
        if self._current_thread == thread_id:
            self._replay_advance_clock_to(deadline)
            self._condition.notify_all()
            return True
        return False

    def wait_for_turn(self, thread_id: int) -> bool:
        return self._wait_for_turn(thread_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        if frame is not None:
            _process_opcode(frame, self, thread_id)
        return self._wait_for_turn(thread_id)

    def before_io(self, thread_id: int, resource_id: str) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return

                    if self._current_thread in self._threads_done:
                        self._current_thread = self._schedule_next()
                        if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                            self._condition.notify_all()
                            return
                        self._condition.notify_all()
                        continue

                    if self._active_io_thread is not None and self._active_io_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        if self._io_index >= len(self._io_schedule):
                            self._error = RuntimeError(
                                "DPOR IO-anchored replay desynchronised: "
                                f"unexpected extra I/O anchor {(thread_id, resource_id)!r}"
                            )
                            self._condition.notify_all()
                            return

                        expected_tid, expected_resource_id = self._io_schedule[self._io_index]
                        if expected_tid != thread_id or not self._anchors_match(expected_resource_id, resource_id):
                            self._error = RuntimeError(
                                "DPOR IO-anchored replay desynchronised: "
                                f"expected {(expected_tid, expected_resource_id)!r}, "
                                f"got {(thread_id, resource_id)!r}"
                            )
                            self._condition.notify_all()
                            return

                        self._io_index += 1
                        self._active_io_thread = thread_id
                        self._next_thread_after_io = self._schedule_next()
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            self._current_thread = self._schedule_next()
                            if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR IO-anchored replay deadlock before {resource_id}: "
                            f"waiting for thread {thread_id}, current is {self._current_thread}, "
                            f"io_index={self._io_index}"
                        )
                        self._condition.notify_all()
                        return
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_io(self, thread_id: int, resource_id: str) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                if self._finished or self._error:
                    return
                if self._active_io_thread == thread_id:
                    self._active_io_thread = None
                    self._current_thread = self._next_thread_after_io
                    self._next_thread_after_io = None
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                self._condition.notify_all()
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def _wait_for_turn(self, thread_id: int) -> bool:
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False

                # If the IO schedule points at a deadline-blocked thread,
                # time must pass for it to reach its next IO boundary.
                self._wake_scheduled_sleeper()

                if self._current_thread in self._threads_done:
                    self._current_thread = self._schedule_next()
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                        self._condition.notify_all()
                        return False
                    self._condition.notify_all()
                    continue

                if self._active_io_thread == thread_id:
                    return True

                if self._current_thread == thread_id:
                    return True

                if not self._condition.wait(timeout=self.deadlock_timeout):
                    if self._current_thread in self._threads_done:
                        self._current_thread = self._schedule_next()
                        if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                        self._condition.notify_all()
                        continue
                    self._error = TimeoutError(
                        f"DPOR IO-anchored replay deadlock: waiting for thread {thread_id}, "
                        f"current is {self._current_thread}, io_index={self._io_index}"
                    )
                    self._condition.notify_all()
                    return False


# ---------------------------------------------------------------------------
# DPOR Bytecode Runner
# ---------------------------------------------------------------------------
