# ruff: noqa: F403, F405
# pyright: reportUnusedClass=false

from __future__ import annotations

from frontrun._dpor_core import ReplayEngine as _ReplayEngine
from frontrun._dpor_core import ReplayExecution as _ReplayExecution
from frontrun._dpor_core import RowLockRegistry, advance_replay_index, extend_replay_schedule

from ._shared import *
from ._shared import _dpor_tls, _get_instructions, _process_opcode
from .preload_bridge import _PreloadBridge

_SENTINEL = object()


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
    ) -> None:
        self.engine = engine
        self.execution = execution
        self.num_threads = num_threads
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
        # reporter, and the main explore_dpor loop.
        self._engine_lock: threading.Lock = engine_lock if engine_lock is not None else real_lock()
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self._current_thread: int | None = None

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

        # Request the first scheduling decision
        self._current_thread = self._schedule_next()

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which thread to run next.

        If the engine selects a thread that is blocked on a DPOR row lock,
        override the decision and schedule the lock holder instead.  This
        prevents the scheduler from cycling between a blocked thread and
        its holder (defect #6).

        Also snapshots ``engine.path_position`` under the engine lock so
        that ``report_and_wait`` can attribute subsequent lock events to
        the correct scheduling step (see ``_last_scheduled_path_id``).
        """
        with self._engine_lock:
            runnable = self.execution.runnable_threads()
            if not runnable:
                self._last_scheduled_path_id = None
                return None

            scheduled = self.engine.schedule(self.execution)
            # Snapshot path position under engine_lock. On free-threaded
            # Python, another thread may call schedule() concurrently
            # after we release the lock, advancing path.pos.  The saved
            # position ensures _sync_reporter attributes lock events to
            # the correct step.
            _pp = getattr(self.engine, "path_position", None)
            self._last_scheduled_path_id = _pp - 1 if _pp is not None else None
            if scheduled is not None and scheduled in self._row_lock_blocked:
                holder = self._row_lock_blocked[scheduled]
                if holder not in self._threads_done:
                    return holder
                # Holder is done — lock should have been released via
                # mark_done → _release_row_locks_unlocked.  Clean up the
                # stale blocked entry so scheduled can proceed.
                self._row_lock_blocked.pop(scheduled, None)
            return scheduled

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

                    if self._active_sync_thread is not None and self._active_sync_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        next_thread = self._schedule_next()
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        self._active_sync_thread = thread_id
                        self._next_thread_after_sync = next_thread
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return True

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
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
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            # Current thread is done, try scheduling again
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
                        if self._current_thread in self._threads_done:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
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
        with self._condition:
            self._threads_done.add(thread_id)
            with self._engine_lock:
                self.execution.finish_thread(thread_id)
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

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Block until all *resource_ids* can be held by *thread_id*.

        If another thread holds a conflicting lock, waits on the condition
        variable.  On timeout (the holder is likely blocked in C too), lets
        the C call proceed — the ``lock_timeout`` PostgreSQL safety net
        will handle it as a fast error rather than an indefinite hang.

        When a WaitForGraph is installed, registers waiting/holding edges for
        instant cycle-based deadlock detection.
        """
        from frontrun._deadlock import DeadlockError, SchedulerAbort, format_cycle, get_wait_for_graph

        graph = get_wait_for_graph()
        with self._condition:
            for res_id in resource_ids:
                lock_int_id = self._row_lock_int_id(res_id)
                while True:
                    holder = self._active_row_locks.get(res_id)
                    if holder is None or holder == thread_id:
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
                            return
                        # Timeout — the holder is probably blocked in C too.
                        # Let the C call proceed; lock_timeout safety net will handle it.
                        return
                    self._row_lock_blocked.pop(thread_id, None)
                    self._engine_unblock_thread(thread_id)
                    if graph is not None:
                        graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                    if self._finished or self._error:
                        return
                # Record ownership and notify graph — shared logic via registry.
                self._row_lock_registry.record_acquire(thread_id, res_id, graph)
                # Report row-lock acquire to the DPOR engine so vector clocks
                # reflect the serialization from database row locking.
                _elock = getattr(self, "_engine_lock", None)
                if _elock is not None:
                    _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                    with _elock:
                        self.engine.report_sync(self.execution, thread_id, "lock_acquire", lock_int_id, _saved_path_id)

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
    """Replay a fixed DPOR schedule using the DPOR runner and SQL row-lock logic."""

    def __init__(
        self,
        schedule: list[int],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        detect_io: bool = False,
    ) -> None:
        self._replay_schedule = list(schedule)
        self._replay_index = 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        super().__init__(
            _ReplayEngine(),  # type: ignore[arg-type]
            _ReplayExecution(),  # type: ignore[arg-type]
            num_threads,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
        )

    def _extend_schedule(self) -> bool:
        return extend_replay_schedule(
            self._replay_schedule,
            self._replay_index,
            self._replay_max_ops,
            self.num_threads,
            self._threads_done,
        )

    def _schedule_next(self) -> int | None:
        self._replay_index, next_actor = advance_replay_index(
            self._replay_schedule,
            self._replay_index,
            self._extend_schedule,
            self._threads_done,
        )
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
        )
        # _schedule_next() in super().__init__ consumed an entry.
        # Reset so the first IO scheduling point matches entry 0.
        self._io_index = 0
        # Set initial current_thread to the first IO schedule entry.
        self._current_thread = io_schedule[0][0] if io_schedule else 0

    def _schedule_next(self) -> int | None:
        """Override to use IO schedule instead of DPOR engine."""
        if self._io_index >= len(self._io_schedule):
            active = [t for t in range(self.num_threads) if t not in self._threads_done]
            return active[0] if active else None
        return self._io_schedule[self._io_index][0]

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
                        if expected_tid != thread_id or expected_resource_id != resource_id:
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
