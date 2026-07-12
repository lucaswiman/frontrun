# ruff: noqa: F403, F405

from __future__ import annotations

from contextlib import contextmanager

from frontrun._dpor_core import event_wake_sync_id as _event_wake_sync_id
from frontrun._dpor_core.worker import WorkerTarget
from frontrun._opcode_observer import (
    OpcodeTraceHandle,
    install_thread_opcode_trace,
    set_access_sink,
    start_opcode_trace,
    stop_opcode_trace,
    uninstall_thread_opcode_trace,
)
from frontrun._threaded_runner import PatchScope, notify_scheduler_timeout
from frontrun._virtual_clock import warn_if_captured_time_reference

from ._shared import *
from ._shared import (
    _append_unique_lock_event,
    _dpor_tls,
    _make_object_key,
    _should_trace_file,
)
from .preload_bridge import _PreloadBridge
from .scheduler import DporScheduler
from .worker_set import ThreadWorkerSet


class DporBytecodeRunner:
    """Runs threads under DPOR-controlled bytecode-level interleaving.

    Opcode tracing is delegated to the tracer-backend in
    :mod:`frontrun._opcode_observer`, which picks ``sys.monitoring`` on
    Python 3.12+ and ``sys.settrace`` on 3.10–3.11.
    """

    def __init__(
        self,
        scheduler: DporScheduler,
        detect_io: bool = True,
        preload_bridge: _PreloadBridge | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.detect_io = detect_io
        self._preload_bridge = preload_bridge
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, BaseException] = {}
        self._lock_patched = False
        self._io_patched = False
        self._sleep_patched = False
        self._opcode_handle: OpcodeTraceHandle | None = None

    def _patch_locks(self) -> None:
        install_wait_for_graph()
        patch_locks()
        self._lock_patched = True

    def _unpatch_locks(self) -> None:
        if self._lock_patched:
            unpatch_locks()
            uninstall_wait_for_graph()
            self._lock_patched = False

    def _patch_sleep(self) -> None:
        patch_sleep()
        self._sleep_patched = True

    def _unpatch_sleep(self) -> None:
        if self._sleep_patched:
            unpatch_sleep()
            self._sleep_patched = False

    def _patch_io(self) -> None:
        if not self.detect_io:
            return
        patch_io()
        patch_sql()
        patch_redis()
        self._io_patched = True

    def _unpatch_io(self) -> None:
        if self._io_patched:
            unpatch_redis()
            unpatch_sql()
            unpatch_io()
            self._io_patched = False

    def patch_scope(self, *, patch_sleep: bool = True) -> PatchScope:
        # The time.* patch is owned by the driver's clock_scope, held once
        # across setup/run/invariant, rather than churned here per phase.
        scope = PatchScope()
        scope.add(self._patch_locks, self._unpatch_locks)
        scope.add(self._patch_io, self._unpatch_io)
        scope.add(self._patch_sleep, self._unpatch_sleep, enabled=patch_sleep)
        return scope

    def _start_opcode_trace(self) -> None:
        scheduler = self.scheduler
        _detect_io = scheduler._detect_io

        def _get_tid() -> int | None:
            tid = getattr(_dpor_tls, "thread_id", None)
            if tid is not None and getattr(_dpor_tls, "scheduler", None) is scheduler:
                return tid  # type: ignore[no-any-return]
            return None

        def _on_opcode(code: Any, offset: int, frame: Any, tid: int) -> bool:
            if scheduler._clock_diagnostics:
                warn_if_captured_time_reference(frame)
            yielded = process_opcode_with_coarsening(code, offset, frame, scheduler, tid, _detect_io)
            # report_and_wait returns after a scheduler abort so the traced
            # worker can unwind.  Do not let it resume user bytecode and
            # mutate state outside the schedule.
            if yielded and (scheduler._finished or scheduler._error is not None):
                raise SchedulerAbort("scheduler aborted while waiting for an opcode turn")
            return yielded

        self._opcode_handle = start_opcode_trace(
            get_thread_id=_get_tid,
            on_opcode=_on_opcode,
            remove_shadow_stack=scheduler.remove_shadow_stack,
            detect_io=_detect_io,
            is_active=lambda: not (scheduler._finished or scheduler._error),
            tool_name="frontrun._dpor",
        )

    def _stop_opcode_trace(self) -> None:
        handle = self._opcode_handle
        if handle is None:
            return
        stop_opcode_trace(handle)
        self._opcode_handle = None

    # --- Thread entry points ---

    def _setup_dpor_tls(self, thread_id: int) -> None:
        """Set up both shared cooperative TLS and DPOR-specific TLS."""
        scheduler = self.scheduler
        scheduler.register_worker_thread()
        engine = scheduler.engine
        execution = scheduler.execution
        engine_lock = scheduler._engine_lock
        # Shared context for cooperative primitives
        set_context(self.scheduler, thread_id)

        # Sync reporter so cooperative Lock/RLock report to the DPOR engine.
        # Must hold the engine_lock to serialise PyO3 &mut self borrows.
        # Also handles block/unblock for cooperative lock spinning:
        #   "lock_wait"    → execution.block_thread  (DPOR skips this thread)
        #   "lock_acquire" → execution.unblock_thread (DPOR can schedule again)
        #   "lock_release" → unblock all waiters for this lock
        # XOR masks matching Rust DporEngine constants for virtual lock objects.
        lock_object_xor = 0x4C4F_434B_4C4F_434B  # "LOCKLOCK"
        lock_release_xor = 0x524C_5345_524C_5345  # "RLSERLSE"

        def _register_lock_in_reverse_map(stable_id: int, lock_obj: object) -> None:
            """Register virtual lock objects in the reverse map for human-readable race info."""
            rmap = get_object_key_reverse_map()
            if rmap is None:
                return
            type_name = type(lock_obj).__name__
            acq_key = stable_id ^ lock_object_xor
            rel_key = stable_id ^ lock_release_xor
            if acq_key not in rmap:
                rmap[acq_key] = f"{type_name}(id={stable_id}).acquire"
            if rel_key not in rmap:
                rmap[rel_key] = f"{type_name}(id={stable_id}).release"

        def _append_lock_event(schedule_index: int, event_type: str, lock_id: int) -> None:
            if scheduler._lock_event_collector is None:
                return
            from frontrun._report import LockEvent as _LockEvent

            event = _LockEvent(
                schedule_index=schedule_index,
                thread_id=thread_id,
                event_type=event_type,
                lock_id=lock_id,
            )
            _append_unique_lock_event(scheduler._lock_event_collector, event)

        def _sync_reporter(event: str, obj_id: int, lock_obj: object) -> None:
            from frontrun._cooperative import _scheduler_tls

            # Use stable IDs so that lock identifiers are consistent across
            # DPOR executions (each execution creates fresh state via setup(),
            # so raw id() values change).  This is the same mechanism used for
            # shared-object access tracking.
            stable_lock_id = scheduler._stable_ids.get(lock_obj)
            _register_lock_in_reverse_map(stable_lock_id, lock_obj)

            # Set reentrancy guard so that GC-triggered __del__ chains
            # (e.g., redis.Redis.__del__) that acquire cooperative locks
            # will fall back to real blocking instead of re-entering the
            # scheduler.  See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:
                if event == "lock_wait":
                    with engine_lock:
                        scheduler._lock_waiters.setdefault(stable_lock_id, set()).add(thread_id)
                        execution.block_thread(thread_id)
                        _trace_len_wait = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_wait = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    if scheduler._error is not None:
                        # Capture teardown boundary once, then suppress all further events.
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_wait
                        return
                    if _trace_snap_wait is not None:
                        _wait_idx = next(
                            (i for i in range(len(_trace_snap_wait) - 1, -1, -1) if _trace_snap_wait[i] == thread_id),
                            max(0, len(_trace_snap_wait) - 1),
                        )
                        _append_lock_event(_wait_idx, "wait", stable_lock_id)
                    return
                if event == "lock_acquire":
                    with engine_lock:
                        waiter_set = scheduler._lock_waiters.get(stable_lock_id)
                        if waiter_set is not None and thread_id in waiter_set:
                            waiter_set.discard(thread_id)
                            execution.unblock_thread(thread_id)
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        engine.report_sync(execution, thread_id, "lock_acquire", stable_lock_id, _saved_path_id)
                        _trace_len_acq = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_acq = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    new_depth = getattr(_dpor_tls, "lock_depth", 0) + 1
                    _dpor_tls.lock_depth = new_depth
                    scheduler._lock_depth_by_thread[thread_id] = new_depth
                    if scheduler._error is not None:
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_acq
                        return
                    if _trace_snap_acq is not None:
                        _acq_idx = next(
                            (i for i in range(len(_trace_snap_acq) - 1, -1, -1) if _trace_snap_acq[i] == thread_id),
                            max(0, len(_trace_snap_acq) - 1),
                        )
                        _append_lock_event(_acq_idx, "acquire", stable_lock_id)
                    return
                if event == "lock_release":
                    with engine_lock:
                        waiters = scheduler._lock_waiters.pop(stable_lock_id, set())
                        for waiter in waiters:
                            execution.unblock_thread(waiter)
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        engine.report_sync(execution, thread_id, "lock_release", stable_lock_id, _saved_path_id)
                        _trace_len_rel = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_rel = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    new_depth = max(0, getattr(_dpor_tls, "lock_depth", 1) - 1)
                    _dpor_tls.lock_depth = new_depth
                    scheduler._lock_depth_by_thread[thread_id] = new_depth
                    if scheduler._error is not None:
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_rel
                        return
                    if _trace_snap_rel is not None:
                        _rel_idx = next(
                            (i for i in range(len(_trace_snap_rel) - 1, -1, -1) if _trace_snap_rel[i] == thread_id),
                            max(0, len(_trace_snap_rel) - 1),
                        )
                        _append_lock_event(_rel_idx, "release", stable_lock_id)
                    # Wake threads that may now be schedulable
                    with scheduler._condition:
                        scheduler._condition.notify_all()
                    return
                # Cooperative Event machinery (see CooperativeEvent): waiters
                # engine-block until set().  Wake happens-before edges use
                # per-waiter sync ids (like the virtual clock's wake edges) so
                # each of N waiters gets its own release/acquire pair, and the
                # lock_depth bookkeeping above is deliberately not touched.
                if event == "event_wait":
                    with engine_lock:
                        if lock_obj.is_set():  # type: ignore[attr-defined]
                            # set() already ran (serialized by engine_lock):
                            # blocking now would be a lost wakeup.
                            return
                        scheduler._lock_waiters.setdefault(stable_lock_id, set()).add(thread_id)
                        execution.block_thread(thread_id)
                    return
                if event == "event_set":
                    with engine_lock:
                        waiters = scheduler._lock_waiters.pop(stable_lock_id, set())
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        for waiter in sorted(waiters):
                            execution.unblock_thread(waiter)
                            engine.report_sync(
                                execution,
                                thread_id,
                                "lock_release",
                                _event_wake_sync_id(stable_lock_id, waiter),
                                _saved_path_id,
                            )
                    with scheduler._condition:
                        scheduler._condition.notify_all()
                    return
                if event in ("event_read", "event_write"):
                    state_key = _make_object_key(stable_lock_id, "__event_state__")
                    with engine_lock:
                        engine.report_access(
                            execution,
                            thread_id,
                            state_key,
                            "read" if event == "event_read" else "write",
                        )
                    return
                if event == "event_wake":
                    with engine_lock:
                        waiter_set = scheduler._lock_waiters.get(stable_lock_id)
                        if waiter_set is not None:
                            waiter_set.discard(thread_id)
                        execution.unblock_thread(thread_id)
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        engine.report_sync(
                            execution,
                            thread_id,
                            "lock_acquire",
                            _event_wake_sync_id(stable_lock_id, thread_id),
                            _saved_path_id,
                        )
                    return
                with engine_lock:
                    _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                    engine.report_sync(execution, thread_id, event, stable_lock_id, _saved_path_id)
            finally:
                _scheduler_tls._in_dpor_machinery = False

        set_sync_reporter(_sync_reporter)
        # Access sink: exploration records the access trace for possible
        # access-anchored replay; replay schedulers gate racing accesses on
        # the recorded order (defect #20).
        set_access_sink(self.scheduler.on_traced_access)
        # DPOR-specific TLS for _process_opcode (shadow stacks, etc.)
        _dpor_tls.scheduler = self.scheduler
        _dpor_tls.thread_id = thread_id
        _dpor_tls.engine = engine
        _dpor_tls.execution = execution
        _dpor_tls._last_path_id = None  # set by report_and_wait after schedule()

        # IO detection: defer I/O reports while this thread is inside locks.
        # If another thread later reaches a real I/O boundary, it will flush
        # these deferred accesses before reporting its own I/O so the race
        # window is preserved without adding spurious paths for threads that
        # only block on the same lock.
        _dpor_tls.lock_depth = 0
        self.scheduler._lock_depth_by_thread[thread_id] = 0
        pending_io: list[tuple[int, str, bool]] = []
        _dpor_tls.pending_io = pending_io
        self.scheduler._pending_io_by_thread[thread_id] = pending_io

        # Register OS TID → DPOR thread ID mapping for LD_PRELOAD events.
        if self._preload_bridge is not None:
            self._preload_bridge.register_thread(threading.get_native_id(), thread_id)

        from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id

        set_dpor_scheduler(self.scheduler)
        set_dpor_thread_id(thread_id)

        if self.detect_io:
            _recorder = scheduler.trace_recorder
            _registered_groups: set[int] = set()

            def _io_reporter(resource_id: str, kind: str) -> None:
                object_key = _make_object_key(hash(resource_id), resource_id)
                pending: list[tuple[int, str, bool]] = _dpor_tls.pending_io
                pending.append((object_key, kind, True))  # synced=True: Python-level I/O respects locks
                # Register resource group for SQL resources so the DPOR engine
                # can skip inline wakeup insertion and rely on deferred notdep
                # processing. This prevents backtrack explosion from intermediate
                # operations on unrelated tables (Defect #15).
                if resource_id.startswith("sql:"):
                    parts = resource_id.split(":")
                    table_group = f"sql:{parts[1]}" if len(parts) >= 2 else resource_id
                    group_key = hash(table_group) & 0xFFFFFFFFFFFFFFFF
                    if object_key not in _registered_groups:
                        with engine_lock:
                            engine.register_resource_group(object_key, group_key)
                        _registered_groups.add(object_key)
                # Record I/O event in the trace for human-readable output
                if _recorder is not None:
                    _frame = sys._getframe(1)
                    # Walk up to find user code (skip frontrun internals)
                    while _frame is not None and not _should_trace_file(_frame.f_code.co_filename):
                        _frame = _frame.f_back
                    if _frame is not None:
                        chain = build_call_chain(_frame, filter_fn=_should_trace_file)
                        _recorder.record(
                            thread_id=thread_id,
                            frame=_frame,
                            opcode="IO",
                            access_type=kind,
                            attr_name=resource_id,
                            call_chain=chain,
                        )

            set_io_reporter(_io_reporter)

    def _teardown_dpor_tls(self) -> None:
        """Clean up both shared and DPOR-specific TLS."""
        _tid = getattr(_dpor_tls, "thread_id", None)
        if self._preload_bridge is not None:
            self._preload_bridge.unregister_thread(threading.get_native_id())

        # Flush any orphaned pending_io before tearing down TLS.
        # _capture_insert_id adds the indexical alias WRITE to pending_io
        # AFTER the INSERT's report_and_wait scheduling point.  If the INSERT
        # is the last operation before the thread exits, the alias WRITE
        # would be silently discarded.  Flushing here ensures the DPOR engine
        # sees all I/O events, even those reported after the last scheduling
        # point.
        if _tid is not None:
            from frontrun._cooperative import _scheduler_tls

            previous_guard = getattr(_scheduler_tls, "_in_dpor_machinery", False)
            _scheduler_tls._in_dpor_machinery = True
            try:
                with self.scheduler._condition:
                    self.scheduler._flush_pending_io_for_unlocked(_tid, allow_inside_lock=True)
                    self.scheduler._lock_depth_by_thread.pop(_tid, None)
                    self.scheduler._pending_io_by_thread.pop(_tid, None)
            finally:
                _scheduler_tls._in_dpor_machinery = previous_guard

        if self.detect_io:
            set_io_reporter(None)
        from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id

        set_dpor_scheduler(None)
        set_dpor_thread_id(None)
        clear_context()
        set_sync_reporter(None)
        set_access_sink(None)
        _dpor_tls.scheduler = None
        _dpor_tls.thread_id = None
        _dpor_tls.engine = None
        _dpor_tls.execution = None
        _dpor_tls.lock_depth = 0
        _dpor_tls.pending_io = []

    @contextmanager
    def _thread_runtime(self, thread_id: int):
        self._setup_dpor_tls(thread_id)
        handle = self._opcode_handle
        if handle is not None:
            install_thread_opcode_trace(handle)
        try:
            yield
        finally:
            if handle is not None:
                uninstall_thread_opcode_trace(handle)
            self._teardown_dpor_tls()
            self.scheduler.mark_done(thread_id)
            self.scheduler.unregister_worker_thread()

    def _run_thread(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        try:
            with self._thread_runtime(thread_id):
                func(*args)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except BaseException as e:
            self.errors[thread_id] = e
            if isinstance(e, Exception):
                self.scheduler.report_error(e)
            else:
                # SystemExit/KeyboardInterrupt raised in a worker thread do not
                # reach the caller automatically. Wake peer workers with an
                # ordinary scheduler error; _raise_recorded_errors re-raises
                # the original BaseException on the driver thread.
                self.scheduler.report_error(RuntimeError(f"worker {thread_id} terminated with {type(e).__name__}"))

    def run(
        self,
        funcs: list[Callable[..., None]],
        args: list[tuple[Any, ...]] | None = None,
        timeout: float = 10.0,
    ) -> None:
        if args is None:
            args = [() for _ in funcs]

        self._start_opcode_trace()
        run_thread = self._run_thread

        targets = [
            WorkerTarget(worker_id=i, func=run_thread, args=(i, func, tuple(thread_args)))
            for i, (func, thread_args) in enumerate(zip(funcs, args))
        ]

        def on_timeout(alive: list[threading.Thread]) -> None:
            notify_scheduler_timeout(self.scheduler, alive)

        worker_set = ThreadWorkerSet(name_prefix="dpor", thread_store=self.threads)
        worker_set.run(
            targets,
            timeout=timeout,
            on_timeout=on_timeout,
            teardown=self._stop_opcode_trace,
        )

        self._raise_recorded_errors()

    def _raise_recorded_errors(self) -> None:
        """Raise the first genuine worker error, if any.

        A ``TimeoutError`` recorded by one thread must not mask a real failure
        (e.g. an ``AssertionError``) recorded by another thread.  We therefore
        prefer the first non-timeout error and only treat the run as
        timeout-only when *every* recorded error is a ``TimeoutError`` (finding
        4)."""
        if not self.errors:
            return
        for error in self.errors.values():
            if not isinstance(error, TimeoutError):
                raise error


# ---------------------------------------------------------------------------
# Fixed-schedule replay
# ---------------------------------------------------------------------------
