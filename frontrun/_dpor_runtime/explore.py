# ruff: noqa: F403, F405

from __future__ import annotations

from frontrun._dpor_core import (
    compute_serializable_baseline_sync,
    dpor_exploration_iter,
    format_race_failure_explanation,
    make_deadline,
    make_dpor_engine,
    record_dpor_failure,
)
from frontrun._virtual_clock import ClockMode, VirtualClock, clock_scope, validate_clock_options

from ._shared import *
from ._shared import _require_frontrun_env, _set_active_trace_filter, _TraceFilter
from .preload_bridge import _PreloadBridge
from .replay import _reproduce_dpor_counterexample
from .runner import DporBytecodeRunner
from .scheduler import DporScheduler


def _scheduler_run_evaluable(error: BaseException | None) -> bool:
    """Whether a run's invariant/race/serializability checks are meaningful.

    ``False`` when the scheduler recorded a deadlock (``DeadlockError``) or
    hit its fallback deadlock-timeout (a plain ``TimeoutError``).  In the
    timeout case surviving threads free-ran *unscheduled*, so the final state
    corresponds to no DPOR-controlled schedule (finding 5).
    """
    return not isinstance(error, (DeadlockError, TimeoutError))


def _explore_dpor(  # pyright: ignore[reportUnusedFunction]  # called cross-module by frontrun._strategy and contrib helpers
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    detect_io: bool = True,
    deadlock_timeout: float = 5.0,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
    trace_packages: list[str] | None = None,
    track_dunder_dict_accesses: bool = False,
    search: str | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[T], Any] | bool = False,
    error_on_any_race: bool = False,
    clock: ClockMode = "real",
    clock_diagnostics: bool = False,
) -> InterleavingResult:
    """Systematically explore interleavings using DPOR.

    Internal sync DPOR exploration. Called via :func:`frontrun.explore`
    (strategy='dpor', the default).  Uses the DPOR algorithm to explore only
    distinct interleavings (modulo independent operation reordering).

    Args:
        setup: Creates fresh shared state for each execution.
        threads: List of callables, each receiving the shared state.
        invariant: Predicate over shared state; must be True after all
            threads complete.
        max_executions: Safety limit on total executions (None = unlimited).
        preemption_bound: Limit on preemptions per execution. 2 catches most
            bugs. None = unbounded (full DPOR).
        max_branches: Maximum scheduling points per execution.
        timeout_per_run: Timeout for each individual run.
        stop_on_first: If True (default), stop exploring as soon as the
            first invariant violation is found.  Set to False to collect
            all failing interleavings.
        detect_io: Automatically detect socket/file I/O operations and
            report them as resource accesses (default True).  Two threads
            accessing the same endpoint or file will be treated as
            conflicting, enabling DPOR to explore their orderings.
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
        lock_timeout: If set, automatically execute
            ``SET lock_timeout = '<N>ms'`` on every new PostgreSQL
            connection created through the patched ``psycopg2.connect``
            (or ``psycopg.connect``).  This prevents the cooperative
            scheduler from deadlocking when two threads contend on the
            same PostgreSQL row lock (defect #6).  Value is in
            milliseconds; 2000 (2 seconds) is a good default.
        trace_packages: List of package name patterns (fnmatch syntax) to
            trace in addition to user code.  By default, code in
            site-packages is skipped.  Use this to include specific
            installed packages, e.g. ``["django_*", "mylib.*"]``.
        track_dunder_dict_accesses: If True, report accesses on ``obj.__dict__``
            in addition to direct attribute accesses.  This catches the
            rare case where one thread uses ``self.x = v`` and another
            uses ``self.__dict__['x'] = v``, but doubles wakeup tree
            insertions and can cause combinatorial explosion.  Default
            False.
        search: Controls the order in which wakeup tree branches are
            explored.  All strategies visit the same set of Mazurkiewicz
            trace equivalence classes; only the order differs.  Accepted
            values:

            - ``None`` or ``"dfs"`` — classic DFS, lowest thread ID first
              (default, matches the paper's Algorithm 2).  **Best for
              exhaustive exploration** (``stop_on_first=False``): produces
              the optimal (minimum) number of executions.
            - ``"bit-reversal"`` or ``"bit-reversal:<seed>"`` — visit
              children in bit-reversal permutation order for maximal
              spread across distinct conflict points early.
            - ``"round-robin"`` or ``"round-robin:<seed>"`` — cycle
              through available threads in rotating order.
            - ``"stride"`` or ``"stride:<seed>"`` — visit every s-th
              sibling (s coprime to branching factor, derived from seed).
            - ``"conflict-first"`` — reverse of DFS (highest thread ID
              first), preferring threads added by race reversals.

            **Use a non-DFS strategy when the trace space is large and
            you have a limited execution budget** (``stop_on_first=True``
            or a low ``max_executions``).  DFS explores traces in a fixed
            order and may spend many executions on similar interleavings
            before reaching a bug.  The alternative strategies spread
            exploration across different conflict points earlier, finding
            bugs faster on average.
        patch_sleep: If True (default), replace ``time.sleep`` with a
            cooperative scheduler hook. With ``clock="real"``, sleep calls are
            zero-wall-time scheduling points. With a virtual clock, positive
            sleeps become virtual deadlines and ``sleep(0)`` remains a yield.
            Set to False if your code depends on real delays.
        clock: ``"real"`` (default) leaves time untouched.  ``"virtual"``
            gives each execution a scheduler-owned virtual clock:
            ``time.time``/``time.monotonic``/``time.perf_counter`` return
            virtual time in explored code, ``time.sleep`` becomes a timed
            block, and the clock autojumps to the earliest pending deadline
            when no thread is runnable.  ``"explored"`` additionally models
            the clock advance as a synthetic DPOR actor, so "the timer fired
            between your read and your write" is explored like any other
            interleaving.  Requires ``patch_sleep=True``.

    Returns:
        InterleavingResult with exploration statistics and any counterexample found.

    .. note::

       When running under **pytest**, this function requires the
       ``frontrun`` CLI wrapper (``frontrun pytest ...``) or the
       ``--frontrun-patch-locks`` flag.  Without it, the test is
       automatically skipped.
    """
    _require_frontrun_env("frontrun.explore")
    clock = validate_clock_options(
        clock,
        patch_sleep=patch_sleep,
        serializable_invariant=serializable_invariant,
        clock_diagnostics=clock_diagnostics,
    )
    if trace_packages is not None:
        _set_active_trace_filter(_TraceFilter(trace_packages))

    # Compute serializable baseline if requested.
    serial_valid_states, serial_hash_fn = compute_serializable_baseline_sync(setup, threads, serializable_invariant)

    num_threads = len(threads)
    # With a virtual clock the engine gets one extra thread: the clock actor
    # (id == num_threads), whose steps advance the clock (see scheduler.py).
    clock_actor_id = num_threads if clock != "real" else None
    engine = make_dpor_engine(
        num_threads=num_threads + (1 if clock_actor_id is not None else 0),
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
        search=search,
    )

    result = InterleavingResult(property_holds=True)
    stable_ids = StableObjectIds()
    # Shared lock serialising ALL PyO3 calls to engine/execution objects.
    # On free-threaded Python, PyO3 &mut self borrows panic rather than
    # block when contested, so we need a Python-level lock shared across
    # worker threads, the sync reporter, and the main loop.
    engine_lock = real_lock()
    total_deadline = make_deadline(total_timeout)

    # Set up the LD_PRELOAD → DPOR bridge for C-level I/O detection.
    # When code under test uses C extensions that call libc send()/recv()
    # directly (e.g. psycopg2/libpq), the Python-level monkey-patches in
    # _io_detection can't see those calls.  The LD_PRELOAD library
    # intercepts them at the C level and writes events to a pipe.  The
    # IOEventDispatcher reads the pipe in a background thread and the
    # _PreloadBridge routes events to the correct DPOR thread for
    # conflict analysis.
    preload_dispatcher = None
    preload_bridge: _PreloadBridge | None = None
    if detect_io:
        from frontrun._preload_io import IOEventDispatcher

        preload_dispatcher = IOEventDispatcher()
        preload_bridge = _PreloadBridge(dispatcher=preload_dispatcher)
        preload_dispatcher.add_listener(preload_bridge.listener)
        preload_dispatcher.start()

    clear_sql_metadata()

    # Warm SQL parsers (sqlglot) BEFORE the first _patch_locks() call.
    # sqlglot creates a module-level _import_lock = threading.RLock() on
    # first import.  If that import happens after _patch_locks() replaces
    # threading.RLock with CooperativeRLock, the lock becomes cooperative.
    # If a worker thread is then killed while holding it (e.g. timeout),
    # the underlying real lock stays locked forever, causing deadlocks in
    # later phases (_reproduce_dpor_counterexample).  Warming here ensures
    # the lock is a real RLock.
    if detect_io:
        from frontrun._sql_cursor import _warm_sql_parsers

        _warm_sql_parsers()

    # Inject SET lock_timeout on new PG connections (defect #6 workaround).
    from frontrun._sql_cursor import get_lock_timeout, set_lock_timeout

    prev_lock_timeout = get_lock_timeout()
    if lock_timeout is not None:
        set_lock_timeout(lock_timeout)

    # Set up report collection if --frontrun-report is active
    from frontrun._report import (
        _MAX_RECORDED_EXECUTIONS,
        ExecutionRecord,
        ExplorationReport,
        _global_report_path,
        generate_html_report,
    )

    def _build_race_info(raw_races: list[tuple[int, int, int, int | None]]) -> list[dict[str, Any]] | None:
        if not raw_races:
            return None
        rmap = get_object_key_reverse_map() or {}
        non_dict_keys: set[tuple[int, int, int]] = set()
        for r in raw_races:
            obj_name = rmap.get(r[3], f"object {r[3]}") if r[3] is not None else None
            if not (obj_name and obj_name.startswith("dict.")):
                non_dict_keys.add((r[0], r[1], r[2]))
        result = []
        for r in raw_races:
            obj_name = rmap.get(r[3], f"object {r[3]}") if r[3] is not None else None
            key = (r[0], r[1], r[2])
            if obj_name and obj_name.startswith("dict.") and key in non_dict_keys:
                continue
            result.append(
                {
                    "prev_step": r[0],
                    "current_step": r[1],
                    "thread_id": r[2],
                    "object": obj_name,
                }
            )
        return result or None

    report_path = _global_report_path
    report: ExplorationReport | None = None
    if report_path is not None:
        report = ExplorationReport(
            num_threads=num_threads,
            thread_names=[f"Thread {i}" for i in range(num_threads)],
        )
        set_object_key_reverse_map({})

    def _record_and_emit_report(*, was_deadlock: bool = False) -> None:
        """Record the current execution to the report and write the HTML file."""
        if report is None or report_path is None:
            return
        if not _collecting_report:
            generate_html_report(report, report_path)
            return
        with engine_lock:
            sched = list(execution.schedule_trace)
            races = engine.pending_races()
        report.executions.append(
            ExecutionRecord(
                index=len(report.executions),
                schedule_trace=sched,
                switch_points=switch_points,
                invariant_held=False,
                was_deadlock=was_deadlock,
                race_info=_build_race_info(races),
                step_events=scheduler._step_event_collector or {},
                lock_events=scheduler._lock_event_collector or [],
                deadlock_at=scheduler._deadlock_at,
                deadlock_cycle_description=getattr(scheduler._error, "cycle_description", None)
                if was_deadlock
                else None,
            )
        )
        generate_html_report(report, report_path)

    try:
        for step in dpor_exploration_iter(
            engine=engine,
            engine_lock=engine_lock,
            stable_ids=stable_ids,
            total_deadline=total_deadline,
        ):
            execution = step.execution
            recorder = TraceRecorder()
            # Clear bridge state for this new execution.
            if preload_bridge is not None:
                preload_bridge.clear()
            # Clear persistent SQL suppression flags from previous execution.
            from frontrun._sql_cursor import clear_permanent_suppressions

            clear_permanent_suppressions()
            # Set up switch point collection for the report
            _collecting_report = report is not None and len(report.executions) < _MAX_RECORDED_EXECUTIONS
            switch_points: list[Any] = []
            # Fresh virtual clock per execution so every interleaving starts
            # from the same deterministic epoch.
            virtual_clock = VirtualClock() if clock != "real" else None
            scheduler = DporScheduler(
                engine,
                execution,
                num_threads,
                engine_lock=engine_lock,
                deadlock_timeout=deadlock_timeout,
                trace_recorder=recorder,
                preload_bridge=preload_bridge,
                detect_io=detect_io,
                stable_ids=stable_ids,
                switch_point_collector=switch_points if _collecting_report else None,
                track_dunder_dict_accesses=track_dunder_dict_accesses,
                virtual_clock=virtual_clock,
                clock_mode=clock,
                clock_actor_id=clock_actor_id,
                clock_diagnostics=clock_diagnostics,
            )
            runner = DporBytecodeRunner(scheduler, detect_io=detect_io, preload_bridge=preload_bridge)

            with runner.patch_scope(patch_sleep=patch_sleep, virtual_time=virtual_clock is not None):
                with clock_scope(virtual_clock):
                    state = setup()
                # Assign stable object IDs in deterministic, schedule-independent
                # order *before* any worker runs.  Without this, IDs are assigned
                # in first-touch order, which DPOR backtracks permute across
                # executions — corrupting the Rust sleep-set / trace-cache
                # comparison that carries object-ID-keyed state across executions
                # (silently pruning genuinely distinct interleavings).
                stable_ids.pre_register(state)

                def make_thread_func(thread_func: Callable[[T], None], s: T) -> Callable[[], None]:
                    def wrapper() -> None:
                        thread_func(s)

                    return wrapper

                funcs = [make_thread_func(t, state) for t in threads]
                try:
                    runner.run(funcs, timeout=timeout_per_run)
                except TimeoutError:
                    pass

            result.num_explored += 1

            # Check for deadlock before running the invariant — a deadlock
            # means the program never completed, so the invariant can never be
            # satisfied.  Report it as a property violation with a clear message.
            _deadlock_err = scheduler._error if isinstance(scheduler._error, DeadlockError) else None
            # A scheduler-internal TimeoutError means the run free-ran
            # unscheduled (finding 5): the program state does not describe any
            # DPOR schedule, so invariant/race/serializability checks below
            # must be skipped rather than scored as a normal completion.
            scheduler_timed_out = isinstance(scheduler._error, TimeoutError)
            _evaluate_invariant = _scheduler_run_evaluable(scheduler._error)
            if _deadlock_err is not None:
                with engine_lock:
                    schedule = execution.schedule_trace
                schedule_list = record_dpor_failure(
                    result,
                    list(schedule),
                    f"Deadlock detected after {result.num_explored} interleaving(s).\n\n"
                    f"{_deadlock_err.cycle_description}",
                )

                # Replay the counterexample to measure reproducibility
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    attempts, successes = _reproduce_dpor_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        threads=threads,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                        reproduce_on_failure=reproduce_on_failure,
                        lock_timeout=lock_timeout,
                        invariant=None,
                        detect_io=detect_io,
                        io_schedule=list(scheduler._io_trace) if detect_io and scheduler._io_trace else None,
                        patch_sleep=patch_sleep,
                        clock=clock,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes

                    from frontrun._preload_io import _set_preload_pipe_fd

                    if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
                        _set_preload_pipe_fd(preload_dispatcher._write_fd)

                if stop_on_first:
                    clear_instr_cache()
                    _record_and_emit_report(was_deadlock=True)
                    return result

            if warn_nondeterministic_sql:
                check_uncaptured_inserts()

            # --- error_on_any_race: treat unsynchronized races as failures ---
            if error_on_any_race and _evaluate_invariant:
                with engine_lock:
                    raw_races_check = engine.attribute_races()
                if raw_races_check:
                    with engine_lock:
                        schedule = execution.schedule_trace
                    record_dpor_failure(
                        result,
                        list(schedule),
                        format_race_failure_explanation(
                            result.num_explored,
                            len(raw_races_check),
                            actor_plural="threads",
                        ),
                        races_detected=True,
                    )
                    if stop_on_first:
                        clear_instr_cache()
                        _record_and_emit_report()
                        return result

            # --- serializable_invariant: check against sequential baselines ---
            if serial_valid_states is not None and _evaluate_invariant:
                ser_explanation = check_serializability_violation(
                    state, serial_valid_states, serial_hash_fn, result.num_explored
                )
                if ser_explanation is not None:
                    with engine_lock:
                        schedule = execution.schedule_trace
                    record_dpor_failure(result, list(schedule), ser_explanation)
                    if stop_on_first:
                        clear_instr_cache()
                        _record_and_emit_report()
                        return result

            if not _evaluate_invariant:
                invariant_failed, assertion_msg = False, None
            else:
                # The invariant runs on the driver thread; register the
                # execution's virtual clock so TTL-style reads see the same
                # (virtual) time the workers saw.
                with clock_scope(virtual_clock):
                    invariant_failed, assertion_msg = check_invariant(invariant, state)
            if invariant_failed:
                with engine_lock:
                    schedule = execution.schedule_trace
                # explanation=None defers message-setting: it depends on
                # reproduction counts computed just below.
                schedule_list = record_dpor_failure(result, list(schedule), None)

                # Replay the counterexample to measure reproducibility
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    # Access-anchored replay (defect #20): resolve the racing
                    # objects of this failing execution to run-stable labels
                    # and extract their recorded access order, so the bytecode
                    # replay can enforce the orderings that matter even when
                    # the positional schedule drifts (e.g. a real subprocess
                    # between the racing write and read).
                    with engine_lock:
                        _raced_keys = {r[3] for r in engine.pending_races() if r[3] is not None}
                    access_schedule = scheduler.racing_access_schedule(_raced_keys) if _raced_keys else None
                    attempts, successes = _reproduce_dpor_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        threads=threads,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                        reproduce_on_failure=reproduce_on_failure,
                        lock_timeout=lock_timeout,
                        invariant=invariant,
                        detect_io=detect_io,
                        io_schedule=list(scheduler._io_trace) if detect_io and scheduler._io_trace else None,
                        patch_sleep=patch_sleep,
                        access_schedule=access_schedule,
                        clock=clock,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes

                    # Re-enable pipe writes for subsequent DPOR executions.
                    from frontrun._preload_io import _set_preload_pipe_fd

                    if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
                        _set_preload_pipe_fd(preload_dispatcher._write_fd)

                if result.explanation is None:
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
                if result.sql_anomaly is None:
                    result.sql_anomaly = classify_sql_anomaly(recorder.events)
                if stop_on_first:
                    clear_instr_cache()
                    _record_and_emit_report()
                    return result

            # Clear instruction cache between executions to avoid stale code ids
            clear_instr_cache()

            # Collect report data before next_execution() consumes pending races
            if _collecting_report and report is not None:
                with engine_lock:
                    schedule_trace = list(execution.schedule_trace)
                    raw_races = engine.pending_races()
                race_info = _build_race_info(raw_races)
                was_deadlock = isinstance(scheduler._error, DeadlockError)
                # Check if this specific execution failed: it was appended to failures
                # with the current num_explored as its execution number
                this_exec_failed = any(n == result.num_explored for n, _ in result.failures)
                # A scheduler timeout means the run never completed under DPOR
                # control, so its invariant did not meaningfully "hold" (finding 5).
                invariant_held = not was_deadlock and not scheduler_timed_out and not this_exec_failed
                report.executions.append(
                    ExecutionRecord(
                        index=len(report.executions),
                        schedule_trace=schedule_trace,
                        switch_points=switch_points,
                        invariant_held=invariant_held,
                        was_deadlock=was_deadlock,
                        race_info=race_info,
                        step_events=scheduler._step_event_collector or {},
                        lock_events=scheduler._lock_event_collector or [],
                        deadlock_at=scheduler._deadlock_at,
                        deadlock_cycle_description=getattr(scheduler._error, "cycle_description", None)
                        if was_deadlock
                        else None,
                    )
                )

    finally:
        if trace_packages is not None:
            _set_active_trace_filter(None)
        set_lock_timeout(prev_lock_timeout)
        if preload_dispatcher is not None:
            preload_dispatcher.stop()
        set_object_key_reverse_map(None)

    # Generate HTML report if requested
    if report is not None and report_path is not None:
        generate_html_report(report, report_path)

    return result
