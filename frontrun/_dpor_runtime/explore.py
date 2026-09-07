# ruff: noqa: F403, F405

from __future__ import annotations

from frontrun._certificate import PassEvidence, certify_pass
from frontrun._dpor_core import (
    compute_serializable_baseline_sync,
    dpor_exploration_iter,
    format_race_failure_explanation,
    make_deadline,
    make_dpor_engine,
    record_dpor_failure,
)
from frontrun._tracing import trace_filter_scope
from frontrun._virtual_clock import ClockConfig, ClockMode, clock_scope
from frontrun.common import _call_sync_setup, _reject_deferred_sync_result

from ._shared import *
from ._shared import _require_frontrun_env
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
    clock_config = ClockConfig(mode=clock, diagnostics=clock_diagnostics).validate(
        patch_sleep=patch_sleep,
        serializable_invariant=serializable_invariant,
    )
    clock = clock_config.mode
    # Compute serializable baseline if requested.
    with trace_filter_scope(trace_packages):
        serial_valid_states, serial_hash_fn = compute_serializable_baseline_sync(setup, threads, serializable_invariant)

    num_threads = len(threads)
    # With a virtual clock the engine gets one extra thread: the clock actor
    # (id == num_threads), whose steps advance the clock (see scheduler.py).
    clock_actor_id = clock_config.actor_id(num_threads)
    engine = make_dpor_engine(
        num_threads=num_threads + (1 if clock_actor_id is not None else 0),
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
        search=search,
    )

    # Verdict-less accumulator: the pass verdict is only stamped by
    # certify_pass() at the end, from evidence gathered below.
    result = InterleavingResult(property_holds=None)
    workers_entered = [False] * num_threads
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

    def _record_execution(*, emit: bool = False) -> None:
        """Record an execution, optionally emitting the report before an early return."""
        if report is None or report_path is None:
            return
        if not _collecting_report:
            if emit:
                generate_html_report(report, report_path)
            return
        with engine_lock:
            sched = list(execution.schedule_trace)
            races = engine.pending_races()
        was_deadlock = isinstance(scheduler._error, DeadlockError)
        this_exec_failed = bool(result.failures and result.failures[-1][0] == result.num_explored)
        report.executions.append(
            ExecutionRecord(
                index=len(report.executions),
                schedule_trace=sched,
                switch_points=switch_points,
                invariant_held=not was_deadlock and not this_exec_failed,
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
        if emit:
            generate_html_report(report, report_path)

    def _record_reproduction(
        schedule_list: list[int],
        invariant_fn: Callable[[T], bool] | None,
        access_schedule: list[tuple[int, str, str]] | None = None,
    ) -> None:
        result.reproduction_attempts, result.reproduction_successes = _reproduce_dpor_counterexample(
            schedule_list=schedule_list,
            setup=setup,
            threads=threads,
            timeout_per_run=timeout_per_run,
            deadlock_timeout=deadlock_timeout,
            reproduce_on_failure=reproduce_on_failure,
            lock_timeout=lock_timeout,
            invariant=invariant_fn,
            detect_io=detect_io,
            io_schedule=list(scheduler._io_trace) if detect_io and scheduler._io_trace else None,
            patch_sleep=patch_sleep,
            access_schedule=access_schedule,
            clock=clock,
        )
        # Replay disables pipe writes; restore them for subsequent executions.
        from frontrun._preload_io import _set_preload_pipe_fd

        if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
            _set_preload_pipe_fd(preload_dispatcher._write_fd)

    # Baseline threads are snapshotted ONCE for the whole exploration, not per
    # execution: a helper thread spawned by a worker during execution N (e.g. a
    # lazily-warmed pool) survives into execution N+1 and must still count as a
    # live *external* thread there — a per-execution snapshot would classify it
    # as inert baseline and break external-liveness reasoning (false exact
    # deadlocks / stalls under a virtual clock).
    baseline_threads = [t for t in threading.enumerate() if t.is_alive()]
    with trace_filter_scope(trace_packages):
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
                virtual_clock = clock_config.new_clock()
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
                    baseline_threads=baseline_threads,
                )
                runner = DporBytecodeRunner(scheduler, detect_io=detect_io, preload_bridge=preload_bridge)

                # Both scopes span setup/run/invariant: the invariant runs on the
                # driver thread under the same clock AND sleep/lock/io patches the
                # workers and setup() saw.  Ending patch_scope before evaluation
                # (as this once did) made a TTL-style invariant's time.sleep run
                # on the REAL wall clock while its time.* reads stayed frozen at
                # virtual time — a self-inconsistent clock (elapsed == 0.0 after
                # sleep(5)) costing real seconds per explored interleaving.  The
                # nested runs of _reproduce_dpor_counterexample are safe under the
                # held scope: every patch is reference-counted.
                with clock_scope(virtual_clock), runner.patch_scope(patch_sleep=patch_sleep):
                    state = _call_sync_setup(setup)
                    # Assign stable object IDs in deterministic, schedule-independent
                    # order *before* any worker runs.  Without this, IDs are assigned
                    # in first-touch order, which DPOR backtracks permute across
                    # executions — corrupting the Rust sleep-set / trace-cache
                    # comparison that carries object-ID-keyed state across executions
                    # (silently pruning genuinely distinct interleavings).
                    stable_ids.pre_register(state)

                    def make_thread_func(idx: int, thread_func: Callable[[T], None], s: T) -> Callable[[], None]:
                        def wrapper() -> None:
                            # Pass-certificate evidence: this worker's body was entered.
                            workers_entered[idx] = True
                            result = thread_func(s)
                            _reject_deferred_sync_result(result, thread_func)

                        return wrapper

                    funcs = [make_thread_func(i, t, state) for i, t in enumerate(threads)]
                    try:
                        runner.run(funcs, timeout=timeout_per_run)
                    except TimeoutError:
                        if runner.worker_originated_errors:
                            raise

                    result.num_explored += 1

                    # Fail closed on the coverage claim (issue #250): a worker
                    # blocked on a modeled row lock after its engine step was
                    # already committed (before_sync_retry in
                    # _sql_cursor._dpor_schedule_and_suppress_sync runs before
                    # acquire_row_locks can redirect to the holder), so the
                    # engine's schedule and the physical execution can diverge at
                    # row-lock boundaries and derived executions may be silently
                    # pruned. Thread mode normally leaves ``exhausted`` unset
                    # (None); demote it to an explicit False — sticky for the
                    # rest of this exploration since ``result`` is never reset —
                    # without touching property_holds/failure reporting.
                    if scheduler._row_lock_redirected:
                        result.exhausted = False

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
                    if scheduler_timed_out:
                        # Python threads cannot be terminated safely. Once a run
                        # times out, survivors may continue outside scheduler
                        # control and no later execution is trustworthy, so the
                        # search stops here and never claims full coverage.
                        clear_instr_cache()
                        result.exhausted = False
                        if result.property_holds is False:
                            # An earlier execution already proved a counterexample.
                            # The timeout bounds how much more of the space was
                            # searched; it does not unprove the failure.
                            return result
                        # Otherwise the timed-out partial run is neither a passing
                        # proof nor a counterexample: report it as inconclusive.
                        result.property_holds = None
                        result.inconclusive_reason = (
                            f"DPOR execution {result.num_explored} timed out before all worker threads completed. "
                            "The search is inconclusive because Python threads cannot be killed safely; increase "
                            "timeout_per_run/deadlock_timeout or remove unmanaged blocking from explored workers."
                        )
                        result.explanation = result.inconclusive_reason
                        return result
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
                        if reproduce_on_failure > 0 and result.reproduction_attempts == 0 and not runner.timed_out:
                            _record_reproduction(schedule_list, None)

                        if stop_on_first:
                            clear_instr_cache()
                            _record_execution(emit=True)
                            return result

                    if warn_nondeterministic_sql:
                        ensure_no_uncaptured_inserts()

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
                                _record_execution(emit=True)
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
                                _record_execution(emit=True)
                                return result

                    if not _evaluate_invariant:
                        invariant_failed, assertion_msg = False, None
                    else:
                        # The invariant runs on the driver thread under the enclosing
                        # clock_scope(virtual_clock), so TTL-style reads see the same
                        # (virtual) time the workers saw.
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
                            _record_reproduction(schedule_list, invariant, access_schedule)

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
                            _record_execution(emit=True)
                            return result

                    # Clear instruction cache between executions to avoid stale code ids
                    clear_instr_cache()

                    # Collect report data before next_execution() consumes pending races
                    _record_execution()

        finally:
            set_lock_timeout(prev_lock_timeout)
            if preload_dispatcher is not None:
                preload_dispatcher.stop()
            set_object_key_reverse_map(None)

    # Generate HTML report if requested
    if report is not None and report_path is not None:
        generate_html_report(report, report_path)

    if result.property_holds is False:
        # Failures recorded with stop_on_first=False fall through to here.
        return result
    # No failure found: certify (or honestly refuse to certify) the pass.
    # dpor_exploration_iter always yields the baseline execution, so a
    # zero-execution outcome can only mean the total_timeout budget expired
    # before it completed.
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
