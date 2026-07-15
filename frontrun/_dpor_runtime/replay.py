# ruff: noqa: F403, F405
# pyright: reportUnusedFunction=false

from __future__ import annotations

from frontrun._dpor_core import is_reproduction_run
from frontrun._virtual_clock import ClockConfig, VirtualClock, clock_scope, validate_clock

from ._shared import *
from .runner import DporBytecodeRunner
from .scheduler import DporScheduler, _IOAnchoredReplayScheduler, _ReplayDporScheduler


def _run_dpor_schedule(
    schedule: list[int],
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    timeout: float = 5.0,
    detect_io: bool = False,
    deadlock_timeout: float = 5.0,
    trace_recorder: TraceRecorder | None = None,
    io_schedule: list[tuple[int, str]] | None = None,
    patch_sleep: bool = True,
    access_schedule: list[tuple[int, str, str]] | None = None,
    clock: str = "real",
    virtual_clock: VirtualClock | None = None,
) -> T:
    """Replay a DPOR schedule using the DPOR runner rather than OpcodeScheduler.

    When *io_schedule* is provided and *detect_io* is True, uses the
    IO-anchored replay scheduler (defect #16) which only enforces the
    schedule at IO boundaries, tolerating state-dependent changes in
    opcode-level scheduling points.

    When *access_schedule* is provided, the bytecode replay additionally
    gates accesses to the racing objects on the recorded access order
    (access-anchored replay, defect #20).

    When *clock* is not ``"real"``, the recorded schedule may contain
    clock-actor steps (id == ``len(threads)``); the replay schedulers
    perform the corresponding clock advances against *virtual_clock*.
    """
    config = ClockConfig(mode=validate_clock(clock))
    # Under clock="real" ignore any caller-supplied clock; otherwise reuse it or
    # mint a fresh one.  Folding this here removes the per-scheduler re-guards.
    virtual_clock = virtual_clock if config.active else None
    if config.active and virtual_clock is None:
        virtual_clock = config.new_clock()
    clock_actor_id = config.actor_id(len(threads))
    if io_schedule is not None and detect_io:
        scheduler: DporScheduler = _IOAnchoredReplayScheduler(
            io_schedule,
            len(threads),
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
            virtual_clock=virtual_clock,
            clock_mode=clock,
            clock_actor_id=clock_actor_id,
        )
    else:
        scheduler = _ReplayDporScheduler(
            schedule,
            len(threads),
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
            access_schedule=access_schedule,
            virtual_clock=virtual_clock,
            clock_mode=clock,
            clock_actor_id=clock_actor_id,
        )
    runner = DporBytecodeRunner(scheduler, detect_io=detect_io)

    # One clock_scope owns the time.* patch across setup + the worker phase;
    # the workers resolve their virtual clock via scheduler TLS.
    with clock_scope(virtual_clock), runner.patch_scope(patch_sleep=patch_sleep):
        state = setup()
        # Mirror exploration (explore.py): assign stable IDs to the fresh
        # setup() graph in deterministic order before any worker runs.  The
        # scheduler's StableObjectIds is freshly constructed (the analogue of
        # exploration's per-execution reset_execution_state), so setup-time
        # first-touch IDs and this walk reproduce exploration's numbering
        # exactly — access anchors embed these IDs to distinguish instances
        # of the same class (defect #22).
        scheduler._stable_ids.pre_register(state)

        def make_thread_func(thread_func: Callable[[T], None], thread_state: T) -> Callable[[], None]:
            def thread_wrapper() -> None:
                thread_func(thread_state)

            return thread_wrapper

        funcs: list[Callable[[], None]] = [make_thread_func(t, state) for t in threads]
        try:
            runner.run(funcs, timeout=timeout)
        except TimeoutError:
            # Exact deadlocks are represented by the scheduler error even
            # when worker teardown surfaces as a timeout.  Preserve that
            # stronger diagnosis, but never treat an ordinary timeout as a
            # completed replay with a partially mutated state.
            if isinstance(scheduler._error, DeadlockError):
                raise scheduler._error
            raise
        if runner.timed_out:
            # Even if timeout notification let the workers unwind during the
            # cleanup join, this replay exceeded its bound.  Its partial state
            # is not a reproduction, and another in-process replay would be
            # unsafe if any managed or unmanaged work survived.
            raise TimeoutError("DPOR replay timed out before all worker threads completed")
        if scheduler._error is not None:
            raise scheduler._error
    return state


def _reproduce_dpor_counterexample(
    *,
    schedule_list: list[int],
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    timeout_per_run: float,
    deadlock_timeout: float,
    reproduce_on_failure: int,
    lock_timeout: int | None,
    invariant: Callable[[T], bool] | None = None,
    detect_io: bool = True,
    io_schedule: list[tuple[int, str]] | None = None,
    patch_sleep: bool = True,
    access_schedule: list[tuple[int, str, str]] | None = None,
    clock: str = "real",
) -> tuple[int, int]:
    """Measure how often a DPOR counterexample reproduces under the DPOR runner.

    Reproduction runs with the same IO interception (SQL, Redis) as
    exploration so that the replay scheduler can enforce interleavings at
    IO boundaries, not just bytecode boundaries.

    When *io_schedule* is provided, replay is anchored to explicit I/O
    boundaries (defect #16) so state-dependent opcode paths do not
    desynchronise the schedule.
    """
    from frontrun._preload_io import _set_preload_pipe_fd
    from frontrun._redis_client import patch_redis, set_redis_replay_mode, unpatch_redis
    from frontrun._sql_cursor import get_lock_timeout, patch_sql, set_lock_timeout, unpatch_sql

    _set_preload_pipe_fd(-1)
    if reproduce_on_failure <= 0:
        return reproduce_on_failure, 0

    _prev_lt = get_lock_timeout()
    _replay_lock_timeout = lock_timeout if lock_timeout is not None else 5000
    set_lock_timeout(_replay_lock_timeout)
    patch_sql()
    patch_redis()
    set_redis_replay_mode(True)
    successes = 0
    attempts = 0
    clock_config = ClockConfig(mode=validate_clock(clock))
    try:
        for _ in range(reproduce_on_failure):
            attempts += 1
            deadlocked = False
            inv_failed = False
            replay_clock = clock_config.new_clock()
            try:
                replay_state = _run_dpor_schedule(
                    schedule_list,
                    setup,
                    threads,
                    timeout=timeout_per_run,
                    detect_io=detect_io,
                    deadlock_timeout=deadlock_timeout,
                    io_schedule=io_schedule,
                    patch_sleep=patch_sleep,
                    access_schedule=access_schedule,
                    clock=clock,
                    virtual_clock=replay_clock,
                )
                if invariant is not None:
                    # Use check_invariant so assert-style invariants (which
                    # raise AssertionError) are scored as failures, matching
                    # exploration (explore.py) and the async reproduction path.
                    # Evaluating the invariant raw here would let AssertionError
                    # escape into the broad ``except Exception: continue`` below,
                    # scoring every replay as a non-reproduction.
                    with clock_scope(replay_clock):
                        inv_failed, _ = check_invariant(invariant, replay_state)
            except DeadlockError:
                deadlocked = True
            except TimeoutError:
                # Python threads cannot be killed safely.  A timed-out replay
                # poisons the in-process harness, so never launch another
                # attempt after it.
                break
            except Exception:
                continue  # crash during replay — not a reproduction
            if is_reproduction_run(
                deadlocked=deadlocked, has_invariant=invariant is not None, invariant_failed=inv_failed
            ):
                successes += 1

    finally:
        set_redis_replay_mode(False)
        unpatch_redis()
        unpatch_sql()
        set_lock_timeout(_prev_lt)
    return attempts, successes


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
