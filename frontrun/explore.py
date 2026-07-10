"""Unified entry point for frontrun interleaving exploration.

This module provides :func:`explore`, a single function that dispatches to the
appropriate underlying implementation based on worker type and strategy.

Examples::

    import frontrun

    # Sync DPOR (default)
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()

    # Async — detected automatically from coroutine function
    async def worker(state): ...
    result = await frontrun.explore(setup=make_state, workers=worker, count=2, invariant=...)

    # Strategy selection
    result = frontrun.explore(..., strategy="dpor")    # default
    result = frontrun.explore(..., strategy="random")  # random schedule sampling
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from frontrun._strategy import ASYNC_STRATEGIES, STRATEGIES
from frontrun._virtual_clock import ClockMode, validate_clock_options
from frontrun.common import all_async, any_async

Strategy = Literal["dpor", "random"]
Execution = Literal["thread", "process"]
#: Alias of :data:`frontrun._virtual_clock.ClockMode` (single source of truth).
Clock = ClockMode

#: Options that change *which* bugs are found but are not honored under
#: ``execution="process"`` (state is external; there is no in-process opcode
#: trace, replay, or virtual clock).  An explicitly-passed value here is rejected
#: rather than silently ignored — a silent no-op is a correctness footgun when
#: porting a thread test.  Membership is matched against ``explicit_options``
#: (options whose value differs from the signature default).
_PROCESS_UNSUPPORTED_OPTIONS = frozenset(
    {
        # In-process trace / analysis knobs with no process-mode analog
        # (timeout_per_run's analog is deadlock_timeout).
        "serializable_invariant",
        "error_on_any_race",
        "lock_timeout",
        "trace_packages",
        "track_dunder_dict_accesses",
        "detect_sql",
        "detect_io",
        "patch_sleep",
        "timeout_per_run",
        "reproduce_on_failure",
        "warn_nondeterministic_sql",
        # Random-strategy-only knobs: process mode forces strategy='dpor'.
        "max_attempts",
        "max_ops",
        "seed",
        "debug",
        # The virtual clock lives in the in-process scheduler; worker processes
        # read real time, so a non-default value is a correctness footgun.
        "clock",
        "clock_diagnostics",
    }
)

#: The inverse of :data:`_PROCESS_UNSUPPORTED_OPTIONS`: options that only make
#: sense under ``execution="process"``.  Thread execution rejects an
#: explicitly-passed value here rather than silently ignoring it, by the same
#: no-silent-no-op principle.
_PROCESS_ONLY_OPTIONS = frozenset({"reuse_workers"})


def explore(
    setup: Callable[[], Any],
    workers: Callable[[Any], Any] | list[Callable[[Any], Any]] | tuple[Callable[[Any], Any], ...],
    invariant: Callable[[Any], bool],
    *,
    count: int | None = None,
    strategy: Strategy = "dpor",
    execution: Execution = "thread",
    # Time control (both strategies, sync + async)
    clock: Clock = "real",
    clock_diagnostics: bool = False,
    # DPOR-specific kwargs
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    detect_io: bool = True,
    reuse_workers: bool = False,
    deadlock_timeout: float | None = None,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
    trace_packages: list[str] | None = None,
    track_dunder_dict_accesses: bool = False,
    search: str | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[Any], Any] | bool = False,
    error_on_any_race: bool = False,
    # Random-specific kwargs
    max_attempts: int = 200,
    max_ops: int | None = None,
    seed: int | None = None,
    debug: bool = False,
    # Async-specific kwargs (kept for passthrough)
    detect_sql: bool = False,
) -> Any:
    """Explore thread/task interleavings for concurrency bugs.

    A unified entry point that dispatches to the appropriate underlying
    implementation based on worker type (sync vs async) and strategy.

    Args:
        setup: Creates fresh shared state for each execution.
        workers: A callable (when ``count`` is provided) or a list/tuple of
            callables. Sync callables run as threads; async callables (coroutine
            functions) run as asyncio tasks.
        invariant: Predicate over shared state; must be True after all
            workers complete.
        count: When ``workers`` is a single callable, replicate it this many
            times. Must be positive. Cannot be used when ``workers`` is a
            list/tuple.
        strategy: ``"dpor"`` (default) for systematic DPOR exploration, or
            ``"random"`` for random schedule sampling.
        execution: ``"thread"`` (default) runs workers as threads/async tasks in
            this process; ``"process"`` runs each worker in its own spawned
            Python process, coordinating over a socket. Process mode has the same
            ``setup`` / ``workers`` / ``invariant`` / ``count`` shape; workers and
            the ``setup()`` return value are serialised with dill (so closures and
            lambdas work, not just module-level functions), and ``setup()`` should
            return a handle to external SQL/Redis state (a DB path/URL, not a live
            connection). Supports ``strategy="dpor"`` with sync workers only and
            needs the ``process`` extra (``pip install frontrun[process]``). See
            :doc:`/cross_process`.
        reuse_workers: Process execution only. Spawn each worker process once and
            re-run it per interleaving instead of respawning (amortises spawn
            cost). Thread execution rejects an explicit ``reuse_workers=True``
            with ``ValueError`` (there are no worker processes to reuse).
        max_executions: Safety limit on total executions (DPOR only).
        preemption_bound: Limit on preemptions per execution (DPOR only).
        max_branches: Maximum scheduling points per execution (DPOR only).
        timeout_per_run: Timeout for each individual run.
        stop_on_first: Stop on first invariant violation (DPOR only).
        detect_io: Detect socket/file I/O operations as resource accesses.
            For async DPOR, also activates Redis key-level patching. For the
            async *random* strategy the flag is narrower: it only gates SQL
            driver patching (``detect_io=True`` implies ``detect_sql=True``);
            socket/file/Redis detection is not available on that path. Note
            the difference from the standalone
            :func:`frontrun.explore_async_random`, whose ``detect_sql``
            defaults to ``False`` — going through ``explore(strategy="random")``
            with async workers patches SQL drivers by default because
            ``detect_io`` defaults to ``True`` here.
        deadlock_timeout: Seconds to wait before declaring a deadlock. Defaults
            to 5.0 for thread execution and 15.0 for process execution (spawning
            processes is slower), unless set explicitly.
        reproduce_on_failure: Replay counterexample this many times (not
            supported for async random).
        total_timeout: Maximum total exploration time in seconds.
        warn_nondeterministic_sql: Raise on nondeterministic SQL INSERT (not
            supported for async random).
        lock_timeout: Auto-set PostgreSQL lock_timeout (milliseconds; DPOR
            only).
        trace_packages: Package patterns to trace in addition to user code.
        track_dunder_dict_accesses: Report ``obj.__dict__`` accesses (sync
            DPOR only).
        search: Wakeup-tree traversal strategy (sync DPOR and process
            execution only).
        patch_sleep: For ``clock="real"``, make ``time.sleep`` /
            ``asyncio.sleep`` cooperative zero-wall-time yields. For
            ``clock="virtual"`` or ``"explored"``, required: positive sleeps
            become scheduler-owned virtual deadlines and ``sleep(0)`` remains a
            yield.
        serializable_invariant: Check serializability against sequential runs.
        error_on_any_race: Treat unsynchronized races as failures (DPOR
            only; the random strategies reject ``True`` with their own
            ValueError rather than silently ignoring it).
        clock: ``"real"`` (default) leaves time untouched. ``"virtual"`` gives
            each execution a scheduler-owned virtual clock: explored code reads
            virtual time from ``time.time()`` / ``time.monotonic()`` /
            ``time.perf_counter()``, sleeps become zero-wall-time virtual
            deadlines, timed lock acquires time out deterministically, and the
            clock autojumps to the earliest pending deadline when nothing is
            runnable. ``"explored"`` additionally makes the clock advance a
            schedulable choice, so timer firings are explored against other
            operations ("the retry fired between the read and the write").
            Rule of thumb: use ``"virtual"`` to make timeout/TTL logic
            reachable deterministically at zero wall cost; add ``"explored"``
            when the *timing* of a timer firing is itself the race you are
            hunting. Works with both strategies, sync and async. Requires
            ``patch_sleep=True``; not supported with ``execution="process"``
            (worker processes read real time) or ``serializable_invariant``
            (the sequential baseline runs outside the scheduler). See
            :doc:`/virtual_clock`.
        clock_diagnostics: When using a virtual clock, warn when traced worker
            frames hold references to real ``time.*`` functions captured before
            frontrun patched the time module. Diagnostics do not change
            scheduling behavior. Requires frame tracing: DPOR and sync random
            can emit diagnostics; async random accepts the option for API
            compatibility but cannot inspect frames.
        max_attempts: Random schedule samples to try (random strategy only).
        max_ops: Maximum schedule length per attempt (random strategy only).
        seed: RNG seed for reproducibility (random strategy only).
        debug: Enable debug output (sync random only).
        detect_sql: Patch async SQL drivers (async workers only;
            ``detect_io=True`` already implies it).

    Returns:
        :class:`~frontrun.common.InterleavingResult` (sync) or a coroutine
        that resolves to one (async workers).

    Raises:
        ValueError: If ``count`` and a list of workers are both provided,
            ``count <= 0``, ``workers`` mixes async and sync callables,
            ``strategy``, ``execution`` or ``clock`` is
            unrecognised, or a non-real ``clock`` is combined with
            ``patch_sleep=False``, ``serializable_invariant``, or
            ``execution="process"``. Also raised for any explicitly-passed
            option the selected strategy/mode does not support, rather than
            silently ignoring it: e.g. ``seed=`` with ``strategy="dpor"``,
            ``preemption_bound=`` with ``strategy="random"``,
            ``reproduce_on_failure=`` with async ``strategy="random"``,
            ``detect_sql=`` with sync workers, or ``reuse_workers=`` with
            ``execution="thread"``. ``execution="process"``
            additionally rejects async workers, ``strategy="random"``, and
            every option that requires the in-process scheduler
            (``serializable_invariant``, ``error_on_any_race``,
            ``lock_timeout``, ``trace_packages``,
            ``track_dunder_dict_accesses``, ``detect_sql``, non-real
            ``clock`` / ``clock_diagnostics``, and non-default ``detect_io``,
            ``patch_sleep``, ``timeout_per_run``, ``reproduce_on_failure``,
            ``warn_nondeterministic_sql``, ``max_attempts``, ``max_ops``,
            ``seed``, ``debug``). Explicit-option detection is value-based:
            passing an option at its default value is indistinguishable from
            omitting it, and is accepted (a no-op either way).
    """
    worker_list = _resolve_workers(workers, count)

    # A mixed worker list has no single execution model: any_async() would
    # route the whole list to the async engine and the sync workers would be
    # silently mishandled instead of running as threads. Reject it eagerly.
    if any_async(worker_list) and not all_async(worker_list):
        raise ValueError(
            "explore(): workers mix async and sync callables; frontrun explores one execution model at a time. "
            "Make every worker `async def` (run as asyncio tasks) or every worker a plain `def` (run as threads)."
        )

    validate_clock_options(
        clock,
        patch_sleep=patch_sleep,
        serializable_invariant=serializable_invariant,
        clock_diagnostics=clock_diagnostics,
    )

    # Every strategy-tunable keyword argument the user could have passed.  The
    # chosen adapter filters this dict against its own ``allowed_keys`` set;
    # anything *explicitly* passed outside that set is rejected below.
    all_kwargs: dict[str, Any] = {
        "max_executions": max_executions,
        "preemption_bound": preemption_bound,
        "max_branches": max_branches,
        "timeout_per_run": timeout_per_run,
        "stop_on_first": stop_on_first,
        "detect_io": detect_io,
        "deadlock_timeout": deadlock_timeout,
        "reproduce_on_failure": reproduce_on_failure,
        "total_timeout": total_timeout,
        "warn_nondeterministic_sql": warn_nondeterministic_sql,
        "lock_timeout": lock_timeout,
        "trace_packages": trace_packages,
        "track_dunder_dict_accesses": track_dunder_dict_accesses,
        "search": search,
        "patch_sleep": patch_sleep,
        "serializable_invariant": serializable_invariant,
        "error_on_any_race": error_on_any_race,
        "clock": clock,
        "clock_diagnostics": clock_diagnostics,
        "max_attempts": max_attempts,
        "max_ops": max_ops,
        "seed": seed,
        "debug": debug,
        "detect_sql": detect_sql,
        "reuse_workers": reuse_workers,
    }
    # Snapshot which options were explicitly passed (differ from the signature
    # default) *before* any local resolution mutates them.  An option passed at
    # its default value is indistinguishable from an omitted one, which is
    # acceptable: passing the default explicitly is a no-op either way.
    defaults = explore.__kwdefaults__ or {}
    explicit_options = frozenset(
        name for name, value in all_kwargs.items() if not (value is defaults[name] or value == defaults[name])
    )

    # A deadlock_timeout left unset resolves per execution mode: process spawn is
    # slow, so it gets a longer default than in-process threads.
    if deadlock_timeout is None:
        deadlock_timeout = 15.0 if execution == "process" else 5.0
    all_kwargs["deadlock_timeout"] = deadlock_timeout

    # Cross-process execution: each worker runs in its own Python process,
    # coordinating over a socket. Same call shape as threads/async; workers and
    # the setup() state must be picklable, and state is external (SQL/Redis).
    if execution == "process":
        if any_async(worker_list):
            raise ValueError(
                "explore(): async workers are not supported with execution='process' "
                "(worker processes run sync code only; use sync workers or execution='thread')"
            )
        if strategy != "dpor":
            raise ValueError(
                f"explore(): strategy={strategy!r} is not supported with execution='process' "
                "(process mode always drives the DPOR engine; drop strategy= or use execution='thread')"
            )
        unsupported = sorted(explicit_options & _PROCESS_UNSUPPORTED_OPTIONS)
        if unsupported:
            names = ", ".join(f"{name}=" for name in unsupported)
            verb = "is" if len(unsupported) == 1 else "are"
            raise ValueError(
                f"explore(): {names} {verb} not supported with execution='process' "
                "(a silently ignored option is a correctness footgun; these require the in-process scheduler, "
                "so drop them or use execution='thread')"
            )
        from frontrun.cross_process import _explore_process

        return _explore_process(
            setup,
            worker_list,
            invariant,
            deadlock_timeout=deadlock_timeout,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            max_branches=max_branches,
            total_timeout=total_timeout,
            stop_on_first=stop_on_first,
            search=search,
            reuse_workers=reuse_workers,
        )
    if execution != "thread":
        raise ValueError(f"explore(): unknown execution={execution!r}; must be 'thread' or 'process'")

    # Mirror of the process branch: an explicitly-passed process-only option
    # would be a silent no-op under thread execution, so reject it.
    process_only = sorted(explicit_options & _PROCESS_ONLY_OPTIONS)
    if process_only:
        names = ", ".join(f"{name}=" for name in process_only)
        verb = "is" if len(process_only) == 1 else "are"
        raise ValueError(
            f"explore(): {names} {verb} not supported with execution='thread' "
            "(a silently ignored option is a correctness footgun; these only apply to spawned worker processes, "
            "so drop them or use execution='process')"
        )

    is_async = any_async(worker_list)
    registry = ASYNC_STRATEGIES if is_async else STRATEGIES
    if strategy not in registry:
        valid = ", ".join(repr(k) for k in sorted(registry))
        raise ValueError(f"explore(): unknown strategy={strategy!r}; must be one of {valid}")

    # Mirror the process branch's principle for thread execution: an option the
    # selected strategy would silently drop is a correctness footgun, so reject
    # any explicitly-passed option outside the adapter's ``allowed_keys``.
    allowed_keys = getattr(registry[strategy], "allowed_keys", None)
    if allowed_keys is not None:
        _reject_unsupported_strategy_options(strategy, explicit_options - allowed_keys, is_async=is_async)

    if is_async:
        async_adapter = ASYNC_STRATEGIES[strategy]
        return async_adapter.run(setup=setup, workers=worker_list, invariant=invariant, **all_kwargs)
    sync_adapter = STRATEGIES[strategy]
    return sync_adapter.run(setup=setup, workers=worker_list, invariant=invariant, **all_kwargs)


def _reject_unsupported_strategy_options(strategy: str, unsupported: frozenset[str], *, is_async: bool) -> None:
    """Raise ValueError for explicitly-passed options the strategy ignores.

    ``unsupported`` holds the explicitly-passed option names outside the
    selected adapter's ``allowed_keys``.  Each is annotated with what *does*
    support it — another strategy in the same (sync/async) registry, or the
    other worker kind — derived from the adapters' ``allowed_keys`` so the
    error stays in sync with the actual plumbing.
    """
    if not unsupported:
        return
    same_registry = ASYNC_STRATEGIES if is_async else STRATEGIES
    other_registry = STRATEGIES if is_async else ASYNC_STRATEGIES
    parts: list[str] = []
    for name in sorted(unsupported):
        supporters = sorted(
            key for key, adapter in same_registry.items() if name in getattr(adapter, "allowed_keys", frozenset())
        )
        if supporters:
            parts.append(f"{name}= ({', '.join(supporters)} only)")
        elif any(name in getattr(adapter, "allowed_keys", frozenset()) for adapter in other_registry.values()):
            parts.append(f"{name}= ({'sync' if is_async else 'async'} workers only)")
        else:
            parts.append(f"{name}=")
    kind = "async" if is_async else "sync"
    verb = "is" if len(parts) == 1 else "are"
    raise ValueError(
        f"explore(): {', '.join(parts)} {verb} not supported with strategy={strategy!r} and {kind} workers "
        "(a silently ignored option is a correctness footgun; drop it or switch strategy)"
    )


def _resolve_workers(
    workers: Callable[[Any], Any] | list[Callable[[Any], Any]] | tuple[Callable[[Any], Any], ...],
    count: int | None,
) -> list[Callable[[Any], Any]]:
    """Expand workers + count into a concrete list."""
    if isinstance(workers, (list, tuple)):
        if count is not None:
            raise ValueError(
                "explore(): 'count' cannot be used when 'workers' is already a list or tuple. "
                "Either pass a single callable with count=N, or pass a list without count."
            )
        worker_list = list(workers)
    else:
        if count is not None:
            if count <= 0:
                raise ValueError(f"explore(): count must be a positive integer, got {count!r}")
            worker_list = [workers] * count
        else:
            worker_list = [workers]

    if not worker_list:
        raise ValueError("explore(): workers list is empty")

    return worker_list
