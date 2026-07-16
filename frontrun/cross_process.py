"""Public API for cross-process exploration.

Deterministically interleaves separate OS processes contending on shared
external SQL/Redis state, scheduling at external-access granularity. See the
cross-process exploration guide in the frontrun documentation.

Example::

    import frontrun

    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess("myapp.checkout:run", ("order-1",)),
            "w1": frontrun.Subprocess("myapp.checkout:run", ("order-1",)),
        },
        setup=reset_db,                       # runs in this process; returns a state handle
        invariant=lambda state: stock() >= 0,  # receives setup()'s handle; reads the DB here
    )
    if not result.ok:
        raise AssertionError(result.failure)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator, CrossProcessResult
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.launch import MpLauncher, Subprocess, SubprocessLauncher
from frontrun.common import _call_sync_setup, check_invariant

__all__ = ["CrossProcessResult", "Subprocess", "explore_processes"]


def _validate_positive(name: str, value: int | float, *, api: str = "explore_processes()") -> None:
    if value <= 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"{api}: {name} must be positive and finite, got {value!r}")


def _validate_dpor_bounds(
    *,
    deadlock_timeout: float,
    max_executions: int | None,
    preemption_bound: int | None,
    max_branches: int,
    total_timeout: float | None,
    api: str = "explore_processes()",
) -> None:
    _validate_positive("deadlock_timeout", deadlock_timeout, api=api)
    if max_executions is not None:
        _validate_positive("max_executions", max_executions, api=api)
    if preemption_bound is not None and preemption_bound < 0:
        raise ValueError(f"{api}: preemption_bound must be non-negative or None, got {preemption_bound!r}")
    _validate_positive("max_branches", max_branches, api=api)
    if total_timeout is not None:
        _validate_positive("total_timeout", total_timeout, api=api)


def _to_interleaving_result(result: CrossProcessResult) -> Any:
    """Map a CrossProcessResult onto the InterleavingResult explore() returns.

    Keeps ``execution="process"`` result-compatible with threads/async
    (``property_holds`` / ``counterexample`` / ``explanation`` / ``assert_holds``).
    An ``ok`` result is routed through :func:`frontrun._certificate.certify_pass`
    with the coordinator's real evidence (iterations, per-worker DONE frames,
    the exhausted/divergence claim and any truncation cause), so a zero-iteration
    truncation surfaces as inconclusive rather than a vacuous pass.
    """
    from frontrun._certificate import PassEvidence, certify_pass
    from frontrun.common import InterleavingResult

    if result.ok is not False:
        return certify_pass(
            result=InterleavingResult(
                property_holds=None,
                num_explored=result.iterations,
                unique_interleavings=result.iterations,
                exhausted=result.exhausted,
            ),
            evidence=PassEvidence(
                executions=result.iterations,
                workers_executed=result.workers_executed,
                exhausted_claim=result.exhausted,
                vacuous_reason=(
                    f"cross-process exploration completed no interleavings: {result.truncation}; "
                    "increase the budget or reduce the workload"
                )
                if result.truncation
                else (
                    "cross-process exploration completed no interleavings before the search budget expired; "
                    "increase total_timeout/max_iterations or reduce the workload"
                ),
            ),
        )
    kind = f"[{result.failure_kind}] " if result.failure_kind else ""
    where = f" at schedule {result.failing_schedule}" if result.failing_schedule is not None else ""
    explanation = f"{kind}{result.failure or 'invariant violated'}{where}"
    # Surface the external-access trace so a process-mode failure is
    # diagnosable without dropping down to explore_processes().
    if result.accesses:
        trace = ", ".join(
            f"{result.worker_labels.get(wid, f'w{wid}')}:{access}:{rid}" for wid, rid, access in result.accesses
        )
        explanation += f"\n  accesses: {trace}"
    return InterleavingResult(
        property_holds=False,
        counterexample=result.failing_schedule,
        num_explored=result.iterations,
        unique_interleavings=result.iterations,
        failures=[(num, list(schedule)) for num, schedule in result.failures],
        explanation=explanation,
        exhausted=result.exhausted,
        failure_kind=result.failure_kind,
    )


def _state_threaded_hooks(
    setup: Callable[[], Any],
    invariant: Callable[[Any], bool],
) -> tuple[Callable[[], None], Callable[[], bool], dict[str, Any]]:
    """Bridge the state-threaded public hooks onto the nullary coordinator contract.

    The cross-process coordinators call ``setup()`` / ``invariant()`` with no
    arguments, but the public API threads ``setup()``'s return value into
    ``invariant(state)`` (mirroring thread/async mode). This captures the handle
    ``setup()`` returns and feeds it to ``invariant`` on each check. The box is
    returned too so callers that also need the handle elsewhere (e.g. to hand it
    to worker processes) can read the same captured value.
    """
    state_box: dict[str, Any] = {}

    def coord_setup() -> None:
        state_box["state"] = _call_sync_setup(setup)

    def coord_invariant() -> bool:
        failed, _message = check_invariant(invariant, state_box.get("state"))
        return not failed

    return coord_setup, coord_invariant, state_box


def _explore_process(  # pyright: ignore[reportUnusedFunction]  # imported lazily by frontrun.explore
    setup: Callable[[], Any],
    workers: list[Callable[[Any], Any]],
    invariant: Callable[[Any], bool],
    *,
    deadlock_timeout: float = 15.0,
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    total_timeout: float | None = None,
    stop_on_first: bool = True,
    search: str | None = None,
    reuse_workers: bool = False,
) -> Any:
    """Back the ``frontrun.explore(execution="process")`` path.

    Runs each worker callable in its own ``multiprocessing`` process. ``setup()``
    returns a *picklable* handle to the shared external state (e.g. a DB URL);
    it is passed to every ``worker(state)`` and to ``invariant(state)``. Both
    ``setup`` and ``invariant`` run in this (coordinator) process. With
    ``reuse_workers`` the processes are spawned once and re-run per interleaving.
    """
    _validate_dpor_bounds(
        deadlock_timeout=deadlock_timeout,
        max_executions=max_executions,
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        total_timeout=total_timeout,
        api="explore()",
    )
    # Workers also need setup()'s handle (via state_fn); read it from the same box
    # the helper captures it into, so invariant(state) and the workers agree.
    coord_setup, coord_invariant, state_box = _state_threaded_hooks(setup, invariant)

    coordinator = DporCrossProcessCoordinator(
        num_workers=len(workers),
        deadlock_timeout=deadlock_timeout,
        max_executions=max_executions,
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        total_timeout=total_timeout,
        stop_on_first=stop_on_first,
        search=search,
        reuse_workers=reuse_workers,
    )
    worker_set = MpLauncher(workers, state_fn=lambda: state_box.get("state"), reuse=reuse_workers)
    result = coordinator.explore(worker_set=worker_set, setup=coord_setup, invariant=coord_invariant)
    return _to_interleaving_result(result)


def _resolve_specs(
    processes: Mapping[str, Subprocess] | Sequence[Subprocess] | Subprocess,
    count: int | None,
) -> list[Subprocess]:
    """Expand ``processes`` (+ optional ``count``) into a concrete list of specs.

    A single :class:`Subprocess` with ``count=N`` replicates it N times (the
    process-side mirror of ``explore(workers=fn, count=N)``); a mapping or
    sequence is taken as-is and forbids ``count``.
    """
    if isinstance(processes, Subprocess):
        if count is None:
            return [processes]
        if count <= 0:
            raise ValueError(f"explore_processes(): count must be a positive integer, got {count!r}")
        return [processes] * count
    if count is not None:
        raise ValueError(
            "explore_processes(): 'count' can only be used with a single Subprocess; "
            "pass a mapping/sequence without count, or one Subprocess with count=N."
        )
    return list(processes.values()) if isinstance(processes, Mapping) else list(processes)


def _worker_labels(processes: Mapping[str, Subprocess] | Sequence[Subprocess] | Subprocess) -> dict[int, str]:
    """Preserve mapping labels alongside the engine's dense numeric ids."""
    return dict(enumerate(processes)) if isinstance(processes, Mapping) else {}


def explore_processes(
    processes: Mapping[str, Subprocess] | Sequence[Subprocess] | Subprocess,
    *,
    setup: Callable[[], Any],
    invariant: Callable[[Any], bool],
    count: int | None = None,
    strategy: Literal["dpor", "exhaustive"] = "dpor",
    max_iterations: int = 4096,
    max_steps_per_run: int = 100_000,
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    total_timeout: float | None = None,
    stop_on_first: bool = True,
    search: str | None = None,
    deadlock_timeout: float = 15.0,
    reuse_workers: bool = False,
) -> CrossProcessResult:
    """Explore interleavings of *processes* contending on shared external state.

    ``processes`` is a mapping of label → :class:`Subprocess` (preserved as
    ``result.worker_labels``), a plain sequence, or a single
    :class:`Subprocess` with ``count=N`` to replicate it (the mirror of
    ``explore(workers=fn, count=N)``).

    ``setup`` resets the external state before each interleaving and returns a
    handle to it (e.g. a DB URL / connection info). That handle is passed to
    ``invariant(state)``, which checks the state afterwards and returns a bool —
    matching ``explore(execution="process")``. Both run in this (coordinator)
    process; ``invariant`` may ignore ``state`` and read the shared store directly.

    ``strategy``:

    * ``"dpor"`` (default) drives the Rust DPOR engine, pruning equivalent
      interleavings (partial-order reduction) and detecting cross-worker
      ``SELECT FOR UPDATE`` deadlocks. ``max_executions`` / ``preemption_bound``
      / ``max_branches`` / ``total_timeout`` bound the search, ``search``
      selects the wakeup-tree traversal order, and ``stop_on_first=False``
      keeps exploring after a failure, accumulating every failing execution in
      ``CrossProcessResult.failures``. ``exhausted=True`` (full coverage)
      requires ``preemption_bound=None``; the default bound (2) truncates the
      search, so bounded runs report ``exhausted=False``.
    * ``"exhaustive"`` enumerates every interleaving at external-access
      granularity, bounded by ``max_iterations`` and ``max_steps_per_run``
      per execution. Useful as a reduction-free cross-check.

    Each strategy rejects the other's bounds when passed explicitly (a
    silently ignored option is a correctness footgun): ``max_iterations`` is
    exhaustive-only; the DPOR knobs above are DPOR-only. Explicit-option
    detection is value-based, so passing a knob at its default value is
    indistinguishable from omitting it.
    """
    specs = _resolve_specs(processes, count)
    worker_labels = _worker_labels(processes)
    if not specs:
        raise ValueError("explore_processes requires at least one Subprocess")
    _validate_positive("deadlock_timeout", deadlock_timeout)
    if reuse_workers and strategy == "exhaustive":
        raise ValueError("explore_processes(): reuse_workers is not supported with strategy='exhaustive'")
    # Coordinators call setup()/invariant() nullary; thread setup()'s handle into
    # invariant(state) here to match explore(execution="process").
    coord_setup, coord_invariant, _ = _state_threaded_hooks(setup, invariant)
    if strategy == "dpor":
        # Exhaustive-only knobs must not silently no-op: the DPOR branch never
        # reads max_iterations (4096 is the signature default; anything else
        # was explicit).
        if max_iterations != 4096:
            raise ValueError(
                "explore_processes(): max_iterations only applies to strategy='exhaustive'; "
                "bound strategy='dpor' with max_executions instead."
            )
        if max_steps_per_run != 100_000:
            raise ValueError("explore_processes(): max_steps_per_run only applies to strategy='exhaustive'.")
        _validate_dpor_bounds(
            deadlock_timeout=deadlock_timeout,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            max_branches=max_branches,
            total_timeout=total_timeout,
        )
        result = DporCrossProcessCoordinator(
            num_workers=len(specs),
            deadlock_timeout=deadlock_timeout,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            max_branches=max_branches,
            total_timeout=total_timeout,
            stop_on_first=stop_on_first,
            search=search,
            reuse_workers=reuse_workers,
        ).explore(
            worker_set=SubprocessLauncher(specs, reuse=reuse_workers), setup=coord_setup, invariant=coord_invariant
        )
        return replace(result, worker_labels=worker_labels)
    if strategy == "exhaustive":
        # DPOR-only knobs must not silently no-op (same principle as the
        # rejected in-process-only explore() options): the exhaustive
        # coordinator has no engine to bound.
        if max_executions is not None:
            raise ValueError(
                "explore_processes(): max_executions only applies to strategy='dpor'; "
                "bound strategy='exhaustive' with max_iterations instead."
            )
        if preemption_bound != 2:  # 2 is the signature default; anything else was explicit
            raise ValueError("explore_processes(): preemption_bound only applies to strategy='dpor'.")
        if max_branches != 100_000:  # 100_000 is the signature default; anything else was explicit
            raise ValueError("explore_processes(): max_branches only applies to strategy='dpor'.")
        if total_timeout is not None:
            raise ValueError(
                "explore_processes(): total_timeout only applies to strategy='dpor'; "
                "bound strategy='exhaustive' with max_iterations instead."
            )
        if stop_on_first is not True:
            raise ValueError(
                "explore_processes(): stop_on_first only applies to strategy='dpor'; "
                "the exhaustive coordinator always stops at the first failure."
            )
        if search is not None:
            raise ValueError("explore_processes(): search only applies to strategy='dpor'.")
        _validate_positive("max_iterations", max_iterations)
        _validate_positive("max_steps_per_run", max_steps_per_run)
        result = CrossProcessCoordinator(
            num_workers=len(specs),
            max_steps_per_run=max_steps_per_run,
            deadlock_timeout=deadlock_timeout,
        ).explore(
            worker_set=SubprocessLauncher(specs),
            setup=coord_setup,
            invariant=coord_invariant,
            max_iterations=max_iterations,
        )
        return replace(result, worker_labels=worker_labels)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'dpor' or 'exhaustive'")
