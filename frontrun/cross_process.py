"""Public API for cross-process exploration (Phase 1).

Deterministically interleaves separate OS processes contending on shared
external (SQL) state, scheduling at external-access granularity. See
``ideas/cross_process_exploration.md``.

Example::

    import frontrun

    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess("myapp.checkout:run", ("order-1",)),
            "w1": frontrun.Subprocess("myapp.checkout:run", ("order-1",)),
        },
        setup=reset_db,                  # runs in this process between iterations
        invariant=lambda: stock() >= 0,  # reads the DB in this process
    )
    if not result.ok:
        raise AssertionError(result.failure)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator, CrossProcessResult
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.launch import MpLauncher, Subprocess, SubprocessLauncher

__all__ = ["CrossProcessResult", "Subprocess", "explore_processes"]


def _to_interleaving_result(result: CrossProcessResult) -> Any:
    """Map a CrossProcessResult onto the InterleavingResult explore() returns.

    Keeps ``execution="process"`` result-compatible with threads/async
    (``property_holds`` / ``counterexample`` / ``explanation`` / ``assert_holds``).
    """
    from frontrun.common import InterleavingResult

    explanation = None
    if not result.ok:
        kind = f"[{result.failure_kind}] " if result.failure_kind else ""
        where = f" at schedule {result.failing_schedule}" if result.failing_schedule is not None else ""
        explanation = f"{kind}{result.failure or 'invariant violated'}{where}"
        # Surface the external-access trace so a process-mode failure is
        # diagnosable without dropping down to explore_processes().
        if result.accesses:
            trace = ", ".join(f"w{wid}:{access}:{rid}" for wid, rid, access in result.accesses)
            explanation += f"\n  accesses: {trace}"
    return InterleavingResult(
        property_holds=result.ok,
        counterexample=result.failing_schedule,
        num_explored=result.iterations,
        unique_interleavings=result.iterations,
        explanation=explanation,
    )


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
    **_ignored: Any,
) -> Any:
    """Back the ``frontrun.explore(execution="process")`` path.

    Runs each worker callable in its own ``multiprocessing`` process. ``setup()``
    returns a *picklable* handle to the shared external state (e.g. a DB URL);
    it is passed to every ``worker(state)`` and to ``invariant(state)``. Both
    ``setup`` and ``invariant`` run in this (coordinator) process. With
    ``reuse_workers`` the processes are spawned once and re-run per interleaving.
    """
    state_box: dict[str, Any] = {}

    def coord_setup() -> None:
        state_box["state"] = setup()

    def coord_invariant() -> bool:
        return bool(invariant(state_box.get("state")))

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
    launcher = MpLauncher(workers, state_fn=lambda: state_box.get("state"), reuse=reuse_workers)
    result = coordinator.explore(launch=launcher, setup=coord_setup, invariant=coord_invariant)
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


def explore_processes(
    processes: Mapping[str, Subprocess] | Sequence[Subprocess] | Subprocess,
    *,
    setup: Callable[[], Any],
    invariant: Callable[[], bool],
    count: int | None = None,
    strategy: Literal["dpor", "exhaustive"] = "dpor",
    max_iterations: int = 4096,
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    deadlock_timeout: float = 15.0,
    reuse_workers: bool = False,
) -> CrossProcessResult:
    """Explore interleavings of *processes* contending on shared external state.

    ``processes`` is a mapping of label → :class:`Subprocess` (labels are for
    readability), a plain sequence, or a single :class:`Subprocess` with
    ``count=N`` to replicate it (the mirror of ``explore(workers=fn, count=N)``).
    ``setup`` resets the external state before each interleaving and ``invariant``
    checks it afterwards; both run in this (coordinator) process.

    ``strategy``:

    * ``"dpor"`` (default) drives the Rust DPOR engine, pruning equivalent
      interleavings (partial-order reduction) and detecting cross-worker
      ``SELECT FOR UPDATE`` deadlocks. ``max_executions`` / ``preemption_bound``
      tune the search.
    * ``"exhaustive"`` brute-forces every interleaving at external-access
      granularity, bounded by ``max_iterations``. Useful as a reduction-free
      cross-check.
    """
    specs = _resolve_specs(processes, count)
    if not specs:
        raise ValueError("explore_processes requires at least one Subprocess")
    if reuse_workers and strategy == "exhaustive":
        raise ValueError("explore_processes(): reuse_workers is not supported with strategy='exhaustive'")
    if strategy == "dpor":
        return DporCrossProcessCoordinator(
            num_workers=len(specs),
            deadlock_timeout=deadlock_timeout,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            reuse_workers=reuse_workers,
        ).explore(launch=SubprocessLauncher(specs, reuse=reuse_workers), setup=setup, invariant=invariant)
    if strategy == "exhaustive":
        return CrossProcessCoordinator(
            num_workers=len(specs),
            deadlock_timeout=deadlock_timeout,
        ).explore(launch=SubprocessLauncher(specs), setup=setup, invariant=invariant, max_iterations=max_iterations)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'dpor' or 'exhaustive'")
