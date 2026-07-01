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
from typing import Any

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
        where = f" at schedule {result.failing_schedule}" if result.failing_schedule is not None else ""
        explanation = f"{result.failure or 'invariant violated'}{where}"
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
    stop_on_first: bool = True,
    **_ignored: Any,
) -> Any:
    """Back the ``frontrun.explore(execution="process")`` path.

    Runs each worker callable in its own ``multiprocessing`` process. ``setup()``
    returns a *picklable* handle to the shared external state (e.g. a DB URL);
    it is passed to every ``worker(state)`` and to ``invariant(state)``. Both
    ``setup`` and ``invariant`` run in this (coordinator) process.
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
        stop_on_first=stop_on_first,
    )
    launcher = MpLauncher(workers, state_fn=lambda: state_box.get("state"))
    result = coordinator.explore(launch=launcher, setup=coord_setup, invariant=coord_invariant)
    return _to_interleaving_result(result)


def explore_processes(
    processes: Mapping[str, Subprocess] | Sequence[Subprocess],
    *,
    setup: Callable[[], Any],
    invariant: Callable[[], bool],
    strategy: str = "dpor",
    max_iterations: int = 4096,
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    deadlock_timeout: float = 15.0,
    reuse_workers: bool = False,
) -> CrossProcessResult:
    """Explore interleavings of *processes* contending on shared external state.

    ``processes`` is a mapping of label → :class:`Subprocess` (labels are for
    readability) or a plain sequence. ``setup`` resets the external state before
    each interleaving and ``invariant`` checks it afterwards; both run in this
    (coordinator) process.

    ``strategy``:

    * ``"dpor"`` (default) drives the Rust DPOR engine, pruning equivalent
      interleavings (partial-order reduction) and detecting cross-worker
      ``SELECT FOR UPDATE`` deadlocks. ``max_executions`` / ``preemption_bound``
      tune the search.
    * ``"exhaustive"`` brute-forces every interleaving at external-access
      granularity, bounded by ``max_iterations``. Useful as a reduction-free
      cross-check.
    """
    specs = list(processes.values()) if isinstance(processes, Mapping) else list(processes)
    if not specs:
        raise ValueError("explore_processes requires at least one Subprocess")
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
