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
from frontrun._dpor_runtime.xproc.launch import Subprocess, SubprocessLauncher

__all__ = ["CrossProcessResult", "Subprocess", "explore_processes"]


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
    launcher = SubprocessLauncher(specs)
    if strategy == "dpor":
        return DporCrossProcessCoordinator(
            num_workers=len(specs),
            deadlock_timeout=deadlock_timeout,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
        ).explore(launch=launcher, setup=setup, invariant=invariant)
    if strategy == "exhaustive":
        return CrossProcessCoordinator(
            num_workers=len(specs),
            deadlock_timeout=deadlock_timeout,
        ).explore(launch=launcher, setup=setup, invariant=invariant, max_iterations=max_iterations)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'dpor' or 'exhaustive'")
