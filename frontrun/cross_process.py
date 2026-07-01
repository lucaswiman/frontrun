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
from frontrun._dpor_runtime.xproc.launch import Subprocess, SubprocessLauncher

__all__ = ["CrossProcessResult", "Subprocess", "explore_processes"]


def explore_processes(
    processes: Mapping[str, Subprocess] | Sequence[Subprocess],
    *,
    setup: Callable[[], Any],
    invariant: Callable[[], bool],
    max_iterations: int = 4096,
    deadlock_timeout: float = 15.0,
) -> CrossProcessResult:
    """Explore interleavings of *processes* contending on shared external state.

    ``processes`` is a mapping of label → :class:`Subprocess` (labels are for
    readability) or a plain sequence. ``setup`` resets the external state before
    each interleaving and ``invariant`` checks it afterwards; both run in this
    (coordinator) process. Exploration is exhaustive over the access-interleaving
    space, bounded by ``max_iterations``.
    """
    specs = list(processes.values()) if isinstance(processes, Mapping) else list(processes)
    if not specs:
        raise ValueError("explore_processes requires at least one Subprocess")
    coordinator = CrossProcessCoordinator(num_workers=len(specs), deadlock_timeout=deadlock_timeout)
    launcher = SubprocessLauncher(specs)
    return coordinator.explore(
        launch=launcher,
        setup=setup,
        invariant=invariant,
        max_iterations=max_iterations,
    )
