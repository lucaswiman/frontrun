"""Backend-agnostic worker/scheduler *ports* for DPOR exploration.

The DPOR *decision core* — the Rust engine, the :class:`RowLockRegistry`, and
the wait-for graph — is independent of how the competing units of work are run
and how the scheduler hands them their turns. Two seams capture that split so a
new backend can plug in without touching the core:

``WorkerSet``
    Launches N workers, drives them to completion against a deadline, and
    reports which ones overran. In-process DPOR runs the workers as OS threads
    (:func:`frontrun._threaded_runner.run_thread_group`, wrapped by
    :class:`frontrun._dpor_runtime.worker_set.ThreadWorkerSet`). The planned
    cross-process backend (``ideas/cross_process_exploration.md``) spawns
    subprocesses under the ``frontrun`` CLI and implements the same Protocol.

``TurnTransport``
    The *coordinator-internal* turn primitive: blocks a worker until the
    scheduler grants it the turn, and lets the scheduler hand the turn off.
    In-process this is a ``threading.Condition`` over a shared "current
    worker"/"done" set (today fused into
    :class:`frontrun._dpor_runtime.scheduler.DporScheduler`); cross-process the
    coordinator implements it by withholding/sending a GRANT frame on the
    per-worker socket. Lifting ``DporScheduler`` onto it is a later slice.

Note the worker-facing *scheduler surface* is a separate seam, not
``TurnTransport``. The SQL/Redis interception layers call ``report_and_wait`` /
``acquire_row_locks`` / ``release_row_locks`` (plus the io-reporter callable) on
whatever scheduler is in thread-local context; cross-process that object is
:class:`frontrun._dpor_runtime.xproc.proxy.SchedulerProxy`, which forwards those
calls over the socket to a coordinator that drives ``TurnTransport``.

Nothing in this module imports threading or asyncio — these are pure typing
contracts shared by every backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkerTarget:
    """One unit of work to launch.

    ``worker_id`` is the stable, dense index the DPOR engine uses as the
    "thread id" of this worker (``0..n-1``); it is identical whether the worker
    is an OS thread or a remote process, which is what lets the engine and
    row-lock state transfer across backends unchanged.
    """

    worker_id: int
    func: Callable[..., None]
    args: tuple[Any, ...] = ()


@runtime_checkable
class WorkerSet(Protocol):
    """Launches workers and joins them against a deadline.

    A single ``run()`` call corresponds to exploring **one** interleaving: the
    workers are started, joined until they finish or ``timeout`` elapses, and
    any still-alive workers are surfaced via ``on_timeout``. ``teardown`` runs
    unconditionally afterwards (even on timeout), mirroring the lifecycle of
    the existing thread runner.
    """

    def run(
        self,
        targets: Sequence[WorkerTarget],
        run_one: Callable[[WorkerTarget], None],
        *,
        timeout: float,
        on_timeout: Callable[[list[Any]], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        """Launch ``run_one(target)`` for each target and join against ``timeout``.

        ``on_timeout`` receives the backend-specific handles (OS threads,
        subprocess handles) of workers still alive at the deadline.
        """
        ...


@runtime_checkable
class TurnTransport(Protocol):
    """Turn arbitration between the scheduler core and one worker.

    In-process: a ``threading.Condition`` plus a shared "current worker" int.
    Cross-process: a unix socket per worker, where ``wait_for_turn`` blocks on
    a socket read and ``grant`` writes a token. ``DporScheduler`` does not yet
    delegate to this port — the contract is recorded here ahead of that work so
    the cross-process coordinator and the in-process scheduler converge on one
    shape.
    """

    def wait_for_turn(self, worker_id: int) -> bool:
        """Block until ``worker_id`` is granted the turn. Return ``False`` to abort."""
        ...

    def grant(self, worker_id: int) -> None:
        """Hand the turn to ``worker_id`` and wake it."""
        ...

    def mark_done(self, worker_id: int) -> None:
        """Record that ``worker_id`` has finished and will take no more turns."""
        ...
