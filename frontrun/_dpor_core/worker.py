"""Backend-agnostic worker *port* for DPOR exploration.

The DPOR *decision core* — the Rust engine, the :class:`RowLockRegistry`, and
the wait-for graph — is independent of how the competing units of work are run.
``WorkerSet`` captures that split so a new backend can plug in without touching
the core: it launches N workers, drives them to completion against a deadline,
and reports which ones overran. In-process DPOR runs the workers as OS threads
(:func:`frontrun._threaded_runner.run_thread_group`, wrapped by
:class:`frontrun._dpor_runtime.worker_set.ThreadWorkerSet`); the cross-process
backend (``ideas/cross_process_exploration.md``) spawns processes.

The worker-facing *scheduler surface* is a separate concern: the SQL/Redis
interception layers call ``report_and_wait`` / ``acquire_row_locks`` /
``release_row_locks`` (plus the io-reporter callable) on whatever scheduler is
in thread-local context; cross-process that object is
:class:`frontrun._dpor_runtime.xproc.proxy.SchedulerProxy`.

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
