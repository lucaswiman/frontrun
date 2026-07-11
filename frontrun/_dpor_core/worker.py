"""Backend-agnostic worker-launch port for DPOR exploration.

The DPOR *decision core* — the Rust engine, the :class:`RowLockRegistry`, and
the wait-for graph — is independent of how the competing units of work are run.
``WorkerSet`` captures that split so backends can plug in without inventing a
parallel launcher shape: it starts a collection of stable worker ids and joins
the backend-specific handles against a deadline. In-process DPOR runs workers as
OS threads; cross-process DPOR uses the same launch/join port for subprocesses.

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

    ``worker_id`` is the stable, dense index the DPOR engine uses as the worker's
    logical thread id (``0..n-1``). ``func`` / ``args`` are interpreted by the
    concrete :class:`WorkerSet`: the in-process thread backend calls
    ``func(*args)`` directly, while cross-process backends keep their callable or
    subprocess specs on the WorkerSet and use ``args`` for launch metadata such
    as the coordinator socket path.
    """

    worker_id: int
    func: Callable[..., None] | None = None
    args: tuple[Any, ...] = ()


@runtime_checkable
class WorkerSet(Protocol):
    """Starts backend-specific workers and joins their handles."""

    def launch(
        self,
        targets: Sequence[WorkerTarget],
    ) -> Any:
        """Start *targets* and return backend-specific handles."""
        ...

    def join(self, handles: Any, timeout: float) -> list[Any]:
        """Join *handles* against *timeout* and return handles still alive."""
        ...


@runtime_checkable
class TerminableWorkerSet(Protocol):
    """Optional capability for forcibly retiring poisoned worker processes.

    Process-backed reusable workers can be killed and launched again after an
    iteration aborts with their protocol stream in an unknown state. Thread
    backends deliberately do not implement this capability: Python cannot
    safely terminate an arbitrary running thread.
    """

    def terminate(self, handles: Any, timeout: float) -> None:
        """Terminate, escalate if necessary, and reap *handles*."""
        ...


@runtime_checkable
class LivenessProbe(Protocol):
    """Optional WorkerSet capability: report on worker liveness without joining.

    ``any_exited``, ``all_exited`` and ``diagnose`` always travel together — both
    process launchers (``MpLauncher``, ``SubprocessLauncher``) implement all
    three, while the thread-backed launchers implement none — so they form a
    single capability Protocol. The coordinator uses ``any_exited``
    (non-destructive) to fail fast when a launched child dies *abnormally* before
    connecting, ``all_exited`` to also fail fast when every child has exited
    (even cleanly, e.g. a target that calls ``sys.exit(0)`` at import) before all
    HELLOs arrive, and ``diagnose`` to recover the real cause for the failure
    message. Probing via ``isinstance`` (rather than ``getattr``) keeps the call
    sites type-checked, so renaming a method breaks loudly instead of silently
    reverting to the slow-timeout / bare-error path.
    """

    def any_exited(self, handles: Any) -> bool:
        """Non-destructive: has any worker exited abnormally (nonzero exit)?"""
        ...

    def all_exited(self, handles: Any) -> bool:
        """Non-destructive: has *every* launched worker exited (any code)?"""
        ...

    def diagnose(self, handles: Any) -> str | None:
        """Describe any worker that exited before connecting, else ``None``."""
        ...


@runtime_checkable
class IterationCustomizer(Protocol):
    """Optional WorkerSet capability: build a per-iteration ITER_START frame.

    Only ``MpLauncher`` needs this: its reused workers re-receive a freshly
    serialised ``(worker_fn, state)`` payload each iteration, so the coordinator
    asks the launcher to build the ITER_START message. Backends without it
    (subprocess, thread) get a plain ITER_START frame.
    """

    def iter_start_message(self, worker_id: int) -> dict[str, Any]:
        """Build the ITER_START frame for one reused worker."""
        ...
