"""Shared concurrency primitives unifying sync (threaded) and async DPOR drivers.

The sync driver in :mod:`frontrun._dpor_runtime` and the async driver in
:mod:`frontrun.async_dpor` share the same outer-loop shape but historically
diverged on two small concurrency-shaped details:

* The **engine lock** — sync DPOR runs workers on real threads, so PyO3
  ``&mut self`` borrows on the Rust ``PyDporEngine`` need a real
  :class:`threading.Lock` to serialise them (panics rather than blocks on
  free-threaded Python).  Async DPOR runs all tasks on a single event-loop
  thread, so it uses a no-op context manager.

* The **per-execution boundary** — both drivers loop ``while True``,
  reset per-execution state, call ``engine.begin_execution()``, run the
  workers, then call ``engine.next_execution()`` (and bail out on a
  ``total_timeout`` deadline).  The body of each iteration is necessarily
  driver-specific (threads vs tasks, sync vs ``await``), but the
  *boundary* is identical.

This module exposes both pieces:

* :class:`NoOpLock` — context-manager-shaped no-op lock for async DPOR.
* :func:`dpor_exploration_iter` — generator that yields one
  :class:`ExplorationStep` per execution, holding the engine lock while
  it advances the engine.  The caller (sync or async) owns the body.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from frontrun._dpor_core.utils import reset_execution_state

if TYPE_CHECKING:
    from frontrun._opcode_observer import StableObjectIds


class NoOpLock:
    """Context-manager-shaped no-op lock for single-threaded engine calls.

    Used by the async DPOR driver, which runs every task on the asyncio
    event-loop thread and therefore has no contention on the underlying
    Rust ``PyDporEngine``.  The sync driver passes a real
    :class:`threading.Lock` instead.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class ReplayEngine:
    """No-op engine used when replaying a fixed DPOR schedule.

    Shared by sync (`_ReplayDporScheduler`) and async (`_ReplayAsyncScheduler`)
    replay: a replay drives a *fixed* schedule and has no Rust engine, but the
    instrumentation (cooperative locks, sync reporters) still calls the engine
    surface — every method must exist and do nothing.
    """

    def report_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_access_at(self, execution: Any, thread_id: int, object_id: int, kind: str, path_id: int) -> None:
        return None

    def report_first_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_first_access_at(self, execution: Any, thread_id: int, object_id: int, kind: str, path_id: int) -> None:
        return None

    def report_io_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_synced_io_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_sync(
        self, execution: Any, thread_id: int, event_type: str, sync_id: int, path_id: int | None = None
    ) -> None:
        return None

    def register_resource_group(self, object_id: int, group_id: int) -> None:
        return None


class ReplayExecution:
    """No-op stand-in for ``PyExecution`` during counterexample replay.

    The cooperative lock replacements call ``block_thread`` / ``unblock_thread``
    on ``scheduler.execution`` to keep the DPOR engine from scheduling a
    lock-blocked thread/task; replay has no engine, so these are no-ops — but
    they must exist so lock acquire/release doesn't raise AttributeError.
    """

    def finish_thread(self, thread_id: int) -> None:
        return None

    def block_thread(self, thread_id: int) -> None:
        return None

    def unblock_thread(self, thread_id: int) -> None:
        return None


class _DporEngine(Protocol):
    """Subset of :class:`frontrun._dpor.PyDporEngine` used by the driver loop."""

    def begin_execution(self) -> Any: ...

    def next_execution(self) -> bool: ...


@dataclass(frozen=True)
class ExplorationStep:
    """One iteration of :func:`dpor_exploration_iter`.

    Attributes:
        execution: The fresh ``PyExecution`` returned by ``begin_execution``.
        index: 1-indexed iteration number (the run that's *about to happen*).
    """

    execution: Any
    index: int


def dpor_exploration_iter(
    *,
    engine: _DporEngine,
    engine_lock: AbstractContextManager[Any],
    stable_ids: StableObjectIds,
    total_deadline: float | None,
) -> Iterator[ExplorationStep]:
    """Yield one :class:`ExplorationStep` per DPOR execution to explore.

    Encapsulates the boundary work shared by ``_explore_dpor`` (sync) and
    ``_explore_async_dpor`` (async):

    1. After the required baseline execution, bail out if ``total_deadline``
       (an absolute :func:`time.monotonic` timestamp from
       :func:`make_deadline`) has passed.
    2. :func:`reset_execution_state` to clear per-execution state.
    3. ``engine.begin_execution()`` under ``engine_lock``.
    4. Yield to the caller, which runs the workers and inspects the
       resulting state (invariants, races, deadlocks).
    5. Stop if the total deadline expired while the body ran; otherwise call
       ``engine.next_execution()`` under ``engine_lock`` and stop when it
       returns ``False`` (search tree exhausted).

    The body of the loop runs *outside* the engine lock — workers acquire
    fine-grained subsections of the lock as needed.  The generator works
    in both sync and ``async def`` callers because Python's ``for`` loop
    doesn't care about the function's color.
    """
    index = 0
    while True:
        reset_execution_state(stable_ids)
        with engine_lock:
            execution = engine.begin_execution()
        index += 1
        yield ExplorationStep(execution=execution, index=index)
        # The baseline execution is always permitted, even when a tiny
        # positive budget elapsed before it began.  Check only after its body
        # (and after every subsequent body), before asking the engine to plan a
        # schedule that the caller can no longer run.
        if total_deadline is not None and time.monotonic() > total_deadline:
            return
        with engine_lock:
            if not engine.next_execution():
                return
        # Planning can itself consume a meaningful share of the total budget.
        # Do not start the schedule it produced if the deadline elapsed while
        # next_execution() was building that path.
        if total_deadline is not None and time.monotonic() > total_deadline:
            return


# ---------------------------------------------------------------------------
# Wake-edge sync-object ids (shared by sync and async DPOR)
# ---------------------------------------------------------------------------

#: High bits XORed with a thread/task id to form the sync-object id of that
#: thread's virtual-clock wake edge ("WAKE" in ASCII).  The clock actor
#: reports a ``lock_release`` on this object when it wakes a sleeper; the
#: woken thread reports the matching ``lock_acquire``, giving the engine the
#: happens-before edge "clock advanced → sleeper resumed".
WAKE_SYNC_BASE = 0x57414B45_00000000

#: Same idea for cooperative Event wakes ("EVTW"): ``set()`` reports one
#: ``lock_release`` per engine-blocked waiter on that waiter's id, and each
#: woken waiter reports the matching ``lock_acquire``.
_EVENT_WAKE_BASE = 0x45565457_00000000

_SYNC_ID_MASK = (1 << 63) - 1


def wake_sync_id(thread_id: int) -> int:
    """Sync-object id for a virtual-clock wake edge to *thread_id*."""
    return WAKE_SYNC_BASE ^ thread_id


def event_wake_sync_id(event_id: int, thread_id: int) -> int:
    """Per-waiter sync-object id for an Event wake edge.

    Mixes the event's stable id so distinct events never share wake ids
    (the multiplier spreads small sequential stable ids across the id
    space; masked to a non-negative 63-bit int for the engine).
    """
    return (_EVENT_WAKE_BASE ^ (event_id * 0x9E3779B1) ^ thread_id) & _SYNC_ID_MASK
