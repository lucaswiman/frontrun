"""Clock-actor helpers shared by the sync and async DPOR schedulers.

Both :class:`~frontrun._dpor_runtime.scheduler.DporScheduler` and
:class:`~frontrun.async_dpor.AsyncDporScheduler` drive a virtual clock through an
extra engine thread — the *clock actor* — whose only transition is "advance the
clock to the earliest pending deadline and wake its sleepers".  The two
schedulers kept near-identical copies of the actor enable/disable sync, the
due-deadline dispatch loop, the wake happens-before edge, the autojump
predicate, and the actor retirement on completion.  Those five pieces are pure
(no threading / asyncio) and live here so the two schedulers — and the two
replay schedulers — call one implementation.

The control flow of ``_schedule_next`` / ``sleep_until`` stays per-scheduler
(their wait primitives and replay interplay genuinely differ); only the
decision logic and the correctness-critical wake edge are shared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from frontrun._dpor_core.concurrency import wake_sync_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from frontrun._virtual_clock import DeadlineCoordinator, VirtualClock, WakeEvent


class _EngineExecution(Protocol):
    """The engine-execution surface the clock actor pokes (block/unblock/finish)."""

    def block_thread(self, actor_id: int) -> None: ...
    def unblock_thread(self, actor_id: int) -> None: ...
    def finish_thread(self, actor_id: int) -> None: ...


class _SyncReporter(Protocol):
    """The engine ``report_sync`` surface used to close a wake happens-before edge."""

    def __call__(
        self, execution: object, actor_id: int | None, event_type: str, sync_id: int, path_id: int | None
    ) -> object: ...


def sync_clock_actor(
    execution: _EngineExecution,
    clock_actor_id: int | None,
    clock_mode: str,
    has_pending_deadlines: bool,
) -> None:
    """Keep the clock actor's enabledness in step with pending deadlines.

    In ``"explored"`` mode the actor is runnable whenever a deadline is pending
    (so the engine explores clock-step orderings like any other choice); in
    autojump (``"virtual"``) mode it stays blocked — ``_schedule_next`` enables
    it transiently when nothing else is runnable.
    """
    if clock_actor_id is None:
        return
    if clock_mode == "explored" and has_pending_deadlines:
        execution.unblock_thread(clock_actor_id)
    else:
        execution.block_thread(clock_actor_id)


def can_autojump(virtual_clock: VirtualClock | None, clock_actor_id: int | None, has_pending_deadlines: bool) -> bool:
    """Whether advancing the clock is the only possible transition right now.

    True when a virtual clock and its actor exist and a deadline is pending —
    the state in which ``_schedule_next`` must enable the actor because nothing
    else is runnable.
    """
    return virtual_clock is not None and clock_actor_id is not None and has_pending_deadlines


def retire_actor_if_done(execution: _EngineExecution, clock_actor_id: int | None, done_count: int, total: int) -> None:
    """Retire the clock actor once every real thread/task has finished.

    Unblocks then finishes the actor so the engine sees the execution as
    complete (an actor left blocked would look like a live-but-stuck thread).
    """
    if clock_actor_id is not None and done_count >= total:
        execution.unblock_thread(clock_actor_id)
        execution.finish_thread(clock_actor_id)


def advance_and_dispatch(
    coordinator: DeadlineCoordinator,
    clock: VirtualClock,
    target: float | None,
    *,
    on_sleep: Callable[[WakeEvent], None],
    on_timeout: Callable[[WakeEvent], None],
) -> list[WakeEvent]:
    """Advance *clock* and dispatch every due deadline to a per-kind callback.

    With ``target=None`` jumps to the earliest pending deadline (one actor
    step); otherwise advances exactly to *target*.  The coordinator drops every
    due entry, so this is the sole membership update.  ``sleep``-kind events go
    to *on_sleep* (wake edge / sleeper bookkeeping) and ``timeout``-kind events
    to *on_timeout* (token fire / lock-blocked scrub), matching each caller's
    divergent leaf work.
    """
    due = coordinator.advance_to_next(clock) if target is None else coordinator.advance_to(clock, target)
    for event in due:
        if event.kind == "sleep":
            on_sleep(event)
        elif event.kind == "timeout":
            on_timeout(event)
    return due


def report_clock_sleep_wake(
    engine_report_sync: _SyncReporter,
    execution: _EngineExecution,
    clock_actor_id: int | None,
    event: WakeEvent,
    path_id: int | None,
) -> None:
    """Close the clock-advance → sleeper-resume happens-before edge.

    Unblocks the sleeper in the engine and reports the ``lock_release`` half of
    the wake edge from the clock actor (the resuming sleeper reports the
    matching ``lock_acquire`` half).  Shared verbatim by the sync and async
    exploration schedulers — the subtle, correctness-critical bit of a clock
    advance.
    """
    execution.unblock_thread(event.actor_id)
    engine_report_sync(
        execution,
        clock_actor_id,
        "lock_release",
        event.wake_id if event.wake_id is not None else wake_sync_id(event.actor_id),
        path_id,
    )
