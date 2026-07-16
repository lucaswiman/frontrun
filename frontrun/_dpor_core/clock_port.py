"""Shared scheduler-facing virtual-clock protocol (sync DPOR + random).

Both the systematic :class:`~frontrun._dpor_runtime.scheduler.DporScheduler` and
the random :class:`~frontrun.bytecode.OpcodeScheduler` expose the same
cooperative-primitive-facing surface for virtual timeouts and blocking spins:
``add_timed_wait`` / ``remove_timed_wait`` / ``give_up_timed_wait`` /
``note_blocking_spin`` / ``note_spin_release`` (called from ``_cooperative.py``)
plus a ``blocks_clock_progress`` query and the "process due deadlines" advance.

:class:`VirtualClockPort` owns the shared
:class:`~frontrun._virtual_clock.DeadlineCoordinator` and the ``spin_waiters``
map, and parameterises the two schedulers' differences with callbacks:

* ``block`` / ``unblock`` / ``sync`` — DPOR marks the actor blocked/unblocked in
  the Rust engine and re-syncs the clock actor; the random scheduler has no
  engine, so these are no-ops.
* ``engine_lock`` — a real lock for DPOR (serialises PyO3 ``&mut self`` borrows),
  a null context for the random scheduler.
* ``on_give_up`` — DPOR scrubs the giving-up waiter from its lock-waiter sets.
* ``on_added`` — DPOR performs any replay-owed clock advance after registration.

The ``condition`` is the scheduler's own real condition, so ``notify_all`` wakes
its waiters and every ``spin_waiters`` access stays serialised on the one lock
that all callers already hold.

Locking: mutators take ``condition`` then ``engine_lock`` (the established
scheduler order), matching what both schedulers did before.  ``advance_clock_to``
and ``blocks_clock_progress`` are *core* helpers that assume the caller already
holds the relevant lock (they run inside scheduling code that does).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from frontrun._virtual_clock import _TIMED_WAIT_TOKEN, DeadlineCoordinator

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from frontrun._virtual_clock import VirtualClock, WakeEvent


def _noop_actor(_actor_id: int) -> None:
    pass


def _noop() -> None:
    pass


def noop_on_wake(_event: WakeEvent) -> None:
    """Default per-wake callback: pop the spin flag (done by the port) only."""


class VirtualClockPort:
    """Own deadline membership + blocking-spin flags for one sync scheduler."""

    def __init__(
        self,
        *,
        condition: Any,
        engine_lock: AbstractContextManager[Any] | None = None,
        block: Callable[[int], None] = _noop_actor,
        unblock: Callable[[int], None] = _noop_actor,
        sync: Callable[[], None] = _noop,
        on_give_up: Callable[[int], None] = _noop_actor,
        on_timed_wait_added_locked: Callable[[], None] = _noop,
        on_added: Callable[[], None] = _noop,
    ) -> None:
        self.coordinator = DeadlineCoordinator()
        # actor_id → resource id for actors blocked in cooperative spin loops
        # that have no richer DPOR sync event (Condition/Queue, untimed
        # acquires, virtual timed waits).  NOT deadline state.
        self.spin_waiters: dict[int, int] = {}
        self._condition = condition
        self._engine_lock: AbstractContextManager[Any] = (
            engine_lock if engine_lock is not None else contextlib.nullcontext()
        )
        self._block = block
        self._unblock = unblock
        self._sync = sync
        self._on_give_up = on_give_up
        self._on_timed_wait_added_locked = on_timed_wait_added_locked
        self._on_added = on_added

    # -- Cooperative-primitive-facing protocol (callers hold no scheduler lock) --

    def add_timed_wait(
        self,
        actor_id: int,
        deadline: float | None = None,
        *,
        timeout: float | None = None,
        clock: VirtualClock | None = None,
        wake_id: int | None = None,
    ) -> float:
        """Register a virtual deadline for a timed lock acquire.

        Prefer the relative form (``timeout=`` + ``clock=``): the deadline is
        then computed *inside* the condition hold, so a concurrent explored-mode
        clock advance landing between the caller's ``clock.now()`` read and the
        registration cannot produce an already-expired deadline (a timed wait
        observing more virtual time than its timeout, nondeterministically).
        Returns the registered deadline.
        """
        with self._condition:
            with self._engine_lock:
                if deadline is None:
                    if timeout is None or clock is None:
                        raise TypeError("add_timed_wait needs either deadline= or (timeout= and clock=)")
                    deadline = clock.now() + timeout
                self.coordinator.add_timeout(actor_id, deadline, _TIMED_WAIT_TOKEN, wake_id=wake_id)
                self._sync()
                self._on_timed_wait_added_locked()
            self._condition.notify_all()
        # DPOR replay may owe an actor step that arrived before this
        # registration (schedule drift); perform it now.
        self._on_added()
        return deadline

    def remove_timed_wait(self, actor_id: int) -> None:
        """Deregister a timed-acquire deadline (acquired or gave up)."""
        with self._condition:
            with self._engine_lock:
                self.coordinator.cancel(actor_id, _TIMED_WAIT_TOKEN)
                self._sync()
            self._condition.notify_all()

    def give_up_timed_wait(self, actor_id: int) -> None:
        """Atomically unblock a timed-acquire waiter and drop its deadline.

        Unblock *before* the deadline is dropped, under a single lock hold, so no
        scheduler advance ever observes the waiter engine-blocked with no pending
        deadline (which would arm a spurious exact-deadlock false positive).  A
        transiently stale deadline is harmless (the next advance no-ops on it).
        """
        with self._condition:
            with self._engine_lock:
                self._unblock(actor_id)
                self.coordinator.cancel(actor_id, _TIMED_WAIT_TOKEN)
                self._sync()
                self._on_give_up(actor_id)
            self._condition.notify_all()

    def note_blocking_spin(self, actor_id: int, resource_id: int, waiting: bool, *, timed_wait: bool = False) -> None:
        """Flag/unflag an actor as spinning on a cooperative wait.

        ``timed_wait=True`` marks a spin backed by a virtual timeout deadline.
        Such a flag is *refused* when the actor no longer has a pending timeout
        deadline: the deadline already fired between the caller's expiry check
        and this call (the caller was queued on ``condition`` while an autojump
        advanced past it), and a stale flag would count the waiter as
        clock-blocked — letting the next autojump advance past the waiter's own
        deadline before it re-probes (a timed wait observing more virtual time
        than its timeout), or engine-blocking it with no deadline pending (the
        exact-deadlock false-positive window).  Flag and deadline share
        ``condition``, so the check is race-free; the refused waiter re-probes
        and observes expiry on its next loop iteration.
        """
        with self._condition:
            with self._engine_lock:
                if waiting:
                    if timed_wait and not self.coordinator.in_timed_wait(actor_id):
                        return
                    self.spin_waiters[actor_id] = resource_id
                    self._block(actor_id)
                elif self.spin_waiters.pop(actor_id, None) is not None:
                    self._unblock(actor_id)
                self._sync()
            self._condition.notify_all()

    def note_spin_release(self, resource_id: int) -> None:
        """Wake every spin waiter registered against *resource_id*."""
        with self._condition:
            with self._engine_lock:
                for actor_id, res in list(self.spin_waiters.items()):
                    if res == resource_id:
                        del self.spin_waiters[actor_id]
                        self._unblock(actor_id)
                self._sync()
            self._condition.notify_all()

    # -- Core helpers (caller already holds the scheduler's serialising lock) --

    def advance_clock_to(
        self, clock: VirtualClock, target: float | None, on_wake: Callable[[WakeEvent], None]
    ) -> list[WakeEvent]:
        """Advance the clock and process every deadline that becomes due.

        Pops each due actor's spin flag so a due virtual timed wait must
        re-probe before it can be counted as blocked again, then invokes
        *on_wake* per event for engine-unblock / happens-before reporting.
        With ``target=None`` jumps to the earliest pending deadline.  The
        coordinator itself drops every due entry, so this is the sole source of
        sleep/timed-wait membership updates.
        """
        due = self.coordinator.advance_to_next(clock) if target is None else self.coordinator.advance_to(clock, target)
        for event in due:
            self.spin_waiters.pop(event.actor_id, None)
            on_wake(event)
        return due

    def blocks_clock_progress(self, actor_id: int) -> bool:
        """Whether *actor_id* cannot move until the virtual clock advances."""
        return (
            self.coordinator.is_sleeping(actor_id)
            or self.coordinator.in_timed_wait(actor_id)
            or actor_id in self.spin_waiters
        )

    def spin_waiter_ids(self) -> list[int]:
        """Sorted actor ids currently flagged as blocking spinners (diagnostics)."""
        return sorted(self.spin_waiters)

    def discard_spin_waiter(self, actor_id: int) -> None:
        """Drop *actor_id*'s spin flag (caller holds the serialising lock)."""
        self.spin_waiters.pop(actor_id, None)
