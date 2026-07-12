"""Post-``_finished`` drain loops must not busy-spin on the patched sleep.

When the random (bytecode) scheduler exhausts its schedule or op budget it
sets ``_finished`` while worker threads are still alive.  Blocked cooperative
waiters then enter a bounded real-time poll window ("give notifications a
brief moment to land").  Those poll loops call ``time.sleep(0.001)`` — but
``time.sleep`` is patched during exploration, and for a managed thread the
patched sleep returns immediately once the scheduler is finished.  The 1-second
drain window then degenerates into a hot spin (hundreds of thousands of
iterations, ~1 CPU-second per blocked waiter per attempt).

These tests pin the fix: the drain loops must use the real, unpatched sleep.
The assertion is on *CPU time* consumed by the poll window, which is
implementation-agnostic: a real 1 ms sleep costs ~0 CPU; a no-op patched sleep
costs a full core.
"""

from __future__ import annotations

import time

from frontrun._cooperative import (
    CooperativeCondition,
    CooperativeSemaphore,
    patch_sleep,
    set_context,
    unpatch_sleep,
)


class _FinishedScheduler:
    """Minimal scheduler standing in for a budget-exhausted OpcodeScheduler."""

    _finished = True
    _error = None
    virtual_clock = None

    def wait_for_turn(self, thread_id: int) -> bool:
        return False


def test_semaphore_finished_drain_does_not_busy_spin() -> None:
    scheduler = _FinishedScheduler()
    sem = CooperativeSemaphore(0)  # never acquirable: the drain runs its full window
    patch_sleep()
    try:
        set_context(scheduler, 0)  # type: ignore[arg-type]
        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        assert sem.acquire() is False
        cpu_spent = time.process_time() - cpu_before
        wall_spent = time.perf_counter() - wall_before
    finally:
        set_context(None, None)  # type: ignore[arg-type]
        unpatch_sleep()
    # The drain window is ~1 s of wall time either way; the regression is that
    # a patched no-op sleep burns that whole second as CPU.
    assert wall_spent >= 0.5, f"drain window unexpectedly short ({wall_spent:.3f}s)"
    assert cpu_spent < 0.5, f"finished-scheduler drain busy-spun ({cpu_spent:.3f}s CPU for {wall_spent:.3f}s wall)"


def test_condition_finished_poll_does_not_busy_spin() -> None:
    scheduler = _FinishedScheduler()
    cond = CooperativeCondition()
    patch_sleep()
    try:
        set_context(scheduler, 0)  # type: ignore[arg-type]
        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        with cond:
            served = cond.wait()  # nobody notifies: the poll runs its full window
        cpu_spent = time.process_time() - cpu_before
        wall_spent = time.perf_counter() - wall_before
        assert served is False
    finally:
        set_context(None, None)  # type: ignore[arg-type]
        unpatch_sleep()
    assert wall_spent >= 0.5, f"poll window unexpectedly short ({wall_spent:.3f}s)"
    assert cpu_spent < 0.5, f"finished-scheduler poll busy-spun ({cpu_spent:.3f}s CPU for {wall_spent:.3f}s wall)"
