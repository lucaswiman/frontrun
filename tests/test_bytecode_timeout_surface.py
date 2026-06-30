"""Finding 9d: surface scheduler/global timeouts instead of evaluating the
invariant on a half-finished racing state.

When ``run_with_schedule`` hits the global timeout, daemon worker threads may
still be mutating the shared state.  Previously the TimeoutError was swallowed
and the (incomplete) state returned, so ``explore_random`` evaluated the
invariant on a racing half-finished state and could report a spurious result.
"""

from __future__ import annotations

import time

import pytest

import frontrun
from frontrun.bytecode import run_with_schedule


class _State:
    def __init__(self) -> None:
        self.value = 0


def _slow_blocker(state: _State) -> None:
    # A real (unpatched) blocking sleep the cooperative scheduler cannot
    # preempt -> the global timeout fires before the thread completes.  The
    # sleep is only slightly longer than the run timeout so the daemon thread
    # still finishes promptly and is joined during teardown.
    time.sleep(0.6)
    state.value = 1


def test_run_with_schedule_raises_on_timeout():
    """A timed-out run must surface a TimeoutError, not return a partial state."""
    with pytest.raises(TimeoutError):
        run_with_schedule(
            schedule=[0],
            setup=_State,
            threads=[_slow_blocker],
            timeout=0.3,
            patch_sleep=False,
        )


def test_explore_random_skips_invariant_on_timeout():
    """explore_random must not flag a violation purely from a timed-out run."""
    result = frontrun.explore_random(
        setup=_State,
        threads=[_slow_blocker],
        # Invariant that the half-finished state (value==0) would violate.
        invariant=lambda s: s.value == 1,
        max_attempts=2,
        max_ops=20,
        timeout_per_run=0.3,
        patch_sleep=False,
        seed=1,
    )
    # The run timed out, so the invariant was inconclusive — it must NOT be
    # reported as a violation discovered from a half-finished state.
    assert result.property_holds, f"timed-out run was scored as a property violation: {result.explanation}"
