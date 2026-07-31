"""Regression tests for scheduler-timeout classification (finding 5).

A scheduler-internal ``TimeoutError`` (the fallback deadlock-timeout path)
means the run never executed under DPOR control: surviving threads free-run
unscheduled. Such a run must NOT be evaluated as a normally-completed run, so
the invariant must not be checked against the free-run state.
"""

from __future__ import annotations

import _thread
import time

import pytest

import frontrun
from frontrun._deadlock import DeadlockError
from frontrun._dpor_runtime.explore import _scheduler_run_evaluable


def test_none_is_normal_completion() -> None:
    assert _scheduler_run_evaluable(None) is True


def test_deadlock_error_skips_invariant_checks() -> None:
    assert _scheduler_run_evaluable(DeadlockError("cycle", "T0 -> T1 -> T0")) is False


def test_scheduler_timeout_skips_invariant_checks() -> None:
    # The free-run state is meaningless; the invariant must not be evaluated.
    assert _scheduler_run_evaluable(TimeoutError("DPOR deadlock")) is False


def test_scheduler_timeout_cannot_return_passing_verdict() -> None:
    """An exploration with no completed schedule cannot certify safety."""

    def blocks_past_run_timeout(_state: object) -> None:
        time.sleep(0.2)

    result = frontrun.explore(
        setup=object,
        workers=[blocks_past_run_timeout],
        invariant=lambda _state: True,
        detect_io=False,
        patch_sleep=False,
        timeout_per_run=0.01,
        max_executions=1,
        reproduce_on_failure=0,
    )

    assert not result.property_holds, "a timed-out, unevaluable run was reported as a passing proof"


@pytest.mark.intentionally_leaves_dangling_threads
def test_later_timeout_does_not_unprove_an_earlier_counterexample() -> None:
    """A timeout truncates the search; it cannot retract a proven failure.

    With ``stop_on_first=False`` the search keeps going after a counterexample
    is recorded.  If a later execution then times out, the ``False`` verdict
    must survive: demoting it to inconclusive lets
    ``assert_holds(allow_inconclusive=True)`` -- the sanctioned opt-in for
    budget-bounded lanes -- go green on a run that found a real race.
    """
    executions = {"n": 0}

    def setup() -> dict[str, int]:
        executions["n"] += 1
        return {"c": 0}

    def racer(state: dict[str, int]) -> None:
        state["c"] = state["c"] + 1

    def blocks_after_the_race_is_found(state: dict[str, int]) -> None:
        if executions["n"] >= 3:
            # A raw lock the cooperative layer cannot see: the scheduler gives
            # up on this execution with a TimeoutError.
            unschedulable = _thread.allocate_lock()
            unschedulable.acquire()
            unschedulable.acquire()
        racer(state)

    result = frontrun.explore(
        setup=setup,
        workers=[blocks_after_the_race_is_found, racer],
        invariant=lambda state: state["c"] == 2,
        stop_on_first=False,
        timeout_per_run=3,
        deadlock_timeout=1,
        reproduce_on_failure=0,
    )

    assert result.failures, "expected the lost update to be found before the timing-out execution"
    assert result.property_holds is False, "a later timeout retracted an already-proven counterexample"
    assert result.exhausted is False, "a truncated search must not claim full coverage"
    with pytest.raises(AssertionError):
        result.assert_holds(allow_inconclusive=True)
