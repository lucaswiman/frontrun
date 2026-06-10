"""Regression tests for scheduler-timeout classification (finding 5).

A scheduler-internal ``TimeoutError`` (the fallback deadlock-timeout path)
means the run never executed under DPOR control: surviving threads free-run
unscheduled. Such a run must NOT be evaluated as a normally-completed run, so
the invariant must not be checked against the free-run state.
"""

from __future__ import annotations

from frontrun._deadlock import DeadlockError
from frontrun._dpor_runtime.explore import _classify_scheduler_outcome


def test_none_is_normal_completion() -> None:
    outcome = _classify_scheduler_outcome(None)
    assert outcome.is_deadlock is False
    assert outcome.scheduler_timed_out is False
    assert outcome.evaluate_invariant is True


def test_deadlock_error_is_deadlock() -> None:
    outcome = _classify_scheduler_outcome(DeadlockError("cycle", "T0 -> T1 -> T0"))
    assert outcome.is_deadlock is True
    assert outcome.scheduler_timed_out is False
    assert outcome.evaluate_invariant is False


def test_scheduler_timeout_is_not_normal_completion() -> None:
    outcome = _classify_scheduler_outcome(TimeoutError("DPOR deadlock"))
    assert outcome.is_deadlock is False
    assert outcome.scheduler_timed_out is True
    # The free-run state is meaningless; the invariant must not be evaluated.
    assert outcome.evaluate_invariant is False
