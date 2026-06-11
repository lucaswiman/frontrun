"""Regression tests for scheduler-timeout classification (finding 5).

A scheduler-internal ``TimeoutError`` (the fallback deadlock-timeout path)
means the run never executed under DPOR control: surviving threads free-run
unscheduled. Such a run must NOT be evaluated as a normally-completed run, so
the invariant must not be checked against the free-run state.
"""

from __future__ import annotations

from frontrun._deadlock import DeadlockError
from frontrun._dpor_runtime.explore import _scheduler_run_evaluable


def test_none_is_normal_completion() -> None:
    assert _scheduler_run_evaluable(None) is True


def test_deadlock_error_skips_invariant_checks() -> None:
    assert _scheduler_run_evaluable(DeadlockError("cycle", "T0 -> T1 -> T0")) is False


def test_scheduler_timeout_skips_invariant_checks() -> None:
    # The free-run state is meaningless; the invariant must not be evaluated.
    assert _scheduler_run_evaluable(TimeoutError("DPOR deadlock")) is False
