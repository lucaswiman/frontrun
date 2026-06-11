"""Regression tests for DPOR runner error selection (finding 4).

When multiple worker threads record errors, a genuine failure (e.g. an
AssertionError) must not be discarded just because another thread happened
to record a TimeoutError first.
"""

from __future__ import annotations

import pytest

from frontrun._dpor_runtime.runner import DporBytecodeRunner


def _make_runner() -> DporBytecodeRunner:
    runner = DporBytecodeRunner.__new__(DporBytecodeRunner)
    runner.errors = {}
    return runner


def test_real_error_raised_over_earlier_timeout() -> None:
    runner = _make_runner()
    runner.errors = {0: TimeoutError("worker 0 timed out"), 1: AssertionError("boom")}
    with pytest.raises(AssertionError, match="boom"):
        runner._raise_recorded_errors()


def test_timeout_only_still_suppressed() -> None:
    runner = _make_runner()
    runner.errors = {0: TimeoutError("a"), 1: TimeoutError("b")}
    # All errors are timeouts: nothing should be raised.
    runner._raise_recorded_errors()


def test_no_errors_is_noop() -> None:
    runner = _make_runner()
    runner._raise_recorded_errors()


def test_real_error_first_still_raised() -> None:
    runner = _make_runner()
    runner.errors = {0: ValueError("real"), 1: TimeoutError("late")}
    with pytest.raises(ValueError, match="real"):
        runner._raise_recorded_errors()
