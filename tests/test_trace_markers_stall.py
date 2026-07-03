"""Finding 2: a stalled marker schedule must surface coordinator.error.

When ``wait_for_turn`` times out it sets ``coordinator.error`` and returns; the
trace function re-raises it, but ``build_trace_function`` swallows the
exception (report_error + return None) and the worker thread continues running
UNSCHEDULED.  ``finalize_marker_executor_run`` previously only inspected
``task_errors`` and the ``current_step > 0 and not completed`` case, so a stall
at step 0 produced no exception at all — the threads ran under an uncontrolled
interleaving that ``explore_marker_interleavings`` could then score as a clean
pass.

After the fix, ``finalize_marker_executor_run`` raises ``coordinator.error``
whenever it is set, including the step-0 stall.
"""

import pytest

from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor


def _marker_worker():
    x = 1  # frontrun: m
    return x


def _two_marker_worker():
    x = 1  # frontrun: m
    y = 2  # frontrun: n
    return x + y


def test_step_zero_stall_with_later_marker_reports_stall_not_masked():
    """A stall must surface the stall TimeoutError, not a masked RuntimeError.

    With a worker that hits two markers (``m`` then ``n``) under a schedule
    that stalls at step 0 (expecting ``t2.m`` while only ``t1`` runs), the
    first marker ``m`` stalls and sets ``coordinator.error``.  The frame keeps
    being traced, so ``t1`` reaches its second marker ``n`` and re-enters
    ``_wait_for_marker``; that must not release the already-released execution
    lock (which would raise ``RuntimeError('release unlocked lock')`` and,
    via ``report_error``, overwrite/mask the original stall diagnostic).
    """
    schedule = Schedule([Step("t2", "m"), Step("t1", "m"), Step("t1", "n")])

    executor = TraceExecutor(schedule, deadlock_timeout=0.3)

    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        executor.run({"t1": _two_marker_worker}, timeout=5.0)

    assert "stall" in str(excinfo.value).lower(), excinfo.value
    assert isinstance(executor.coordinator.error, TimeoutError), executor.coordinator.error


def test_step_zero_stall_surfaces_error():
    """A stall at step 0 (waiting for a thread that never runs) must raise.

    The schedule's first step expects ``t2`` to hit marker ``m`` first, but
    only ``t1`` runs.  ``t1`` hits ``m``, calls ``wait_for_turn`` which times
    out and sets ``coordinator.error`` while ``current_step`` is still 0.
    finalize must surface that error instead of returning cleanly (which would
    let the worker run UNSCHEDULED and be scored as a clean pass).
    """
    schedule = Schedule([Step("t2", "m"), Step("t1", "m")])

    executor = TraceExecutor(schedule, deadlock_timeout=0.3)

    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        executor.run({"t1": _marker_worker}, timeout=5.0)

    # The surfaced error must be the coordinator stall error, not silence.
    assert "stall" in str(excinfo.value).lower(), excinfo.value
