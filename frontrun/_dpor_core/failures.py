"""Shared failure-recording helper for sync and async DPOR drivers."""

from __future__ import annotations

from frontrun.common import InterleavingResult


def record_dpor_failure(
    result: InterleavingResult,
    schedule_list: list[int],
    explanation: str | None,
    *,
    races_detected: bool = False,
) -> list[int]:
    """Record a single DPOR execution failure into *result*.

    Updates ``property_holds``, ``races_detected``, ``failures``,
    ``counterexample``, and ``explanation`` (the last two only if not already
    set by an earlier failure).

    Args:
        result: The :class:`~frontrun.common.InterleavingResult` to update.
        schedule_list: The schedule for the failing execution (already
            extracted from the engine, under any necessary lock).
        explanation: Human-readable description of the failure, or ``None``
            to defer setting ``result.explanation`` (useful when the explanation
            depends on reproduction data computed after this call).
        races_detected: When True, OR-accumulates into
            ``result.races_detected``.

    Returns:
        *schedule_list* unchanged (convenience for callers that capture the
        schedule after calling this function).
    """
    result.property_holds = False
    result.races_detected = result.races_detected or races_detected
    result.failures.append((result.num_explored, schedule_list))
    if result.counterexample is None:
        result.counterexample = schedule_list
        if explanation is not None:
            result.explanation = explanation
    return schedule_list


def format_exact_deadlock_desc(
    *,
    noun: str,
    sleepers: list[int],
    done: list[int],
    spin_waiters: list[int] | None = None,
) -> str:
    """Describe an exact virtual-clock deadlock (shared by sync + async DPOR).

    Every live thread/task is blocked with no deadline pending, so no transition
    can ever fire.  *noun* is ``"threads"`` (sync) or ``"tasks"`` (async); the
    sync scheduler also has cooperative spin waiters to report, the async one
    does not (pass ``spin_waiters=None`` to omit that clause).  All list
    arguments are expected pre-sorted.
    """
    parts = [f"sleepers={sleepers}"]
    if spin_waiters is not None:
        parts.append(f"spin_waiters={spin_waiters}")
    parts.append(f"done={done}")
    return f"all live {noun} are blocked and no virtual-clock deadline is pending ({', '.join(parts)})"
