"""Pure DPOR utility helpers shared by sync and async drivers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontrun._opcode_observer import StableObjectIds


def group_schedule_runs(schedule: list[int]) -> list[tuple[int, int]]:
    """Collapse a flat schedule into (actor_id, count) run-length pairs.

    Example::

        group_schedule_runs([0, 0, 1, 1, 1, 2])
        # → [(0, 2), (1, 3), (2, 1)]
    """
    runs: list[tuple[int, int]] = []
    for tid in schedule:
        if runs and runs[-1][0] == tid:
            runs[-1] = (tid, runs[-1][1] + 1)
        else:
            runs.append((tid, 1))
    return runs


def make_deadline(total_timeout: float | None) -> float | None:
    """Return ``time.monotonic() + total_timeout`` or ``None`` when no timeout."""
    if total_timeout is not None and (total_timeout <= 0 or not math.isfinite(total_timeout)):
        raise ValueError(f"total_timeout must be positive and finite or None, got {total_timeout!r}")
    return time.monotonic() + total_timeout if total_timeout is not None else None


def reset_execution_state(stable_ids: StableObjectIds) -> None:
    """Reset per-execution state shared by sync and async DPOR drivers.

    Clears the SQL insert tracker and resets the stable object ID map so
    IDs assigned in a previous execution do not bleed into the next.
    """
    from frontrun._sql_insert_tracker import clear_insert_tracker

    clear_insert_tracker()
    stable_ids.reset_for_execution()


def extend_replay_schedule(
    replay_schedule: list[int],
    replay_index: int,
    replay_max_ops: int,
    num_actors: int,
    actors_done: set[int],
) -> bool:
    """Extend *replay_schedule* with still-active actors when the schedule runs short.

    Used by replay schedulers (both sync and async) to avoid stalling when the
    original counterexample schedule is exhausted before all actors finish.

    Args:
        replay_schedule: The mutable schedule list to extend in-place.
        replay_index:    Current read position in *replay_schedule*.
        replay_max_ops:  Hard upper bound; refuse to extend beyond this.
        num_actors:      Total number of threads/tasks (range base for IDs).
        actors_done:     Set of actor IDs that have already finished.

    Returns:
        ``True`` if the schedule was extended, ``False`` if it could not be
        (either the cap was reached or all actors are done).
    """
    if replay_index >= replay_max_ops:
        return False
    active = [t for t in range(num_actors) if t not in actors_done]
    if not active:
        return False
    replay_schedule.extend(active)
    return True


def advance_replay_index(
    replay_schedule: list[int],
    replay_index: int,
    extend_fn: Callable[[], bool],
    actors_done: set[int],
) -> tuple[int, int | None]:
    """Walk *replay_schedule* forward past finished actors and return the next live one.

    Shared indexing core used by both the sync :class:`_ReplayDporScheduler`
    and the async :class:`_ReplayAsyncScheduler`.  Neither the wait mechanism
    (condition variable vs asyncio callback) nor the "current actor" field are
    touched here — callers own those.

    Args:
        replay_schedule: The schedule list (may be extended in-place by *extend_fn*).
        replay_index:    Current read position; advanced by this function.
        extend_fn:       Zero-arg callable that appends more entries and returns
                         ``True``, or returns ``False`` when the schedule is exhausted.
        actors_done:     Set of actor IDs that have already finished.

    Returns:
        ``(new_index, next_actor)`` where *next_actor* is ``None`` when the
        schedule is exhausted and no live actor can be scheduled.
    """
    while True:
        if replay_index >= len(replay_schedule):
            if not extend_fn():
                return replay_index, None
            continue
        scheduled = replay_schedule[replay_index]
        replay_index += 1
        if scheduled not in actors_done:
            return replay_index, scheduled


def is_reproduction_run(*, deadlocked: bool, has_invariant: bool, invariant_failed: bool) -> bool:
    """Return ``True`` when a single reproduction run counts as a confirmed bug.

    Shared classification logic used by both the sync and async reproduction
    loops — the surrounding setup/teardown and exception handling differ, but
    the decision of *whether a run reproduced the original failure* is identical.

    Rules:
    - If the run **deadlocked** it reproduces iff the original failure *was* a
      deadlock (i.e. there is no invariant to check).
    - If the run **completed** without deadlock it reproduces iff the invariant
      was provided and failed.
    """
    if deadlocked:
        return not has_invariant
    return has_invariant and invariant_failed
