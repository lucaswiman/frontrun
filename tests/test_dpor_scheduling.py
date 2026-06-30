"""Unit tests for shared scheduling-decision helpers (_dpor_core/scheduling).

These pure helpers are lifted out of the otherwise-duplicated ``_schedule_next``
logic in the sync (``DporScheduler``) and async (``AsyncDporScheduler``)
schedulers, whose only real difference is the threading-vs-asyncio block/wake
primitive — the decision below is identical dict/set logic in both.
"""

from __future__ import annotations

from frontrun._dpor_core.scheduling import apply_lock_blocked_override


def test_no_override_when_not_blocked() -> None:
    blocked: dict[int, int] = {}
    assert apply_lock_blocked_override(2, blocked, set()) == 2


def test_none_passes_through() -> None:
    assert apply_lock_blocked_override(None, {0: 1}, set()) is None


def test_overrides_to_live_holder() -> None:
    # Worker 0 is blocked waiting on a lock held by worker 1; schedule the holder.
    blocked = {0: 1}
    assert apply_lock_blocked_override(0, blocked, set()) == 1
    assert blocked == {0: 1}  # entry retained while the holder is still live


def test_stale_holder_is_dropped_and_scheduled_proceeds() -> None:
    # The holder (1) has finished, so the blocked entry is stale: drop it and
    # let the engine's original choice (0) proceed.
    blocked = {0: 1}
    assert apply_lock_blocked_override(0, blocked, {1}) == 0
    assert blocked == {}


def test_unrelated_blocked_entries_are_left_untouched() -> None:
    blocked = {0: 1, 5: 6}
    assert apply_lock_blocked_override(0, blocked, set()) == 1
    assert blocked == {0: 1, 5: 6}
