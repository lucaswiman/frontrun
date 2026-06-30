"""Test that DPOR lock IDs are stable (indexical) across executions.

Regression test: lock IDs used to be raw ``id(self)`` values from
``CooperativeLock``, which changed between executions because each
execution creates fresh state via ``setup()``.  The fix uses
``StableObjectIds`` to assign monotonically increasing IDs that are
consistent across executions.
"""

from __future__ import annotations

import re
from pathlib import Path

from frontrun import explore
from frontrun._cooperative import CooperativeLock


class DiningState:
    def __init__(self) -> None:
        self.fork_a = CooperativeLock()
        self.fork_b = CooperativeLock()
        self.meals: list[str] = []


def test_lock_ids_stable_across_executions(tmp_path: Path) -> None:
    """Lock IDs in the HTML report must be the same across all executions."""

    def setup() -> DiningState:
        return DiningState()

    def philosopher_0(state: DiningState) -> None:
        with state.fork_a:
            with state.fork_b:
                state.meals.append("P0")

    def philosopher_1(state: DiningState) -> None:
        with state.fork_b:
            with state.fork_a:
                state.meals.append("P1")

    def invariant(state: DiningState) -> bool:
        return True

    import frontrun._report as _report_mod

    report_file = tmp_path / "lock_stability.html"
    old_path = _report_mod._global_report_path
    _report_mod._global_report_path = str(report_file)
    try:
        result = explore(
            setup=setup,
            workers=[philosopher_0, philosopher_1],
            invariant=invariant,
        )
    finally:
        _report_mod._global_report_path = old_path

    assert result.num_explored >= 2, "Need multiple executions to test stability"

    html = report_file.read_text()
    lock_ids = {int(x) for x in re.findall(r'"lock_id":\s*(\d+)', html)}

    # With 2 locks we should see exactly 2 unique lock IDs across ALL executions.
    assert len(lock_ids) == 2, (
        f"Expected exactly 2 unique lock IDs across all executions, got {len(lock_ids)}: {sorted(lock_ids)}"
    )

    # Lock IDs should be small stable integers, not raw memory addresses.
    for lid in lock_ids:
        assert lid < 1000, f"Lock ID {lid} looks like a raw memory address, not a stable ID"


def test_lock_race_objects_have_readable_names(tmp_path: Path) -> None:
    """Race info should show human-readable lock names, not opaque object IDs."""
    import frontrun._report as _report_mod

    def setup() -> DiningState:
        return DiningState()

    def philosopher_0(state: DiningState) -> None:
        with state.fork_a:
            with state.fork_b:
                state.meals.append("P0")

    def philosopher_1(state: DiningState) -> None:
        with state.fork_b:
            with state.fork_a:
                state.meals.append("P1")

    def invariant(state: DiningState) -> bool:
        return True

    report_file = tmp_path / "lock_names.html"
    old_path = _report_mod._global_report_path
    _report_mod._global_report_path = str(report_file)
    try:
        result = explore(
            setup=setup,
            workers=[philosopher_0, philosopher_1],
            invariant=invariant,
        )
    finally:
        _report_mod._global_report_path = old_path

    html = report_file.read_text()
    race_objects = set(re.findall(r'"object":\s*"([^"]+)"', html))

    # Filter to lock-related race objects
    lock_races = {o for o in race_objects if "Lock" in o or "lock" in o}

    if lock_races:
        for name in lock_races:
            assert "CooperativeLock" in name, f"Lock race object should mention CooperativeLock, got: {name}"
            # Should NOT be just "object <number>"
            assert not name.startswith("object "), f"Lock race object should have a readable name, not: {name}"
