"""Access-anchored replay tolerates positional drift without confusing sibling objects.

Regressions #20 and #22: run-varying work changes opcode counts; accesses to
same-class, non-racing siblings must not consume the racing object's anchors.
"""

from __future__ import annotations

import random

import pytest

import frontrun


class Holder:
    def __init__(self) -> None:
        self.cb: int | None = None


class State:
    def __init__(self) -> None:
        self.shared = Holder()
        self.mine = [Holder(), Holder()]
        self.results: list[int | None] = []


def _variable_work() -> int:
    # Model subprocess/polling latency with a different traced-opcode count
    # on every exploration and replay run.
    acc = 0
    for _ in range(random.randint(50, 800)):
        acc += 1
    return acc


def _make_worker(tid: int, touch_siblings: bool):
    def worker(s: State) -> None:
        s.shared.cb = tid
        _variable_work()
        if touch_siblings:
            helper = s.mine[tid]
            for _ in range(random.randint(1, 6)):
                # Alternating read/write kinds defeat duplicate collapsing.
                helper.cb = (helper.cb or 0) + 1
            _variable_work()
        s.results.append(s.shared.cb)

    return worker


@pytest.mark.parametrize("touch_siblings", [False, True], ids=["opcode-drift", "same-class-siblings"])
def test_access_anchors_reproduce_write_read_race(touch_siblings: bool) -> None:
    result = frontrun.explore(
        setup=State,
        workers=[_make_worker(0, touch_siblings), _make_worker(1, touch_siblings)],
        invariant=lambda s: not (len(s.results) == 2 and len(set(s.results)) == 1),
        detect_io=False,
        reproduce_on_failure=10,
    )
    assert not result.property_holds, "DPOR failed to detect the write-read race"
    assert result.reproduction_successes >= 8, (
        f"replay reproduced only {result.reproduction_successes}/{result.reproduction_attempts}"
    )
