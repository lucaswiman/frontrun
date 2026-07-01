"""frontrun.explore(execution="process") — the multiprocessing mirror of threads.

Same call shape as the threading/async interface: setup / workers / invariant,
with count= to replicate a worker. The only differences are inherent to
processes — workers are picklable module-level callables and setup() returns a
picklable handle to the shared *external* state (here a SQLite path). Spawns real
processes, so marked e2e.
"""

from __future__ import annotations

import pytest

import frontrun
from frontrun._dpor_runtime.xproc import _demo_counter

pytestmark = pytest.mark.e2e


def _make_setup(db: str):
    def setup():
        _demo_counter.setup(db)
        return db  # picklable handle passed to each worker(state) and invariant(state)

    return setup


def test_process_execution_finds_lost_update(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore(
        setup=_make_setup(db),
        workers=_demo_counter.increment,
        count=2,
        invariant=lambda state: _demo_counter.read(state) == 2,
        execution="process",
    )
    assert not result.property_holds
    assert result.counterexample is not None
    assert result.explanation


def test_process_execution_atomic_increment_holds(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore(
        setup=_make_setup(db),
        workers=[_demo_counter.increment_atomic, _demo_counter.increment_atomic],
        invariant=lambda state: _demo_counter.read(state) == 2,
        execution="process",
    )
    assert result.property_holds, result.explanation
    assert result.num_explored >= 1


def test_process_execution_rejects_async_workers() -> None:
    async def worker(state):  # noqa: RUF029 - intentionally async to trigger the guard
        return None

    with pytest.raises(ValueError, match="does not support async"):
        frontrun.explore(setup=lambda: None, workers=worker, count=2, invariant=lambda s: True, execution="process")
