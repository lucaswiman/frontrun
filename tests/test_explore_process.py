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


def test_process_execution_reuse_finds_lost_update(tmp_path) -> None:
    # reuse_workers spawns each process once and re-runs it per interleaving; it
    # must reach the same lost-update verdict as spawn-per-iteration.
    db = str(tmp_path / "counter.db")
    result = frontrun.explore(
        setup=_make_setup(db),
        workers=_demo_counter.increment,
        count=2,
        invariant=lambda state: _demo_counter.read(state) == 2,
        execution="process",
        reuse_workers=True,
    )
    assert not result.property_holds
    assert result.counterexample is not None


def test_process_execution_accepts_closure_workers(tmp_path) -> None:
    # dill serialisation lets execution="process" take a locally-defined closure
    # (not just a module-level function), matching what thread execution accepts.
    import sqlite3

    db = str(tmp_path / "counter.db")

    def increment(state) -> None:
        conn = sqlite3.connect(state, isolation_level=None)
        try:
            val = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
            conn.execute("UPDATE counter SET val = ? WHERE id = 1", (val + 1,))
        finally:
            conn.close()

    result = frontrun.explore(
        setup=_make_setup(db),
        workers=increment,
        count=2,
        invariant=lambda state: _demo_counter.read(state) == 2,
        execution="process",
    )
    assert not result.property_holds
    assert result.counterexample is not None


def test_process_execution_rejects_async_workers() -> None:
    async def worker(state):  # noqa: RUF029 - intentionally async to trigger the guard
        return None

    with pytest.raises(ValueError, match="does not support async"):
        frontrun.explore(setup=lambda: None, workers=worker, count=2, invariant=lambda s: True, execution="process")
