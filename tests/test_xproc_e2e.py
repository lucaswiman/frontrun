"""End-to-end cross-process exploration tests.

These spawn *real* subprocesses (via the public ``frontrun.explore_processes``
API) that open their own connections to a shared SQLite file and run plain
``sqlite3`` code — no explicit scheduling calls in the user code. The
coordinator interleaves them at SQL-statement granularity and checks an
invariant over the real database after each interleaving.

Marked ``e2e`` (opt-in via ``make test-e2e-3.14`` / ``pytest -m e2e``) because
each interleaving spawns processes.
"""

from __future__ import annotations

import pytest

import frontrun
from frontrun._dpor_runtime.xproc import _demo_counter

pytestmark = pytest.mark.e2e

_TARGET = "frontrun._dpor_runtime.xproc._demo_counter:increment"
_ATOMIC_TARGET = "frontrun._dpor_runtime.xproc._demo_counter:increment_atomic"


def test_lost_update_race_found_across_processes(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_TARGET, (db,)),
            "w1": frontrun.Subprocess(_TARGET, (db,)),
        },
        setup=lambda: _demo_counter.setup(db),
        invariant=lambda: _demo_counter.read(db) == 2,
        max_iterations=50,
    )
    assert not result.ok, "expected the lost-update interleaving to be found"
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None
    # The read/write accesses of both workers must have reached the coordinator.
    assert result.accesses is not None
    assert any(kind == "write" for _wid, _rid, kind in result.accesses)


def test_atomic_increment_has_no_race(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_ATOMIC_TARGET, (db,)),
            "w1": frontrun.Subprocess(_ATOMIC_TARGET, (db,)),
        },
        setup=lambda: _demo_counter.setup(db),
        invariant=lambda: _demo_counter.read(db) == 2,
        max_iterations=50,
    )
    assert result.ok, f"unexpected failure {result.failure!r} at {result.failing_schedule!r}"
    assert result.exhausted
    # A single atomic UPDATE per worker => exactly two interleavings.
    assert result.iterations == 2
