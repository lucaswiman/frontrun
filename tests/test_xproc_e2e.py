"""End-to-end cross-process exploration tests.

These spawn *real* subprocesses (via the public ``frontrun.explore_processes``
API) that open their own connections to a shared SQLite file and run plain
``sqlite3`` code — no explicit scheduling calls in the user code. The
coordinator interleaves them at SQL-statement granularity and checks an
invariant over the real database after each interleaving.

Marked ``e2e`` (selectable on their own via ``make test-e2e-3.14`` /
``pytest -m e2e``; they also run in the default suite) because each
interleaving spawns processes.
"""

from __future__ import annotations

import socket

import pytest

import frontrun
from frontrun._dpor_runtime.xproc.coordinator import accept_hello, worker_targets
from frontrun._dpor_runtime.xproc.launch import SubprocessLauncher
from tests import xproc_demo_counter

pytestmark = pytest.mark.e2e

_TARGET = "tests.xproc_demo_counter:increment"
_ATOMIC_TARGET = "tests.xproc_demo_counter:increment_atomic"
_THREADED_READ_TARGET = "tests.xproc_demo_counter:read_in_joined_thread"


def test_joined_child_thread_sql_access_is_scheduled(tmp_path) -> None:
    """Sequential cross-thread handoff must retain the process scheduler context."""
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        frontrun.Subprocess(_THREADED_READ_TARGET, (db,)),
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: False,
    )

    assert not result.ok
    assert result.accesses, "the joined child thread's SQL access was invisible to xproc"
    assert any(resource.startswith("sql:") for _worker, resource, _kind in result.accesses)


def test_lost_update_race_found_across_processes(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_TARGET, (db,)),
            "w1": frontrun.Subprocess(_TARGET, (db,)),
        },
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
    )
    assert not result.ok, "expected the lost-update interleaving to be found"
    assert result.failure_kind == "invariant"
    assert result.failing_schedule is not None
    # The read/write accesses of both workers must have reached the coordinator.
    assert result.accesses is not None
    assert any(kind == "write" for _wid, _rid, kind in result.accesses)


def test_lost_update_found_with_exhaustive_strategy(tmp_path) -> None:
    # The reduction-free strategy must reach the same verdict end-to-end.
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {"w0": frontrun.Subprocess(_TARGET, (db,)), "w1": frontrun.Subprocess(_TARGET, (db,))},
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
        strategy="exhaustive",
        max_iterations=50,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"


def test_lost_update_found_with_reused_workers(tmp_path) -> None:
    # Persistent workers (reuse_workers=True) must reach the same verdict while
    # re-running the target in place instead of respawning each interleaving.
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {"w0": frontrun.Subprocess(_TARGET, (db,)), "w1": frontrun.Subprocess(_TARGET, (db,))},
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
        reuse_workers=True,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"


def test_poisoned_subprocess_set_is_reaped_before_fresh_launch(tmp_path) -> None:
    """Real persistent children are dead before replacement HELLOs arrive."""
    socket_path = str(tmp_path / "xproc.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(1)
    listener.settimeout(5.0)
    launcher = SubprocessLauncher([frontrun.Subprocess(_TARGET, ("unused.db",))], reuse=True)
    targets = worker_targets(socket_path, [0])
    first = launcher.launch(targets)
    first_sock, first_worker_id = accept_hello(listener, 5.0)
    second = None
    second_sock = None
    try:
        assert first_worker_id == 0
        first_pid = first[0].pid
        first_sock.close()  # model a poisoned/desynchronised protocol stream
        launcher.terminate(first, timeout=5.0)
        assert first[0].poll() is not None

        second = launcher.launch(targets)
        second_sock, second_worker_id = accept_hello(listener, 5.0)
        assert second_worker_id == 0
        assert second[0].pid != first_pid
        assert first[0].poll() is not None
    finally:
        first_sock.close()
        if second_sock is not None:
            second_sock.close()
        if second is not None:
            launcher.terminate(second, timeout=5.0)
        listener.close()


def test_count_shorthand_replicates_spec(tmp_path) -> None:
    # A single Subprocess with count=N mirrors explore(workers=fn, count=N).
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        frontrun.Subprocess(_TARGET, (db,)),
        count=2,
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"


def test_invariant_receives_setup_state_handle(tmp_path) -> None:
    # explore_processes must thread setup()'s return value into invariant(state),
    # matching explore(execution="process"). Both run in the coordinator process,
    # so we can assert object identity of the handle.
    db = str(tmp_path / "counter.db")
    handle = {"db": db}  # arbitrary state handle returned by setup()
    received: list[object] = []

    def setup() -> object:
        xproc_demo_counter.setup(db)
        return handle

    def invariant(state: object) -> bool:
        received.append(state)
        return xproc_demo_counter.read(db) == 2

    frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_TARGET, (db,)),
            "w1": frontrun.Subprocess(_TARGET, (db,)),
        },
        setup=setup,
        invariant=invariant,
    )
    assert received, "invariant was never called"
    assert all(s is handle for s in received), "invariant must receive the setup() return value"


def test_max_executions_cap_reports_not_exhausted(tmp_path) -> None:
    # max_executions truncates the DPOR search. `exhausted` is the tool's coverage
    # guarantee, so a capped run must NOT claim the space was fully covered — else
    # a user bounding a long search and trusting `exhausted` gets false assurance.
    # Two atomic (race-free) workers still have >1 interleaving to explore, so
    # capping at 1 leaves the space unexplored.
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {"w0": frontrun.Subprocess(_ATOMIC_TARGET, (db,)), "w1": frontrun.Subprocess(_ATOMIC_TARGET, (db,))},
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
        max_executions=1,
    )
    assert result.iterations == 1
    assert not result.exhausted, "a max_executions-capped search must report exhausted=False"


def test_bad_target_reports_real_cause_quickly() -> None:
    # A typo'd target must fail fast with the child's real error, not block the
    # full connect budget (deadlock_timeout*2+10) before a bare timeout.
    import time

    start = time.monotonic()
    result = frontrun.explore_processes(
        frontrun.Subprocess("frontrun_no_such_module:go"),
        count=2,
        setup=lambda: None,
        invariant=lambda _state: True,
        deadlock_timeout=2.0,
    )
    elapsed = time.monotonic() - start
    assert result.failure_kind == "worker_error"
    assert "No module named" in (result.failure or "") or "ModuleNotFoundError" in (result.failure or "")
    assert elapsed < 10.0, f"bad-target detection took {elapsed:.1f}s (should fast-fail)"


def test_atomic_increment_has_no_race(tmp_path) -> None:
    db = str(tmp_path / "counter.db")
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_ATOMIC_TARGET, (db,)),
            "w1": frontrun.Subprocess(_ATOMIC_TARGET, (db,)),
        },
        setup=lambda: xproc_demo_counter.setup(db),
        invariant=lambda _state: xproc_demo_counter.read(db) == 2,
        preemption_bound=None,
    )
    assert result.ok, f"unexpected failure {result.failure!r} at {result.failing_schedule!r}"
    assert result.exhausted
    assert result.iterations >= 1
