"""Contracts for the shared sync/async DPOR core."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock, call

import pytest

from frontrun._dpor_core import NoOpLock, RowLockRegistry, dpor_exploration_iter
from frontrun.common import InterleavingResult


def test_row_lock_registry_ids_are_stable_monotonic_and_instance_local() -> None:
    registry = RowLockRegistry()
    resources = [f"row:{i}" for i in range(5)]
    assert [registry._row_lock_int_id(resource) for resource in resources] == list(range(5))
    assert [registry._row_lock_int_id(resource) for resource in resources] == list(range(5))
    assert registry.id_to_resource() == dict(enumerate(resources))
    assert RowLockRegistry()._row_lock_int_id("row:4") == 0


@pytest.mark.parametrize("with_graph", [False, True])
def test_row_lock_registry_acquire_and_release(with_graph: bool) -> None:
    registry = RowLockRegistry()
    graph = Mock() if with_graph else None
    for resource in ["row:1", "row:2"]:
        registry.record_acquire(3, resource, graph)
        assert registry.active_lock_owner(resource) == 3
    assert registry._task_row_locks == {3: {"row:1", "row:2"}}
    if graph is not None:
        assert graph.add_holding.call_args_list == [call(3, 0, kind="row_lock"), call(3, 1, kind="row_lock")]

    assert registry.pop_all(3, graph) == [("row:1", 0), ("row:2", 1)]
    assert registry._task_row_locks == {}
    assert registry._active_row_locks == {}
    assert registry.pop_all(99, graph) == []
    if graph is not None:
        assert graph.remove_holding.call_args_list == [call(3, 0, kind="row_lock"), call(3, 1, kind="row_lock")]


def test_row_lock_registry_pop_selected_preserves_other_holding() -> None:
    registry = RowLockRegistry()
    graph = Mock()
    registry.record_acquire(4, "row:prior", graph)
    registry.record_acquire(4, "row:missing", graph)
    missing_id = registry._row_lock_int_id("row:missing")

    assert registry.pop(4, graph, ["row:missing"]) == [("row:missing", missing_id)]
    assert registry.active_lock_owner("row:prior") == 4
    assert registry.active_lock_owner("row:missing") is None
    assert registry._task_row_locks[4] == {"row:prior"}
    graph.remove_holding.assert_called_once_with(4, missing_id, kind="row_lock")


def test_row_lock_registry_transfer_cannot_be_released_by_previous_owner() -> None:
    registry = RowLockRegistry()
    registry.record_acquire(0, "row:1", None)
    registry.record_acquire(1, "row:1", None)
    assert registry._task_row_locks[1] == {"row:1"}
    assert "row:1" not in registry._task_row_locks.get(0, set())
    assert registry.pop_all(0, None) == []
    assert registry.active_lock_owner("row:1") == 1


def test_noop_lock_is_reentrant_context_manager() -> None:
    lock = NoOpLock()
    with lock as outer:
        with lock as inner:
            assert outer is inner is None


def test_row_lock_registry_pop_all_returns_deterministic_order() -> None:
    from frontrun._dpor_core import RowLockRegistry

    class ReverseSet(set[str]):
        def __iter__(self) -> Any:
            return iter(sorted(set.copy(self), reverse=True))

    reg = RowLockRegistry()
    for i in range(3):
        reg.record_acquire(owner_id=7, res_id=f"row:{i}", graph=None)
    reg._task_row_locks[7] = ReverseSet(reg._task_row_locks[7])

    released = reg.pop_all(owner_id=7, graph=None)
    lids = [lid for _, lid in released]
    assert lids == [0, 1, 2]


def test_record_dpor_failure_contract() -> None:
    """Failures accumulate, while the first constructive counterexample wins."""
    from frontrun._dpor_core import record_dpor_failure

    result = InterleavingResult(property_holds=True)
    first, second, third = [0, 1, 0], [1, 0], [2]
    result.num_explored = 1
    assert record_dpor_failure(result, first, "first", races_detected=False) is first
    assert result.property_holds is False
    assert result.failures == [(1, first)]
    assert result.counterexample is first and result.explanation == "first"
    assert result.races_detected is False

    result.num_explored = 2
    assert record_dpor_failure(result, second, "second", races_detected=True) is second
    assert result.failures == [(1, first), (2, second)]
    assert result.counterexample is first and result.explanation == "first"
    assert result.races_detected is True

    result.num_explored = 3
    assert record_dpor_failure(result, third, "third", races_detected=False) is third
    assert result.failures == [(1, first), (2, second), (3, third)]
    assert result.counterexample is first and result.races_detected is True


class _StubExecution:
    """Sentinel returned by the fake engine for each exploration iteration."""


class _StubEngine:
    """Records `begin_execution` / `next_execution` calls and lock interactions."""

    def __init__(self, num_executions: int) -> None:
        self.num_executions = num_executions
        self._begin_calls = 0
        self._next_calls = 0
        self.current_lock: Any = None

    def begin_execution(self) -> _StubExecution:
        if self.current_lock is not None and not self.current_lock.entered:
            raise AssertionError("begin_execution must run while engine_lock is held")
        self._begin_calls += 1
        return _StubExecution()

    def next_execution(self) -> bool:
        if self.current_lock is not None and not self.current_lock.entered:
            raise AssertionError("next_execution must run while engine_lock is held")
        self._next_calls += 1
        return self._next_calls < self.num_executions


class _RecordingLock:
    """Context manager that records whether it is currently held."""

    def __init__(self) -> None:
        self.entered = False
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> _RecordingLock:
        self.entered = True
        self.enter_count += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        self.entered = False
        self.exit_count += 1


class _StubStableIds:
    def __init__(self) -> None:
        self.resets = 0

    def reset_for_execution(self) -> None:
        self.resets += 1


def _run_stub_exploration(
    num_executions: int, engine_lock: Any, *, total_deadline: float | None = None
) -> tuple[_StubEngine, _StubStableIds, list[Any]]:
    """Construct the common engine/lock/IDs harness and consume its steps."""
    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions)
    stable_ids = _StubStableIds()
    engine.current_lock = engine_lock if isinstance(engine_lock, _RecordingLock) else None
    seen = list(
        dpor_exploration_iter(
            engine=engine, engine_lock=engine_lock, stable_ids=stable_ids, total_deadline=total_deadline
        )
    )
    return engine, stable_ids, seen


@pytest.mark.parametrize("lock_factory", [_RecordingLock, threading.Lock, NoOpLock])
def test_dpor_exploration_iteration_contract(lock_factory: Any) -> None:
    lock = lock_factory()
    engine = _StubEngine(3)
    stable_ids = _StubStableIds()
    if isinstance(lock, _RecordingLock):
        engine.current_lock = lock
    steps = []
    for step in dpor_exploration_iter(engine=engine, engine_lock=lock, stable_ids=stable_ids, total_deadline=None):
        steps.append(step)
        assert isinstance(step.execution, _StubExecution)
        if isinstance(lock, _RecordingLock):
            assert not lock.entered
    assert [step.index for step in steps] == [1, 2, 3]
    assert engine._begin_calls == engine._next_calls == stable_ids.resets == 3
    if isinstance(lock, _RecordingLock):
        assert lock.enter_count == lock.exit_count == 6


def test_dpor_exploration_iter_permits_baseline_after_total_deadline() -> None:
    """An elapsed positive budget still permits the required baseline execution."""
    import time

    lock = _RecordingLock()
    past = time.monotonic() - 1.0
    engine, _, seen = _run_stub_exploration(10, lock, total_deadline=past)
    assert [step.index for step in seen] == [1]
    assert engine._begin_calls == 1


def test_dpor_exploration_iter_stops_when_deadline_expires_mid_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the deadline expires after some iterations, the loop exits cleanly."""
    from frontrun._dpor_core import concurrency

    # Fake clock that advances by 1.0 on each call.
    fake_now = [0.0]

    def _monotonic() -> float:
        t = fake_now[0]
        fake_now[0] += 1.0
        return t

    monkeypatch.setattr(concurrency.time, "monotonic", _monotonic)

    lock = _RecordingLock()

    # The first iteration is guaranteed without consulting the clock. Each
    # subsequent iteration checks after both its body and path planning.
    engine, _, seen = _run_stub_exploration(10, lock, total_deadline=2.5)
    assert len(seen) == 2
    assert engine._next_calls == 2, "the engine must not run a schedule planned after the deadline expires"


def test_dpor_exploration_iter_does_not_run_schedule_planned_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow next_execution() must not authorize another over-budget run."""
    from frontrun._dpor_core import concurrency, dpor_exploration_iter

    fake_now = [0.0]
    monkeypatch.setattr(concurrency.time, "monotonic", lambda: fake_now[0])

    class SlowPlanningEngine(_StubEngine):
        def next_execution(self) -> bool:
            result = super().next_execution()
            fake_now[0] = 2.0
            return result

    engine = SlowPlanningEngine(num_executions=2)
    lock = _RecordingLock()
    engine.current_lock = lock

    seen = list(
        dpor_exploration_iter(
            engine=engine,
            engine_lock=lock,
            stable_ids=_StubStableIds(),
            total_deadline=1.0,
        )
    )

    assert [step.index for step in seen] == [1]
    assert engine._begin_calls == 1
    assert engine._next_calls == 1


def test_advance_replay_index_recheck_bounds_after_extend_fn() -> None:
    """extend_fn returning True without adding entries must not cause IndexError.

    Bug: after extend_fn() returns True the code falls through to
    replay_schedule[replay_index] without re-checking the bounds, so an
    extend_fn that returns True but adds nothing triggers an IndexError.
    """
    from frontrun._dpor_core.utils import advance_replay_index

    call_count = 0

    def extend_noop_then_stop() -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return True
        return False

    replay_schedule: list[int] = [0]
    new_index, actor = advance_replay_index(
        replay_schedule=replay_schedule,
        replay_index=1,
        extend_fn=extend_noop_then_stop,
        actors_done={0},
    )
    assert actor is None
    assert call_count == 2
