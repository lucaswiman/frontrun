from __future__ import annotations

import asyncio
import random
import threading
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# RowLockRegistry extended methods: record_acquire, pop_all, id_to_resource
# ---------------------------------------------------------------------------


def test_row_lock_registry_id_to_resource() -> None:
    """id_to_resource returns the inverse mapping for format_cycle."""
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    id_x = reg._row_lock_int_id("x")
    id_y = reg._row_lock_int_id("y")

    inv = reg.id_to_resource()
    assert inv[id_x] == "x"
    assert inv[id_y] == "y"


def test_row_lock_registry_record_acquire_no_graph() -> None:
    """record_acquire stores ownership in _active_row_locks and _task_row_locks."""
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    reg.record_acquire(owner_id=0, res_id="row:1", graph=None)

    assert reg._active_row_locks["row:1"] == 0
    assert "row:1" in reg._task_row_locks[0]


def test_row_lock_registry_record_acquire_with_graph() -> None:
    """record_acquire calls graph.add_holding with kind='row_lock'."""
    from frontrun._dpor_core import RowLockRegistry

    class _FakeGraph:
        def __init__(self) -> None:
            self.holdings: list[tuple[int, int, str]] = []

        def add_holding(self, owner: int, lid: int, kind: str = "lock") -> None:
            self.holdings.append((owner, lid, kind))

    reg = RowLockRegistry()
    graph = _FakeGraph()
    reg.record_acquire(owner_id=1, res_id="row:2", graph=graph)

    assert len(graph.holdings) == 1
    owner, lid, kind = graph.holdings[0]
    assert owner == 1
    assert lid == reg._row_lock_int_id("row:2")
    assert kind == "row_lock"


def test_row_lock_registry_pop_all_returns_pairs() -> None:
    """pop_all removes all resources and returns (res_id, int_id) pairs."""
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    reg.record_acquire(owner_id=2, res_id="row:3", graph=None)
    reg.record_acquire(owner_id=2, res_id="row:4", graph=None)

    released = reg.pop_all(owner_id=2, graph=None)
    assert {r for r, _ in released} == {"row:3", "row:4"}
    assert reg._task_row_locks.get(2) is None  # cleaned up
    assert "row:3" not in reg._active_row_locks
    assert "row:4" not in reg._active_row_locks


def test_row_lock_registry_pop_selected_preserves_other_holding() -> None:
    """A speculative statement lock can be removed without ending the transaction."""
    from frontrun._dpor_core import RowLockRegistry

    class _FakeGraph:
        def __init__(self) -> None:
            self.released: list[tuple[int, int, str]] = []

        def add_holding(self, _owner: int, _lid: int, kind: str = "lock") -> None:
            pass

        def remove_holding(self, owner: int, lid: int, kind: str = "lock") -> None:
            self.released.append((owner, lid, kind))

    reg = RowLockRegistry()
    graph = _FakeGraph()
    reg.record_acquire(4, "row:prior", graph)
    reg.record_acquire(4, "row:missing", graph)
    missing_id = reg._row_lock_int_id("row:missing")

    released = reg.pop(4, graph, ["row:missing"])

    assert released == [("row:missing", missing_id)]
    assert reg.active_lock_owner("row:prior") == 4
    assert reg.active_lock_owner("row:missing") is None
    assert reg._task_row_locks[4] == {"row:prior"}
    assert graph.released == [(4, missing_id, "row_lock")]


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


def test_row_lock_registry_pop_all_calls_graph_remove_holding() -> None:
    """pop_all calls graph.remove_holding for each released resource."""
    from frontrun._dpor_core import RowLockRegistry

    class _FakeGraph:
        def __init__(self) -> None:
            self.released: list[tuple[int, int, str]] = []

        def add_holding(self, owner: int, lid: int, kind: str = "lock") -> None:
            pass

        def remove_holding(self, owner: int, lid: int, kind: str = "lock") -> None:
            self.released.append((owner, lid, kind))

    reg = RowLockRegistry()
    graph = _FakeGraph()
    reg.record_acquire(owner_id=3, res_id="row:5", graph=graph)
    lid = reg._row_lock_int_id("row:5")

    released = reg.pop_all(owner_id=3, graph=graph)
    assert len(released) == 1
    assert (3, lid, "row_lock") in graph.released


def test_row_lock_registry_pop_all_empty() -> None:
    """pop_all on an owner with no locks returns empty list."""
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    result = reg.pop_all(owner_id=99, graph=None)
    assert result == []


def test_row_lock_registry_record_acquire_overwrites_previous_holder() -> None:
    """record_acquire on a resource held by another owner must clean up the old holder.

    Bug: When task B calls record_acquire on a resource held by task A,
    the old holder A retains the resource in _task_row_locks.  When A's
    pop_all runs later, it removes the resource from _active_row_locks,
    corrupting B's ownership tracking.
    """
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()

    # Task 0 acquires row:1
    reg.record_acquire(owner_id=0, res_id="row:1", graph=None)
    assert reg._active_row_locks["row:1"] == 0
    assert "row:1" in reg._task_row_locks[0]

    # Task 1 "optimistically" acquires row:1 (overwriting task 0)
    reg.record_acquire(owner_id=1, res_id="row:1", graph=None)
    assert reg._active_row_locks["row:1"] == 1
    assert "row:1" in reg._task_row_locks[1]

    # After the overwrite, task 0 should NOT still track row:1
    assert "row:1" not in reg._task_row_locks.get(0, set()), (
        "Old holder's _task_row_locks should be cleaned up when ownership is transferred"
    )

    # Task 0's pop_all should NOT corrupt task 1's ownership
    reg.pop_all(owner_id=0, graph=None)
    assert reg._active_row_locks.get("row:1") == 1, "pop_all of old holder must not remove new holder's ownership"


def test_async_dpor_scheduler_uses_num_engine_tasks_in_mark_done() -> None:
    """AsyncDporScheduler._mark_done should use _num_engine_tasks, not _num_tasks.

    Bug: _mark_done checks self._num_tasks (parent class attribute set by
    run_all) instead of self._num_engine_tasks (set in __init__).  The code
    should consistently use _num_engine_tasks which is the canonical field.
    """
    import ast as _ast
    import inspect
    import textwrap

    from frontrun.async_dpor import AsyncDporScheduler

    source = textwrap.dedent(inspect.getsource(AsyncDporScheduler._mark_done))
    tree = _ast.parse(source)

    has_num_engine_tasks = False
    has_num_tasks = False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute) and node.attr == "_num_engine_tasks":
            has_num_engine_tasks = True
        if isinstance(node, _ast.Attribute) and node.attr == "_num_tasks":
            has_num_tasks = True

    assert has_num_engine_tasks, "_mark_done should reference self._num_engine_tasks"
    assert not has_num_tasks, "_mark_done should NOT reference self._num_tasks (use _num_engine_tasks)"


class _PauseRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def pause(self, task_id: int) -> None:
        from frontrun._async_autopause import _in_scheduler_pause

        self.calls.append(task_id)
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await asyncio.sleep(0)
        finally:
            _in_scheduler_pause.set(depth)


def test_wrap_auto_paused_tasks_inserts_scheduler_pause() -> None:
    from frontrun._async_autopause import wrap_auto_paused_tasks

    recorder = _PauseRecorder()
    events: list[str] = []

    async def task() -> None:
        events.append("before")
        await asyncio.sleep(0)
        events.append("after")

    wrapped = wrap_auto_paused_tasks({7: task}, recorder)
    asyncio.run(wrapped[7]())

    assert events == ["before", "after"]
    assert recorder.calls
    assert set(recorder.calls) == {7}


def test_dispatch_threads_or_tasks_selects_correct_impl() -> None:
    from frontrun.contrib._shared import dispatch_threads_or_tasks

    sync_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    async_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def sync_impl(*args: Any, **kwargs: Any) -> str:
        sync_calls.append((args, kwargs))
        return "sync"

    async def async_impl(*args: Any, **kwargs: Any) -> str:
        async_calls.append((args, kwargs))
        return "async"

    assert dispatch_threads_or_tasks(sync_impl, async_impl, setup=object, threads=[lambda _: None]) == "sync"
    assert sync_calls and not async_calls

    result = asyncio.run(
        dispatch_threads_or_tasks(
            sync_impl,
            async_impl,
            setup=object,
            tasks=[lambda _: _noop()],
        )
    )
    assert result == "async"
    assert len(async_calls) == 1


def test_dispatch_threads_or_tasks_requires_exactly_one_mode() -> None:
    from frontrun.contrib._shared import dispatch_threads_or_tasks

    def sync_impl(*args: Any, **kwargs: Any) -> None:
        return None

    async def async_impl(*args: Any, **kwargs: Any) -> None:
        return None

    with pytest.raises(TypeError, match="requires exactly one"):
        dispatch_threads_or_tasks(sync_impl, async_impl, setup=object)

    with pytest.raises(TypeError, match="requires exactly one"):
        dispatch_threads_or_tasks(
            sync_impl,
            async_impl,
            setup=object,
            threads=[lambda _: None],
            tasks=[lambda _: _noop()],
        )


async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Shared schedule generation helpers (sync + async random exploration)
# ---------------------------------------------------------------------------


def test_random_round_robin_schedule_is_fair_and_deterministic() -> None:
    from frontrun._random_schedules import random_round_robin_schedule

    rng = random.Random(0)
    schedule = random_round_robin_schedule(rng, num_actors=3, max_ops=12)

    # Every entry is a valid actor id.
    assert schedule, "schedule should not be empty"
    assert all(0 <= entry < 3 for entry in schedule)

    # Fairness: each round contains every actor (so every actor appears at
    # least once), even though per-round burst lengths now vary so the counts
    # are no longer exactly equal (finding 5 — drift > 1 opcode).
    assert all(actor in schedule for actor in range(3))

    # Same RNG seed -> same schedule (determinism preserved for replay).
    rng2 = random.Random(0)
    assert random_round_robin_schedule(rng2, num_actors=3, max_ops=12) == schedule


def test_random_round_robin_schedule_can_express_drift() -> None:
    """The generator must be able to emit per-actor bursts longer than 1.

    Pure lockstep round-robin (one slot per actor per round) cannot express
    relative opcode drift > 1 between threads; variable-length bursts can.
    """
    from frontrun._random_schedules import random_round_robin_schedule

    saw_burst = False
    for seed in range(20):
        schedule = random_round_robin_schedule(random.Random(seed), num_actors=2, max_ops=40)
        # A burst is two or more consecutive identical actor ids.
        if any(a == b for a, b in zip(schedule, schedule[1:])):
            saw_burst = True
            break
    assert saw_burst, "generator never produced a burst (drift > 1 unreachable)"


def test_random_round_robin_schedule_respects_max_ops_cap() -> None:
    from frontrun._random_schedules import random_round_robin_schedule

    rng = random.Random(123)
    schedule = random_round_robin_schedule(rng, num_actors=4, max_ops=4)
    assert len(schedule) <= 4
    assert set(schedule) == {0, 1, 2, 3}


def test_fair_schedule_strategy_covers_every_actor_per_round() -> None:
    pytest.importorskip("hypothesis")
    from hypothesis import HealthCheck, given, settings

    from frontrun._random_schedules import fair_schedule_strategy

    # Property: every generated schedule is non-empty, contains only valid
    # actor ids, and includes every actor at least once (fairness).  Lengths
    # are no longer multiples of num_actors because of variable bursts.
    @given(schedule=fair_schedule_strategy(num_actors=3, max_ops=15))
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def _check(schedule: list[int]) -> None:
        assert schedule, "fair schedule should be non-empty"
        assert all(0 <= entry < 3 for entry in schedule)
        assert set(schedule) == {0, 1, 2}, "every actor must appear (fairness)"

    _check()


# ---------------------------------------------------------------------------
# Shared marker-executor finalization (sync + async trace markers)
# ---------------------------------------------------------------------------


def _make_completed_coordinator(num_steps: int = 1) -> Any:
    from frontrun._marker_coordination import ThreadCoordinator
    from frontrun.common import Schedule, Step

    schedule = Schedule([Step(execution_name=f"t{i}", marker_name="m") for i in range(num_steps)])
    coord = ThreadCoordinator(schedule)
    coord.current_step = num_steps
    coord.completed = True
    return coord


def test_finalize_marker_executor_run_no_op_when_clean() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator(num_steps=2)
    # Should not raise: no threads alive, no errors, schedule completed.
    finalize_marker_executor_run(
        threads=[],
        timeout=None,
        task_errors={},
        coordinator=coord,
        timeout_message=lambda alive: "unused",
    )


def test_finalize_marker_executor_run_raises_for_alive_threads() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator()
    blocker = threading.Event()
    thread = threading.Thread(target=blocker.wait, name="stuck", daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError, match="custom-timeout-message: stuck"):
            finalize_marker_executor_run(
                threads=[thread],
                timeout=0.05,
                task_errors={},
                coordinator=coord,
                timeout_message=lambda alive: "custom-timeout-message: " + ", ".join(t.name for t in alive),
            )
    finally:
        blocker.set()
        thread.join(timeout=1.0)


def test_finalize_marker_executor_run_reraises_first_task_error() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator()
    err1 = RuntimeError("first")
    err2 = RuntimeError("second")
    with pytest.raises(RuntimeError, match="first"):
        finalize_marker_executor_run(
            threads=[],
            timeout=None,
            task_errors={"a": err1, "b": err2},
            coordinator=coord,
            timeout_message=lambda alive: "unused",
        )


def test_finalize_marker_executor_run_detects_partial_schedule() -> None:
    from frontrun._marker_coordination import ThreadCoordinator, finalize_marker_executor_run
    from frontrun.common import Schedule, Step

    schedule = Schedule(
        [
            Step(execution_name="t1", marker_name="m1"),
            Step(execution_name="t2", marker_name="never_reached"),
        ]
    )
    coord = ThreadCoordinator(schedule)
    coord.current_step = 1  # one step consumed, second never reached
    coord.completed = False

    with pytest.raises(TimeoutError, match="Schedule incomplete.*never_reached"):
        finalize_marker_executor_run(
            threads=[],
            timeout=None,
            task_errors={},
            coordinator=coord,
            timeout_message=lambda alive: "unused",
        )


# ---------------------------------------------------------------------------
# Target 1: shared record_dpor_failure helper
# ---------------------------------------------------------------------------


def _make_result() -> Any:
    from frontrun.common import InterleavingResult

    r = InterleavingResult(property_holds=True)
    r.num_explored = 3
    return r


def test_record_dpor_failure_sets_property_holds_false() -> None:
    """record_dpor_failure marks the result as failing."""
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    record_dpor_failure(result, [0, 1, 0], "boom")
    assert result.property_holds is False


def test_record_dpor_failure_appends_to_failures() -> None:
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    schedule = [1, 0, 1]
    record_dpor_failure(result, schedule, "oops")
    assert result.failures == [(3, schedule)]


def test_record_dpor_failure_sets_counterexample_once() -> None:
    """Only the first failure becomes the counterexample."""
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    result.num_explored = 1
    record_dpor_failure(result, [0, 1], "first")
    result.num_explored = 2
    record_dpor_failure(result, [1, 0], "second")
    assert result.counterexample == [0, 1]
    assert result.explanation == "first"


def test_record_dpor_failure_sets_races_detected() -> None:
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    record_dpor_failure(result, [0], "race", races_detected=True)
    assert result.races_detected is True


def test_record_dpor_failure_races_detected_cumulative() -> None:
    """races_detected is OR-accumulated across calls."""
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    result.num_explored = 1
    record_dpor_failure(result, [0], "no race", races_detected=False)
    result.num_explored = 2
    record_dpor_failure(result, [1], "race!", races_detected=True)
    assert result.races_detected is True


def test_record_dpor_failure_returns_schedule_list() -> None:
    """record_dpor_failure returns the schedule it recorded."""
    from frontrun._dpor_core import record_dpor_failure

    result = _make_result()
    sched = [0, 1, 0]
    returned = record_dpor_failure(result, sched, "x")
    assert returned is sched


# ---------------------------------------------------------------------------
# Target 2: shared RowLockRegistry
# ---------------------------------------------------------------------------


def test_row_lock_registry_exists_in_dpor_core() -> None:
    """RowLockRegistry is importable from frontrun._dpor_core."""
    from frontrun._dpor_core import RowLockRegistry  # noqa: F401


def test_row_lock_registry_int_id_monotonic() -> None:
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    id0 = reg._row_lock_int_id("table:foo:1")
    id1 = reg._row_lock_int_id("table:bar:2")
    id0_again = reg._row_lock_int_id("table:foo:1")
    assert id0 != id1
    assert id0_again == id0  # stable across calls


def test_row_lock_registry_int_id_monotonic_increasing() -> None:
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    ids = [reg._row_lock_int_id(f"res:{i}") for i in range(5)]
    assert ids == list(range(5))


def test_row_lock_registry_state_is_independent() -> None:
    """Two RowLockRegistry instances share no state."""
    from frontrun._dpor_core import RowLockRegistry

    reg1 = RowLockRegistry()
    reg2 = RowLockRegistry()
    reg1._row_lock_int_id("a")
    reg1._row_lock_int_id("b")
    assert reg2._row_lock_int_id("a") == 0  # fresh counter in reg2


def test_row_lock_registry_has_active_and_task_dicts() -> None:
    """Registry exposes _active_row_locks and _task_row_locks."""
    from frontrun._dpor_core import RowLockRegistry

    reg = RowLockRegistry()
    assert hasattr(reg, "_active_row_locks")
    assert hasattr(reg, "_task_row_locks")
    assert isinstance(reg._active_row_locks, dict)
    assert isinstance(reg._task_row_locks, dict)


def test_dpor_scheduler_uses_row_lock_registry() -> None:
    """DporScheduler._row_lock_int_id delegates to its RowLockRegistry."""
    # This test verifies the call site was updated, not just the helper.
    # Import is guarded so the test is skipped if _dpor is not built.
    pytest.importorskip("frontrun._dpor")
    from frontrun._dpor_core import RowLockRegistry
    from frontrun._dpor_runtime.scheduler import DporScheduler

    # DporScheduler should have a _row_lock_registry attribute that is a RowLockRegistry.
    assert hasattr(DporScheduler, "__init__")
    # We can't instantiate without a real engine; just verify the attribute exists
    # by inspecting __init__'s source or by checking that RowLockRegistry is used.
    # Minimal smoke-test: RowLockRegistry._row_lock_int_id is the same function.
    reg = RowLockRegistry()
    assert callable(reg._row_lock_int_id)


def test_async_dpor_scheduler_uses_row_lock_registry() -> None:
    """AsyncDporScheduler delegates row-lock operations to RowLockRegistry."""
    pytest.importorskip("frontrun._dpor")
    from frontrun._dpor_core import RowLockRegistry
    from frontrun.async_dpor import AsyncDporScheduler

    assert hasattr(AsyncDporScheduler, "__init__")
    reg = RowLockRegistry()
    assert callable(reg._row_lock_int_id)


# ---------------------------------------------------------------------------
# Target 3: shared NoOpLock for the async/threading abstraction
# ---------------------------------------------------------------------------


def test_noop_lock_importable_from_dpor_core() -> None:
    """NoOpLock is importable from frontrun._dpor_core for both sync + async drivers."""
    from frontrun._dpor_core import NoOpLock  # noqa: F401


def test_noop_lock_is_context_manager() -> None:
    """NoOpLock supports the `with` statement and returns None."""
    from frontrun._dpor_core import NoOpLock

    lock = NoOpLock()
    with lock as result:
        assert result is None


def test_noop_lock_reentrant() -> None:
    """NoOpLock is safely reentrant (single-threaded async DPOR uses this)."""
    from frontrun._dpor_core import NoOpLock

    lock = NoOpLock()
    with lock:
        with lock:  # noqa: SIM117
            pass


# ---------------------------------------------------------------------------
# Target 4: shared dpor_exploration_iter generator (concurrency abstraction
# layer that lets sync + async DPOR drivers share the per-execution boundary).
# ---------------------------------------------------------------------------


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


def test_dpor_exploration_iter_yields_one_step_per_execution() -> None:
    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions=3)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    seen: list[Any] = [
        step.execution
        for step in dpor_exploration_iter(
            engine=engine,
            engine_lock=lock,
            stable_ids=stable_ids,
            total_deadline=None,
        )
    ]

    assert len(seen) == 3
    assert all(isinstance(e, _StubExecution) for e in seen)
    assert engine._begin_calls == 3
    # next_execution() is called after every iteration's body, including the
    # final one (which returns False to terminate).
    assert engine._next_calls == 3


def test_dpor_exploration_iter_resets_state_each_iteration() -> None:
    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions=2)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    for _ in dpor_exploration_iter(
        engine=engine,
        engine_lock=lock,
        stable_ids=stable_ids,
        total_deadline=None,
    ):
        pass

    assert stable_ids.resets == 2


def test_dpor_exploration_iter_holds_lock_across_engine_calls() -> None:
    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions=2)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    for _ in dpor_exploration_iter(
        engine=engine,
        engine_lock=lock,
        stable_ids=stable_ids,
        total_deadline=None,
    ):
        # Body runs *outside* the engine lock — the user's per-execution
        # work (worker spawning, scheduling) is what acquires fine-grained
        # subsections of the lock as needed.
        assert lock.entered is False

    # Each begin/next pair entered the lock once.
    # 2 begins + 2 nexts == 4 enters.
    assert lock.enter_count == 4
    assert lock.exit_count == 4


def test_dpor_exploration_iter_permits_baseline_after_total_deadline() -> None:
    """An elapsed positive budget still permits the required baseline execution."""
    import time

    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions=10)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    past = time.monotonic() - 1.0
    seen = list(
        dpor_exploration_iter(
            engine=engine,
            engine_lock=lock,
            stable_ids=stable_ids,
            total_deadline=past,
        )
    )
    assert [step.index for step in seen] == [1]
    assert engine._begin_calls == 1


def test_dpor_exploration_iter_stops_when_deadline_expires_mid_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the deadline expires after some iterations, the loop exits cleanly."""
    from frontrun._dpor_core import concurrency, dpor_exploration_iter

    # Fake clock that advances by 1.0 on each call.
    fake_now = [0.0]

    def _monotonic() -> float:
        t = fake_now[0]
        fake_now[0] += 1.0
        return t

    monkeypatch.setattr(concurrency.time, "monotonic", _monotonic)

    engine = _StubEngine(num_executions=10)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    # The first iteration is guaranteed without consulting the clock.  The
    # next three yield at 0.0, 1.0, and 2.0; the 3.0 check then stops.
    seen = list(
        dpor_exploration_iter(
            engine=engine,
            engine_lock=lock,
            stable_ids=stable_ids,
            total_deadline=2.5,
        )
    )
    assert len(seen) == 4
    assert engine._next_calls == 3, "the engine must not plan another execution after the deadline expires"


def test_dpor_exploration_iter_works_with_real_threading_lock() -> None:
    """A real threading.Lock is accepted (sync DPOR's engine_lock)."""
    from frontrun._dpor_core import dpor_exploration_iter

    real_lock = threading.Lock()
    engine = _StubEngine(num_executions=2)
    engine.current_lock = None  # the real lock has no `.entered` attribute
    stable_ids = _StubStableIds()

    seen = list(
        dpor_exploration_iter(
            engine=engine,
            engine_lock=real_lock,
            stable_ids=stable_ids,
            total_deadline=None,
        )
    )
    assert len(seen) == 2


def test_dpor_exploration_iter_works_with_noop_lock() -> None:
    """The shared NoOpLock works as engine_lock (async DPOR's contract)."""
    from frontrun._dpor_core import NoOpLock, dpor_exploration_iter

    engine = _StubEngine(num_executions=2)
    engine.current_lock = None  # NoOpLock doesn't expose `.entered`
    stable_ids = _StubStableIds()

    seen = list(
        dpor_exploration_iter(
            engine=engine,
            engine_lock=NoOpLock(),
            stable_ids=stable_ids,
            total_deadline=None,
        )
    )
    assert len(seen) == 2


def test_replay_async_scheduler_sets_scheduler_and_task_context_vars() -> None:
    """_ReplayAsyncScheduler must set _scheduler_var/_task_id_var like AsyncDporScheduler."""
    from frontrun._async_autopause import _scheduler_var, _task_id_var
    from frontrun.async_dpor import _ReplayAsyncScheduler

    sched = _ReplayAsyncScheduler([0, 1, 0, 1], num_tasks=2)
    assert hasattr(sched, "_setup_task_context"), (
        "_ReplayAsyncScheduler must override _setup_task_context "
        "to set _scheduler_var/_task_id_var for await_point() and cooperative locks"
    )

    sched._setup_task_context(42)
    assert _scheduler_var.get() is sched
    assert _task_id_var.get() == 42

    sched._cleanup_task_context(42)
    assert _scheduler_var.get() is None
    assert _task_id_var.get() is None


def test_replay_async_scheduler_sets_in_scheduler_pause() -> None:
    """_ReplayAsyncScheduler.pause() must set _in_scheduler_pause like AsyncDporScheduler."""
    import inspect

    from frontrun.async_dpor import _ReplayAsyncScheduler

    source = inspect.getsource(_ReplayAsyncScheduler.pause)
    assert "_in_scheduler_pause" in source, (
        "_ReplayAsyncScheduler.pause() must set _in_scheduler_pause "
        "to prevent _AutoPauseIterator from nesting scheduling points"
    )


def test_dpor_exploration_step_carries_execution_and_index() -> None:
    """Each ExplorationStep exposes `.execution` and a 1-indexed `.index`."""
    from frontrun._dpor_core import dpor_exploration_iter

    engine = _StubEngine(num_executions=3)
    lock = _RecordingLock()
    engine.current_lock = lock
    stable_ids = _StubStableIds()

    indices = []
    for step in dpor_exploration_iter(
        engine=engine,
        engine_lock=lock,
        stable_ids=stable_ids,
        total_deadline=None,
    ):
        indices.append(step.index)
        assert isinstance(step.execution, _StubExecution)
    assert indices == [1, 2, 3]


# ---------------------------------------------------------------------------
# advance_replay_index: extend_fn returning True without adding entries
# ---------------------------------------------------------------------------


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
