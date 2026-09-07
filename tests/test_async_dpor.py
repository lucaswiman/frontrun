"""Tests for async DPOR (Dynamic Partial Order Reduction).

Verifies that ``explore(strategy="dpor")`` on async tasks systematically
explores async interleavings using the Rust DPOR engine, with await points
as scheduling granularity.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import frontrun
from frontrun.cli import require_active
from frontrun.common import InterleavingResult

_async_global_counter = 0
_async_global_augmented = 0


@pytest.mark.parametrize(
    ("bridge_sync_io", "expected_calls"),
    [
        (False, ["async-patch", "async-unpatch"]),
        (True, ["sync-patch", "async-patch", "async-unpatch", "sync-unpatch"]),
    ],
)
def test_async_sql_scopes_sync_driver_patching_to_bridge(
    monkeypatch: pytest.MonkeyPatch,
    bridge_sync_io: bool,
    expected_calls: list[str],
) -> None:
    """Only adapters that bridge through sync workers patch sync SQL drivers."""
    import frontrun.async_dpor as async_dpor

    require_active("test_async_sql_scopes_sync_driver_patching_to_bridge")
    calls: list[str] = []
    monkeypatch.setattr(async_dpor, "patch_sql", lambda: calls.append("sync-patch"))
    monkeypatch.setattr(async_dpor, "unpatch_sql", lambda: calls.append("sync-unpatch"))
    monkeypatch.setattr(async_dpor, "patch_sql_async", lambda: calls.append("async-patch"))
    monkeypatch.setattr(async_dpor, "unpatch_sql_async", lambda: calls.append("async-unpatch"))

    async def worker(_state: object) -> None:
        return None

    result = asyncio.run(
        async_dpor._explore_async_dpor(
            setup=object,
            tasks=[worker],
            invariant=lambda _state: True,
            max_executions=1,
            reproduce_on_failure=0,
            detect_sql=True,
            _bridge_sync_io=bridge_sync_io,
        )
    )

    assert result.property_holds, result.explanation
    assert calls == expected_calls


def test_concurrent_async_exploration_is_rejected_before_global_patching() -> None:
    """A losing concurrent run must not unpatch the active run's primitives."""
    from frontrun._async_cooperative import _CooperativeAsyncEvent, _CooperativeAsyncLock

    started = threading.Event()
    release = threading.Event()
    first_result: list[object] = []
    first_errors: list[BaseException] = []

    class State:
        patches_intact = False

    async def holding_worker(state: State) -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0)
        state.patches_intact = asyncio.Event is _CooperativeAsyncEvent and asyncio.Lock is _CooperativeAsyncLock

    def run_first() -> None:
        try:
            first_result.append(
                asyncio.run(
                    frontrun.explore(
                        setup=State,
                        workers=[holding_worker],
                        invariant=lambda state: state.patches_intact,
                        strategy="dpor",
                        max_executions=1,
                        timeout_per_run=30.0,
                        deadlock_timeout=0.5,
                        reproduce_on_failure=0,
                        detect_io=False,
                    )
                )
            )
        except BaseException as exc:  # noqa: BLE001
            first_errors.append(exc)

    first = threading.Thread(target=run_first)
    first.start()
    assert started.wait(timeout=2.0)
    try:

        async def no_op(_state: object) -> None:
            return None

        with pytest.raises(RuntimeError, match="concurrent async exploration"):
            asyncio.run(
                frontrun.explore(
                    setup=object,
                    workers=[no_op],
                    invariant=lambda _state: True,
                    strategy="dpor",
                    max_executions=1,
                    reproduce_on_failure=0,
                    detect_io=False,
                )
            )
    finally:
        release.set()
        first.join(timeout=3.0)

    assert not first.is_alive()
    assert first_errors == []
    assert len(first_result) == 1
    result = first_result[0]
    assert isinstance(result, InterleavingResult)
    assert result.property_holds, result.explanation


class TestAsyncDporBasic:
    """Basic async DPOR functionality tests."""

    def test_finds_lost_update(self) -> None:
        """DPOR should systematically find the lost-update race."""
        require_active("test_async_dpor_lost_update")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the lost update"
        assert result.num_explored >= 1

    def test_no_race_when_atomic(self) -> None:
        """DPOR should verify correctness when there's no race."""
        require_active("test_async_dpor_no_race")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def safe_increment(counter: Counter) -> None:
            # Atomic: no await between read and write
            counter.value += 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[safe_increment, safe_increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds, f"No race expected: {result.counterexample}"
        assert result.num_explored >= 1

    def test_lock_handoff_survives_immediate_reacquire(self) -> None:
        """A releasing task may race a queued waiter by reacquiring immediately.

        ``asyncio.Lock`` briefly reports unlocked after granting its first
        waiter, but a new acquire still queues behind that waiter.  DPOR must
        model the new acquirer as blocked instead of scheduling a coroutine
        that the real lock has parked.
        """
        require_active("test_async_dpor_lock_handoff_reacquire")

        class Counter:
            def __init__(self) -> None:
                self.value = 0
                self.lock = asyncio.Lock()

        async def increment_twice(counter: Counter) -> None:
            for _ in range(2):
                async with counter.lock:
                    value = counter.value
                    await asyncio.sleep(0)
                    counter.value = value + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment_twice, increment_twice],
                strategy="dpor",
                invariant=lambda counter: counter.value == 4,
                preemption_bound=None,
                max_executions=100,
                deadlock_timeout=0.1,
                reproduce_on_failure=0,
                detect_io=False,
            )
        )

        assert result.property_holds, result.explanation

    def test_finds_queue_producer_ordering_race(self) -> None:
        """Queue contents are shared state, so producer order must be explored.

        ``asyncio.Queue`` keeps its items in a stdlib deque that opcode
        tracing never sees, so unless the wrapper reports the access DPOR
        treats two producers' ``put`` calls as commuting and certifies a pass.
        """
        require_active("test_async_dpor_queue_producer_order")

        class State:
            def __init__(self) -> None:
                self.queue: asyncio.Queue[int] = asyncio.Queue()
                self.drained: list[int] = []

        def make_producer(item: int):  # noqa: ANN202
            async def producer(state: State) -> None:
                await state.queue.put(item)

            return producer

        async def consumer(state: State) -> None:
            state.drained.append(await state.queue.get())
            state.drained.append(await state.queue.get())

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_producer(0), make_producer(1), consumer],
                strategy="dpor",
                invariant=lambda s: s.drained == [0, 1],
                preemption_bound=None,
                max_executions=200,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "DPOR should explore the producer order that drains [1, 0]"

    @pytest.mark.parametrize(
        ("queue_name", "items", "initial_order"),
        [
            ("LifoQueue", (0, 1), [1, 0]),
            ("PriorityQueue", ((0, 0), (0, 1)), [0, 1]),
        ],
    )
    def test_finds_queue_variant_producer_ordering_race(
        self, queue_name: str, items: tuple[object, object], initial_order: list[int]
    ) -> None:
        """LIFO and equal-priority queues need the same state instrumentation."""
        require_active(f"test_async_dpor_{queue_name.lower()}_producer_order")

        class PriorityItem:
            def __init__(self, priority: int, label: int) -> None:
                self.priority = priority
                self.label = label

            def __lt__(self, other: PriorityItem) -> bool:
                return self.priority < other.priority

        class State:
            def __init__(self) -> None:
                queue_type = getattr(asyncio, queue_name)
                self.queue: asyncio.Queue[object] = queue_type()
                self.drained: list[int] = []

        queue_items = tuple(PriorityItem(*item) if isinstance(item, tuple) else item for item in items)

        def make_producer(item: object):  # noqa: ANN202
            async def producer(state: State) -> None:
                await state.queue.put(item)

            return producer

        async def consumer(state: State) -> None:
            for _ in items:
                item = await state.queue.get()
                state.drained.append(item.label if isinstance(item, PriorityItem) else item)

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[*(make_producer(item) for item in queue_items), consumer],
                strategy="dpor",
                invariant=lambda state: state.drained == initial_order,
                preemption_bound=None,
                max_executions=200,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, f"DPOR should explore both {queue_name} producer orders"

    @pytest.mark.parametrize(("observation", "producer_first"), [("qsize", 1), ("empty", False), ("full", True)])
    def test_finds_queue_observation_race(self, observation: str, producer_first: object) -> None:
        """Queue size observations are reads of the same state that put mutates."""
        require_active(f"test_async_dpor_queue_{observation}_race")

        class State:
            def __init__(self) -> None:
                self.queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
                self.observed: object = None

        async def producer(state: State) -> None:
            await state.queue.put(1)

        async def observer(state: State) -> None:
            self_method = getattr(state.queue, observation)
            state.observed = self_method()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[producer, observer],
                strategy="dpor",
                invariant=lambda state: state.observed == producer_first,
                preemption_bound=None,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, f"DPOR should explore {observation} before and after put"

    @pytest.mark.skipif(not hasattr(asyncio, "QueueShutDown"), reason="asyncio.Queue.shutdown requires Python 3.13+")
    def test_queue_shutdown_wakes_engine_parked_getter(self) -> None:
        """shutdown() must wake the wrapper's waiter, not only stdlib waiters."""
        require_active("test_async_dpor_queue_shutdown")

        class State:
            def __init__(self) -> None:
                self.queue: asyncio.Queue[int] = asyncio.Queue()
                self.saw_shutdown = False

        async def getter(state: State) -> None:
            try:
                await state.queue.get()
            except asyncio.QueueShutDown:
                state.saw_shutdown = True

        async def shutdown(state: State) -> None:
            state.queue.shutdown()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[getter, shutdown],
                strategy="dpor",
                invariant=lambda state: state.saw_shutdown,
                preemption_bound=None,
                timeout_per_run=0.5,
                deadlock_timeout=0.1,
            )
        )

        assert result.property_holds is True, result.explanation

    def test_completed_and_timed_out_executions_are_inconclusive(self) -> None:
        """One completed execution cannot turn an incomplete search into proof.

        The setter-first order completes.  In the waiter-first order, the
        waiter snapshots ``ready=False`` and later parks forever on an
        unmanaged Future.  DPOR must not report ``property_holds=True`` merely
        because at least one of the two explored orders completed.
        """
        require_active("test_async_dpor_mixed_timeout")

        class State:
            def __init__(self) -> None:
                self.ready = False
                self.future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def setter(state: State) -> None:
            state.ready = True
            await asyncio.sleep(0)

        async def waiter(state: State) -> None:
            seen_ready = state.ready
            await asyncio.sleep(0)
            if seen_ready:
                state.future.set_result(None)
            await state.future

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[setter, waiter],
                invariant=lambda _state: True,
                strategy="dpor",
                max_executions=20,
                preemption_bound=None,
                timeout_per_run=0.05,
                deadlock_timeout=0.02,
                reproduce_on_failure=0,
                detect_io=False,
                stop_on_first=False,
            )
        )

        assert result.num_explored >= 2
        assert not result.property_holds
        assert result.explanation is not None
        assert "inconclusive" in result.explanation.lower()

    def test_finds_read_modify_write_torn_read(self) -> None:
        """DPOR must explore the read-modify-write interleaving where a checker
        observes a value change across its own await (torn read).

        Reachable interleaving: checker reads value=0, inc runs read+write (→1),
        checker resumes, observes value != tmp, sets torn=True.
        """
        require_active("test_async_dpor_rmw_torn_read")

        class C:
            def __init__(self) -> None:
                self.value = 0
                self.torn = False

        async def inc(c: C) -> None:
            tmp = c.value
            await asyncio.sleep(0)
            c.value = tmp + 1

        async def checker(c: C) -> None:
            tmp = c.value
            await asyncio.sleep(0)
            if c.value != tmp:
                c.torn = True
            c.value = tmp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=C,
                workers=[inc, checker],
                invariant=lambda c: not c.torn,
                max_executions=1000,
                preemption_bound=None,
                stop_on_first=False,
                strategy="dpor",
                detect_io=False,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, (
            f"reachable torn read must be found, got property_holds=True "
            f"in {result.num_explored} executions (under-exploration bug)"
        )

    def test_tracks_stale_read_across_await(self) -> None:
        """DPOR should catch a stale read carried across an await boundary."""
        require_active("test_async_dpor_stale_read_across_await")

        class Counter:
            def __init__(self) -> None:
                self.value = 0
                self.audit_log: list[int] = []

        async def increment(counter: Counter) -> None:
            initial = counter.value
            await asyncio.sleep(0)
            counter.audit_log.append(initial)
            counter.value += initial + 1
            counter.audit_log.append(counter.value)
            await asyncio.sleep(0)

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the stale-read lost update"
        assert result.num_explored >= 1

    def test_independent_objects_collapse_to_one_execution(self) -> None:
        """Independent await-delimited blocks should not create extra executions."""
        require_active("test_async_dpor_independent_objects")

        class State:
            def __init__(self) -> None:
                self.a = 0
                self.b = 0

        async def set_a(state: State) -> None:
            await asyncio.sleep(0)
            state.a = 1
            await asyncio.sleep(0)

        async def set_b(state: State) -> None:
            await asyncio.sleep(0)
            state.b = 1
            await asyncio.sleep(0)

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[set_a, set_b],
                strategy="dpor",
                invariant=lambda s: s.a == 1 and s.b == 1,
                preemption_bound=None,
                max_executions=100,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        assert result.num_explored == 1

    def test_detects_global_race(self) -> None:
        """Async DPOR should trace LOAD_GLOBAL/STORE_GLOBAL conflicts."""
        require_active("test_async_dpor_global_race")

        class State:
            def __init__(self) -> None:
                global _async_global_counter
                _async_global_counter = 0

        async def increment(_state: State) -> None:
            global _async_global_counter
            tmp = _async_global_counter
            await asyncio.sleep(0)
            _async_global_counter = tmp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda _s: _async_global_counter == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the global lost update"

    def test_detects_augmented_global_race(self) -> None:
        """Async DPOR should trace global += via LOAD_GLOBAL/STORE_GLOBAL."""
        require_active("test_async_dpor_global_augassign")

        class State:
            def __init__(self) -> None:
                global _async_global_augmented
                _async_global_augmented = 0

        async def increment(_state: State) -> None:
            global _async_global_augmented
            tmp = _async_global_augmented
            await asyncio.sleep(0)
            _async_global_augmented = tmp
            _async_global_augmented += 1

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda _s: _async_global_augmented == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the global += lost update"

    def test_detects_closure_cell_race(self) -> None:
        """Async DPOR should trace LOAD_DEREF/STORE_DEREF conflicts."""
        require_active("test_async_dpor_closure_race")

        shared = 0

        class State:
            def __init__(self) -> None:
                nonlocal shared
                shared = 0
                self.value = -1

        async def increment(state: State) -> None:
            nonlocal shared
            tmp = shared
            await asyncio.sleep(0)
            shared = tmp + 1
            state.value = shared

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda _s: shared == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the closure-cell lost update"

    def test_traces_sync_helper_inside_coroutine(self) -> None:
        """Tracing should continue through synchronous helper frames."""
        require_active("test_async_dpor_sync_helper")

        class State:
            def __init__(self) -> None:
                self.value = 0

        def helper(state: State, value: int) -> None:
            state.value = value

        async def update(state: State) -> None:
            snapshot = state.value
            await asyncio.sleep(0)
            helper(state, snapshot + 1)

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[update, update],
                strategy="dpor",
                invariant=lambda s: s.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should trace helper-frame writes after an await"

    def test_detects_container_method_conflict(self) -> None:
        """Passthrough builtin reads should conflict with C-level container writes."""
        require_active("test_async_dpor_container_method_conflict")

        class State:
            def __init__(self) -> None:
                self.items: list[int] = []
                self.observed = -1

        async def append_item(state: State) -> None:
            await asyncio.sleep(0)
            state.items.append(1)
            await asyncio.sleep(0)

        async def measure_length(state: State) -> None:
            await asyncio.sleep(0)
            state.observed = len(state.items)
            await asyncio.sleep(0)

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[append_item, measure_length],
                strategy="dpor",
                invariant=lambda s: s.observed in (0, 1),
                preemption_bound=None,
                max_executions=100,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        assert result.num_explored >= 2

    def test_disjoint_dict_keys_collapse_to_one_execution(self) -> None:
        """Disjoint subscript writes should not create extra executions."""
        require_active("test_async_dpor_disjoint_dict_keys")

        class State:
            def __init__(self) -> None:
                self.mapping = {"a": 0, "b": 0}

        async def set_a(state: State) -> None:
            await asyncio.sleep(0)
            state.mapping["a"] = 1
            await asyncio.sleep(0)

        async def set_b(state: State) -> None:
            await asyncio.sleep(0)
            state.mapping["b"] = 1
            await asyncio.sleep(0)

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[set_a, set_b],
                strategy="dpor",
                invariant=lambda s: s.mapping == {"a": 1, "b": 1},
                preemption_bound=None,
                max_executions=100,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        assert result.num_explored == 1

    def test_three_tasks(self) -> None:
        """DPOR should handle three concurrent tasks."""
        require_active("test_async_dpor_three_tasks")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 3,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Should find lost update with 3 tasks"

    def test_multiple_await_points(self) -> None:
        """DPOR should explore interleavings with multiple await points per task."""
        require_active("test_async_dpor_multiple_awaits")

        class State:
            def __init__(self) -> None:
                self.log: list[str] = []

        async def task_a(state: State) -> None:
            state.log.append("a1")
            await asyncio.sleep(0)
            state.log.append("a2")
            await asyncio.sleep(0)
            state.log.append("a3")

        async def task_b(state: State) -> None:
            state.log.append("b1")
            await asyncio.sleep(0)
            state.log.append("b2")

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[task_a, task_b],
                strategy="dpor",
                invariant=lambda s: True,  # always passes
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        # DPOR should have explored multiple distinct interleavings
        assert result.num_explored >= 1

    def test_stop_on_first(self) -> None:
        """stop_on_first=True should stop after finding the first violation."""
        require_active("test_async_dpor_stop_on_first")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                stop_on_first=True,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert len(result.failures) == 1


class TestAsyncDporDeadlock:
    """Deadlocks should surface as property_holds=False, not silently time out.

    Mirrors the sync TestDeadlockAsInvariantViolation tests in test_dpor.py,
    adapted for async coroutines with asyncio.Lock as the locking primitive.
    """

    def test_two_coroutine_row_lock_deadlock(self) -> None:
        """Classic lock-order inversion: C1 locks row1→row2, C2 locks row2→row1.

        C1 acquires row1, C2 acquires row2, then C1 tries row2 (blocked)
        and C2 tries row1 (blocked).  DPOR should find the deadlocking
        interleaving and report it — not actually deadlock.
        """
        require_active("test_async_dpor_two_coroutine_deadlock")

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row2.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,  # no data invariant — deadlock itself is the bug
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_three_coroutine_directed_cycle_deadlock(self) -> None:
        """Three-coroutine deadlock: C1→row1→row2, C2→row2→row3, C3→row3→row1.

        Forms the directed cycle C1→C2→C3→C1 when each holds its first
        lock and waits for its second.
        """
        require_active("test_async_dpor_three_coroutine_deadlock")

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()
                self.row3 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await state.row3.acquire()
                state.row3.release()
            finally:
                state.row2.release()

        async def coroutine3(state: State) -> None:
            await state.row3.acquire()
            try:
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row3.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2, coroutine3],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "3-way deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_combined_asyncio_lock_and_sql_deadlock(self) -> None:
        """Mixed resource deadlock: one coroutine holds a real SQL row lock
        and waits for an asyncio.Lock, the other holds the asyncio.Lock and
        waits for the same SQL row lock.

        This cross-resource deadlock is invisible to both the DB backend
        (which only sees row locks) and asyncio (which only sees asyncio.Lock).
        Only the DPOR scheduler can detect it because it tracks both resource
        types in a unified WaitForGraph.

        Uses aiosqlite so no external database is needed.
        """
        require_active("test_async_dpor_combined_lock_deadlock")
        import os
        import tempfile

        import aiosqlite  # type: ignore[import-untyped]

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        class State:
            def __init__(self) -> None:
                self.app_lock = asyncio.Lock()
                self.db_path = db_path

        async def _setup_db() -> None:
            async with aiosqlite.connect(db_path) as db:
                await db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)")
                await db.execute("INSERT OR REPLACE INTO t (id, v) VALUES (1, 0)")
                await db.commit()

        async def coroutine1(state: State) -> None:
            # Acquire a SQL row lock (UPDATE inside tx), then try the app lock
            async with aiosqlite.connect(state.db_path) as db:
                await db.execute("BEGIN")
                await db.execute("UPDATE t SET v = 1 WHERE id = 1")
                # Row lock on sql:t:(('id', 1)) is now held by this task
                # Try to acquire the asyncio.Lock — may block if C2 holds it
                await state.app_lock.acquire()
                state.app_lock.release()
                await db.execute("COMMIT")

        async def coroutine2(state: State) -> None:
            # Acquire the app lock first, then try to get the SQL row lock
            await state.app_lock.acquire()
            try:
                async with aiosqlite.connect(state.db_path) as db:
                    await db.execute("BEGIN")
                    # This tries to acquire the same row lock held by C1
                    await db.execute("UPDATE t SET v = 2 WHERE id = 1")
                    await db.execute("COMMIT")
            finally:
                state.app_lock.release()

        async def run() -> object:
            await _setup_db()
            return await frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,
                detect_sql=True,
                deadlock_timeout=2.0,
                timeout_per_run=5.0,
            )

        try:
            result = asyncio.run(run())
        finally:
            os.unlink(db_path)

        assert not result.property_holds, "Cross-resource deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_partial_deadlock_with_completing_coroutine(self) -> None:
        """Three coroutines: two deadlock while a third completes normally.

        C1 and C2 have lock-order inversion; C3 does independent work.
        The partial deadlock should still be detected even though C3 finishes.
        """
        require_active("test_async_dpor_partial_deadlock")

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.c3_done = False

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_b.acquire()
            try:
                await state.lock_a.acquire()
                state.lock_a.release()
            finally:
                state.lock_b.release()

        async def coroutine3(state: State) -> None:
            state.c3_done = True

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2, coroutine3],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Partial deadlock should still be detected"

    def test_no_deadlock_consistent_lock_order(self) -> None:
        """Consistent lock ordering should not be reported as a deadlock.

        Both coroutines acquire locks in the same order (lock_a then lock_b),
        so no cycle is possible.
        """
        require_active("test_async_dpor_no_deadlock")

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert result.property_holds, "Consistent lock order should not be reported as deadlock"

    def test_self_deadlock_non_reentrant_lock(self) -> None:
        """Single coroutine tries to acquire the same non-reentrant asyncio.Lock twice.

        asyncio.Lock is not reentrant, so acquiring it while already held
        by the same coroutine is an instant deadlock.
        """
        require_active("test_async_dpor_self_deadlock")

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.lock.acquire()
            # Re-acquire the same non-reentrant lock — instant self-deadlock
            await state.lock.acquire()
            state.lock.release()
            state.lock.release()

        async def coroutine2(state: State) -> None:
            pass

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,
                max_executions=20,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Self-deadlock should set property_holds=False"

    def test_asymmetric_await_points_before_deadlock(self) -> None:
        """Coroutines have different numbers of await points before the
        deadlocking acquire.  Verifies DPOR explores enough interleavings
        to reach the state where both hold one lock.
        """
        require_active("test_async_dpor_asymmetric_deadlock")

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.log: list[str] = []

        async def coroutine1(state: State) -> None:
            # Several await points of work before acquiring locks
            state.log.append("c1_step1")
            await asyncio.sleep(0)
            state.log.append("c1_step2")
            await asyncio.sleep(0)
            state.log.append("c1_step3")
            await asyncio.sleep(0)
            # Now do the lock-order-inversion pattern
            await state.lock_a.acquire()
            try:
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            # Only one await point before locking
            state.log.append("c2_step1")
            await asyncio.sleep(0)
            await state.lock_b.acquire()
            try:
                await state.lock_a.acquire()
                state.lock_a.release()
            finally:
                state.lock_b.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,
                max_executions=50,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Asymmetric deadlock should be found"

    def test_data_dependent_lock_order_deadlock(self) -> None:
        """Lock acquisition order depends on runtime state.

        C1 always acquires lock_a then lock_b.  C2 reads a shared flag
        (set by C1) that determines whether it acquires lock_b then lock_a
        (deadlock) or lock_a then lock_b (safe).  Only the interleaving
        where C1 sets the flag before C2 reads it triggers the deadlock.
        """
        require_active("test_async_dpor_data_dependent_deadlock")

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.reverse_order = False

        async def coroutine1(state: State) -> None:
            state.reverse_order = True
            await asyncio.sleep(0)
            await state.lock_a.acquire()
            try:
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await asyncio.sleep(0)
            if state.reverse_order:
                # Opposite order from C1 → deadlock possible
                await state.lock_b.acquire()
                try:
                    await state.lock_a.acquire()
                    state.lock_a.release()
                finally:
                    state.lock_b.release()
            else:
                # Same order as C1 → safe
                await state.lock_a.acquire()
                try:
                    await state.lock_b.acquire()
                    state.lock_b.release()
                finally:
                    state.lock_a.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[coroutine1, coroutine2],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Data-dependent deadlock should be found"

    def test_dining_philosophers_three(self) -> None:
        """Three dining philosophers: each acquires fork[i] then fork[(i+1)%3].

        Classic deadlock when all philosophers pick up their left fork
        simultaneously.  Uses 3 philosophers to keep the DPOR state space
        manageable (4 philosophers with preemption_bound=2 doesn't explore
        enough interleavings).
        """
        require_active("test_async_dpor_dining_philosophers")

        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [asyncio.Lock() for _ in range(num_philosophers)]

        def make_philosopher(i: int):  # noqa: ANN202
            async def philosopher(state: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                await state.forks[left].acquire()
                try:
                    await state.forks[right].acquire()
                    state.forks[right].release()
                finally:
                    state.forks[left].release()

            return philosopher

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_philosopher(i) for i in range(num_philosophers)],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Dining philosophers deadlock should be found"


class TestAsyncDporCleanup:
    """Tests for resource cleanup when tasks finish without releasing locks."""

    def test_row_locks_released_on_task_finish(self) -> None:
        """Row locks should be released from the WaitForGraph when a task finishes.

        Bug: _cleanup_task_context doesn't call release_row_locks(task_id),
        so stale row lock entries remain in _active_row_locks and the
        WaitForGraph after a task completes without COMMIT.

        Scenario: Task 0 acquires a row lock and finishes without releasing.
        After cleanup, the scheduler's _active_row_locks should NOT still
        show task 0 as the holder.
        """
        require_active("test_async_dpor_row_lock_cleanup")
        from frontrun.async_dpor import AsyncDporScheduler, _patch_asyncio_lock, _unpatch_asyncio_lock

        async def run() -> None:
            from frontrun._dpor import PyDporEngine  # type: ignore[reportAttributeAccessIssue]

            _patch_asyncio_lock()
            try:
                engine = PyDporEngine(num_threads=2, preemption_bound=2)
                execution = engine.begin_execution()
                scheduler = AsyncDporScheduler(engine, execution, 2, deadlock_timeout=2.0)

                # Simulate task 0 acquiring row locks
                scheduler.acquire_row_locks(0, ["sql:t:(('id', 1))"])
                assert "sql:t:(('id', 1))" in scheduler._active_row_locks
                assert scheduler._active_row_locks["sql:t:(('id', 1))"] == 0

                # Simulate task 0 finishing (as _run's finally block does)
                scheduler._cleanup_task_context(0)

                # After cleanup, row locks should be released
                assert "sql:t:(('id', 1))" not in scheduler._active_row_locks, (
                    "Row lock should be released when task finishes, "
                    f"but _active_row_locks still contains: {scheduler._active_row_locks}"
                )
                assert 0 not in scheduler._task_row_locks, (
                    "Task's row lock set should be cleared on finish, "
                    f"but _task_row_locks still contains task 0: {scheduler._task_row_locks}"
                )
            finally:
                _unpatch_asyncio_lock()

        asyncio.run(run())

    def test_asyncio_lock_released_on_task_exception(self) -> None:
        """asyncio.Lock should be released when a task raises an exception.

        Bug: When a task finishes while holding a _CooperativeAsyncLock
        (e.g., exception without release()), the WaitForGraph holding edge
        remains AND the underlying real asyncio.Lock stays locked, blocking
        any other task that tries to acquire it.

        Scenario: Task 0 acquires a lock and crashes. Task 1 should be able
        to acquire the same lock without timing out.
        """
        require_active("test_async_dpor_lock_cleanup_on_exception")

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                self.task1_got_lock = False

        async def task0_crashes(state: State) -> None:
            await state.lock.acquire()
            raise RuntimeError("intentional crash while holding lock")

        async def task1_acquires(state: State) -> None:
            # Task 1 should be able to acquire the lock after task 0 crashes
            await state.lock.acquire()
            state.task1_got_lock = True
            state.lock.release()

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[task0_crashes, task1_acquires],
                strategy="dpor",
                invariant=lambda s: s.task1_got_lock,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        # Task 0 crashes, so "does not crash" meta-invariant should trigger.
        assert not result.property_holds, (
            "Task crash should be treated as a property violation, "
            f"but got: property_holds={result.property_holds}, explanation={result.explanation}"
        )
        assert result.explanation is not None, "Explanation should be set for task crash"
        assert "RuntimeError" in result.explanation or "crash" in result.explanation, (
            f"Explanation should mention the crash, but got: {result.explanation}"
        )


class TestAsyncDporExplanation:
    """Tests for human-readable explanations of invariant violations."""

    def test_explanation_set_for_invariant_violation(self) -> None:
        """result.explanation should be non-None when an invariant violation is found.

        Bug: async explore() (DPOR) sets result.explanation for deadlocks but
        NOT for invariant violations. The explanation field stays None.
        """
        require_active("test_async_dpor_invariant_explanation")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert result.explanation is not None, (
            "result.explanation should be set for invariant violations, but it is None"
        )

    def test_explanation_contains_schedule_info(self) -> None:
        """Explanation should contain information about the interleaving schedule.

        For async DPOR, the explanation should describe which tasks ran
        at which points, making it possible to understand the race condition.
        """
        require_active("test_async_dpor_explanation_content")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert result.explanation is not None
        # Explanation should mention tasks/schedule
        explanation_lower = result.explanation.lower()
        assert "task" in explanation_lower or "schedule" in explanation_lower or "interleav" in explanation_lower, (
            f"Explanation should describe the interleaving, got: {result.explanation}"
        )


class TestAsyncDporReproduceOnFailure:
    """Tests for reproduce_on_failure parameter in async DPOR."""

    def test_reproduce_on_failure_default(self) -> None:
        """By default, reproduce_on_failure=10 replays the counterexample."""
        require_active("test_async_dpor_reproduce_default")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 10, (
            f"Expected 10 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_reproduce_on_failure_zero(self) -> None:
        """reproduce_on_failure=0 skips replay."""
        require_active("test_async_dpor_reproduce_zero")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
                reproduce_on_failure=0,
            )
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 0
        assert result.reproduction_successes == 0

    def test_reproduce_on_failure_custom(self) -> None:
        """reproduce_on_failure=3 replays exactly 3 times."""
        require_active("test_async_dpor_reproduce_custom")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
                reproduce_on_failure=3,
            )
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 3, (
            f"Expected 3 reproduction attempts, got {result.reproduction_attempts}"
        )
        assert result.reproduction_successes == 3, (
            f"Expected 3/3 reproductions, got {result.reproduction_successes}/{result.reproduction_attempts}"
        )


class TestAsyncDporTotalTimeout:
    """Tests for total_timeout parameter in async DPOR."""

    def test_total_timeout_bounds_exploration(self) -> None:
        """total_timeout should stop exploration even if more interleavings remain."""
        require_active("test_async_dpor_total_timeout")
        import time

        class State:
            def __init__(self) -> None:
                self.a = 0
                self.b = 0
                self.c = 0

        async def task_a(s: State) -> None:
            s.a += 1
            await asyncio.sleep(0)
            s.b += 1
            await asyncio.sleep(0)
            s.c += 1

        async def task_b(s: State) -> None:
            s.c += 1
            await asyncio.sleep(0)
            s.b += 1
            await asyncio.sleep(0)
            s.a += 1

        start = time.monotonic()
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[task_a, task_b],
                strategy="dpor",
                invariant=lambda s: True,  # always passes — we want to test timeout
                deadlock_timeout=5.0,
                total_timeout=0.5,
            )
        )
        elapsed = time.monotonic() - start

        assert result.property_holds
        assert elapsed < 3.0, f"total_timeout=0.5 but took {elapsed:.1f}s"


class TestAsyncDporWarnNondeterministicSQL:
    """Tests for warn_nondeterministic_sql parameter on async explore(strategy="dpor")."""

    def test_parameter_accepted(self) -> None:
        """async explore(strategy="dpor") should accept warn_nondeterministic_sql parameter."""
        require_active("test_async_dpor_warn_nondeterministic_sql")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            counter.value += 1
            await asyncio.sleep(0)

        # Should not raise — parameter should be accepted
        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
                warn_nondeterministic_sql=False,
            )
        )
        assert result.property_holds

    def test_raises_on_uncaptured_inserts(self) -> None:
        """When warn_nondeterministic_sql=True (default), should raise NondeterministicSQLError."""
        require_active("test_async_dpor_warn_nondeterministic_sql_raises")
        from frontrun._sql_insert_tracker import clear_insert_tracker, record_insert
        from frontrun.common import NondeterministicSQLError

        class State:
            def __init__(self) -> None:
                self.value = 0

        async def task_with_uncaptured_insert(state: State) -> None:
            # Simulate an INSERT with uncaptured lastrowid (concrete_id=None)
            record_insert("test_table", None)
            state.value += 1
            await asyncio.sleep(0)

        with pytest.raises(NondeterministicSQLError, match="test_table"):
            asyncio.run(
                frontrun.explore(
                    setup=State,
                    workers=[task_with_uncaptured_insert],
                    strategy="dpor",
                    invariant=lambda s: True,
                    deadlock_timeout=5.0,
                    detect_sql=True,
                    warn_nondeterministic_sql=True,
                )
            )

        # Clean up
        clear_insert_tracker()

    def test_suppressed_when_false(self) -> None:
        """When warn_nondeterministic_sql=False, should NOT raise on uncaptured INSERTs."""
        require_active("test_async_dpor_warn_nondeterministic_sql_suppressed")
        from frontrun._sql_insert_tracker import clear_insert_tracker, record_insert

        class State:
            def __init__(self) -> None:
                self.value = 0

        async def task_with_uncaptured_insert(state: State) -> None:
            record_insert("test_table", None)
            state.value += 1
            await asyncio.sleep(0)

        # Should NOT raise
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[task_with_uncaptured_insert],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=5.0,
                detect_sql=True,
                warn_nondeterministic_sql=False,
            )
        )
        assert result.property_holds

        # Clean up
        clear_insert_tracker()

    def test_clears_insert_tracker_between_executions(self) -> None:
        """Insert tracker should be cleared between DPOR executions."""
        require_active("test_async_dpor_clears_insert_tracker")
        from frontrun._sql_insert_tracker import get_records

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        records_during_run: list[int] = []

        async def increment(counter: Counter) -> None:
            # Record how many insert records exist at the start of this execution
            records_during_run.append(len(get_records()))
            temp = counter.value
            await asyncio.sleep(0)
            counter.value = temp + 1

        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
                warn_nondeterministic_sql=False,
                max_executions=5,
            )
        )

        # All recorded counts should be 0 (tracker cleared between executions)
        assert all(c == 0 for c in records_during_run), (
            f"Insert tracker was not cleared between executions: {records_during_run}"
        )


class TestAsyncDporLockTimeout:
    """Tests for lock_timeout parameter on async explore(strategy="dpor")."""

    def test_parameter_accepted(self) -> None:
        """async explore(strategy="dpor") should accept lock_timeout parameter."""
        require_active("test_async_dpor_lock_timeout")

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            counter.value += 1
            await asyncio.sleep(0)

        # Should not raise — parameter should be accepted
        result = asyncio.run(
            frontrun.explore(
                setup=Counter,
                workers=[increment, increment],
                strategy="dpor",
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
                lock_timeout=2000,
            )
        )
        assert result.property_holds

    def test_sets_global_lock_timeout(self) -> None:
        """lock_timeout should be set during exploration and restored after."""
        require_active("test_async_dpor_lock_timeout_global")
        from frontrun._sql_cursor import get_lock_timeout

        class State:
            def __init__(self) -> None:
                self.value = 0

        lock_timeout_during: list[int | None] = []

        async def check_lock_timeout(state: State) -> None:
            lock_timeout_during.append(get_lock_timeout())
            state.value += 1
            await asyncio.sleep(0)

        # Verify lock_timeout is None before
        assert get_lock_timeout() is None

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[check_lock_timeout],
                strategy="dpor",
                invariant=lambda s: True,
                deadlock_timeout=5.0,
                lock_timeout=1500,
            )
        )
        assert result.property_holds

        # During exploration, lock_timeout should have been 1500
        assert all(lt == 1500 for lt in lock_timeout_during), (
            f"lock_timeout was not set during exploration: {lock_timeout_during}"
        )

        # After exploration, lock_timeout should be restored to None
        assert get_lock_timeout() is None, "lock_timeout was not restored after exploration"

    def test_restores_previous_lock_timeout(self) -> None:
        """lock_timeout should restore the previous value, not just None."""
        require_active("test_async_dpor_lock_timeout_restore")
        from frontrun._sql_cursor import get_lock_timeout, set_lock_timeout

        class State:
            def __init__(self) -> None:
                self.value = 0

        async def noop_task(state: State) -> None:
            state.value += 1
            await asyncio.sleep(0)

        # Set a pre-existing lock_timeout
        set_lock_timeout(999)
        try:
            result = asyncio.run(
                frontrun.explore(
                    setup=State,
                    workers=[noop_task],
                    strategy="dpor",
                    invariant=lambda s: True,
                    deadlock_timeout=5.0,
                    lock_timeout=2000,
                )
            )
            assert result.property_holds
            assert get_lock_timeout() == 999, "Previous lock_timeout was not restored"
        finally:
            set_lock_timeout(None)


class TestAsyncDporPreRegister:
    """Async DPOR must call stable_ids.pre_register(state) after setup().

    The sync path in _dpor_runtime/explore.py calls pre_register so that
    objects created in setup() get deterministic IDs regardless of the
    schedule.  The async path at async_dpor.py line 1154 is missing this
    call, causing object IDs to vary with first-touch order across
    executions.
    """

    def test_pre_register_called_for_setup_state(self) -> None:
        """Verify that async DPOR pre-registers setup state objects."""
        require_active("test_async_dpor_pre_register")
        from unittest.mock import patch as mock_patch

        from frontrun._opcode_observer import StableObjectIds

        pre_register_calls: list[object] = []
        original_pre_register = StableObjectIds.pre_register

        def tracking_pre_register(self_ids: StableObjectIds, root: object) -> None:
            pre_register_calls.append(root)
            return original_pre_register(self_ids, root)

        class State:
            def __init__(self) -> None:
                self.value = 0

        async def noop(state: State) -> None:
            state.value += 1
            await asyncio.sleep(0)

        with mock_patch.object(StableObjectIds, "pre_register", tracking_pre_register):
            asyncio.run(
                frontrun.explore(
                    setup=State,
                    workers=[noop],
                    strategy="dpor",
                    invariant=lambda s: True,
                    deadlock_timeout=5.0,
                )
            )

        assert len(pre_register_calls) >= 1, (
            "stable_ids.pre_register(state) must be called at least once (once per execution) but was never called"
        )
