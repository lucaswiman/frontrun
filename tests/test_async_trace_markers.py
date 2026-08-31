"""Tests for the frontrun async_trace_markers module."""

import asyncio
import time

import pytest

from frontrun import TraceExecutor as UnifiedTraceExecutor
from frontrun.async_trace_markers import AsyncTraceExecutor, async_frontrun
from frontrun.common import Schedule, Step


class BankAccount:
    """A simple bank account class with a race condition vulnerability."""

    def __init__(self, balance=0):
        self.balance = balance

    async def get_balance(self):
        return self.balance

    async def set_balance(self, new_balance):
        self.balance = new_balance

    async def transfer(self, amount):
        # frontrun: read_balance
        current = await self.get_balance()
        # frontrun: write_balance
        await self.set_balance(current + amount)


def test_empty_task_mapping_is_rejected() -> None:
    executor = AsyncTraceExecutor(Schedule([Step("task", "marker")]))

    with pytest.raises(ValueError, match="empty"):
        executor.run({})


def test_race_condition_buggy_schedule():
    """Both tasks read before either writes, causing a lost update."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("task1", "read_balance"),
            Step("task2", "read_balance"),
            Step("task1", "write_balance"),
            Step("task2", "write_balance"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run(
        {
            "task1": lambda: account.transfer(50),
            "task2": lambda: account.transfer(50),
        }
    )

    assert account.balance == 150


def test_race_condition_correct_schedule():
    """Each task completes its transaction before the next starts."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("task1", "read_balance"),
            Step("task1", "write_balance"),
            Step("task2", "read_balance"),
            Step("task2", "write_balance"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run(
        {
            "task1": lambda: account.transfer(50),
            "task2": lambda: account.transfer(50),
        }
    )

    assert account.balance == 200


def test_multiple_markers_same_task():
    """A task hitting multiple markers in sequence."""
    results = []

    async def worker_with_markers():
        # frontrun: step1
        await asyncio.sleep(0)
        results.append("step1")
        # frontrun: step2
        await asyncio.sleep(0)
        results.append("step2")
        # frontrun: step3
        await asyncio.sleep(0)
        results.append("step3")

    schedule = Schedule(
        [
            Step("main", "step1"),
            Step("main", "step2"),
            Step("main", "step3"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run({"main": worker_with_markers})

    assert results == ["step1", "step2", "step3"]


def test_alternating_execution():
    """Alternating execution between two tasks."""
    results = []

    async def worker1():
        # frontrun: marker_a
        await asyncio.sleep(0)
        results.append("t1_a")
        # frontrun: marker_b
        await asyncio.sleep(0)
        results.append("t1_b")

    async def worker2():
        # frontrun: marker_a
        await asyncio.sleep(0)
        results.append("t2_a")
        # frontrun: marker_b
        await asyncio.sleep(0)
        results.append("t2_b")

    schedule = Schedule(
        [
            Step("task1", "marker_a"),
            Step("task2", "marker_a"),
            Step("task1", "marker_b"),
            Step("task2", "marker_b"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run({"task1": worker1, "task2": worker2})

    assert results == ["t1_a", "t2_a", "t1_b", "t2_b"]


def test_convenience_function():
    """The async_frontrun() convenience function."""
    results = []

    async def worker1():
        # frontrun: mark
        await asyncio.sleep(0)
        results.append("t1")

    async def worker2():
        # frontrun: mark
        await asyncio.sleep(0)
        results.append("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    async_frontrun(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)

    assert results == ["t1", "t2"]


def test_unified_trace_executor_supports_async_tasks():
    """The public TraceExecutor facade should dispatch async task mappings."""
    results = []

    async def worker1():
        # frontrun: mark
        await asyncio.sleep(0)
        results.append("t1")

    async def worker2():
        # frontrun: mark
        await asyncio.sleep(0)
        results.append("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    executor = UnifiedTraceExecutor(schedule)
    executor.run({"t1": worker1, "t2": worker2})
    executor.wait()

    assert results == ["t1", "t2"]


def test_complex_race_scenario():
    """Three tasks all read before any writes, causing maximum lost updates."""

    class SharedCounter:
        def __init__(self):
            self.value = 0

        async def get_value(self):
            return self.value

        async def set_value(self, value):
            self.value = value

        async def increment_racy(self):
            # frontrun: read_counter
            temp = await self.get_value()
            # frontrun: write_counter
            await self.set_value(temp + 1)

    counter = SharedCounter()

    schedule = Schedule(
        [
            Step("t1", "read_counter"),
            Step("t2", "read_counter"),
            Step("t3", "read_counter"),
            Step("t1", "write_counter"),
            Step("t2", "write_counter"),
            Step("t3", "write_counter"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run(
        {
            "t1": counter.increment_racy,
            "t2": counter.increment_racy,
            "t3": counter.increment_racy,
        }
    )

    assert counter.value == 1


def test_synchronous_function_bodies():
    """Markers work correctly even when awaited functions complete synchronously."""

    class Counter:
        def __init__(self):
            self.value = 0

        async def get_value(self):
            return self.value

        async def set_value(self, new_value):
            self.value = new_value

        async def increment(self):
            # frontrun: read_value
            current = await self.get_value()
            # frontrun: write_value
            await self.set_value(current + 1)

    counter = Counter()

    schedule = Schedule(
        [
            Step("t1", "read_value"),
            Step("t2", "read_value"),
            Step("t3", "read_value"),
            Step("t1", "write_value"),
            Step("t2", "write_value"),
            Step("t3", "write_value"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)
    executor.run(
        {
            "t1": counter.increment,
            "t2": counter.increment,
            "t3": counter.increment,
        }
    )

    assert counter.value == 1


@pytest.mark.intentionally_leaves_dangling_threads
def test_timeout():
    """Timeout raises TimeoutError when tasks don't complete."""
    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
            Step("t2", "marker2"),
            Step("t1", "marker2"),
        ]
    )

    async def worker1():
        # frontrun: marker1
        await asyncio.sleep(0)
        await asyncio.sleep(10)
        # frontrun: marker2
        await asyncio.sleep(0)

    async def worker2():
        # frontrun: marker1
        await asyncio.sleep(0)
        # frontrun: marker2
        await asyncio.sleep(0)

    with pytest.raises(TimeoutError):
        async_frontrun(
            schedule=schedule,
            tasks={"t1": worker1, "t2": worker2},
            timeout=0.5,
        )


def test_exception_propagation():
    """Exceptions in tasks are properly propagated."""
    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
        ]
    )

    async def worker1():
        # frontrun: marker1
        await asyncio.sleep(0)
        raise ValueError("Intentional error in task1")

    async def worker2():
        # frontrun: marker1
        await asyncio.sleep(0)

    with pytest.raises(ValueError, match="Intentional error in task1"):
        async_frontrun(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)


def test_task_errors_tracked():
    """task_errors is properly populated when tasks raise exceptions."""
    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
        ]
    )

    async def worker1():
        # frontrun: marker1
        await asyncio.sleep(0)
        raise ValueError("Error in task1")

    async def worker2():
        # frontrun: marker1
        await asyncio.sleep(0)

    executor = AsyncTraceExecutor(schedule)

    with pytest.raises(ValueError, match="Error in task1"):
        executor.run({"t1": worker1, "t2": worker2})

    assert "t1" in executor.task_errors
    assert isinstance(executor.task_errors["t1"], ValueError)


@pytest.mark.intentionally_leaves_dangling_threads
def test_run_uses_shared_timeout_deadline():
    """AsyncTraceExecutor.run() should use a shared deadline across all threads.

    With N hanging threads, the total wait should be ~timeout, not N*timeout.
    Regression test: previously each thread.join(timeout=timeout) was called
    independently, so 3 threads with timeout=0.5s would wait up to 1.5s total.
    The correct behavior (matching the sync _ThreadTraceExecutor.wait()) is to
    compute a single deadline and reduce remaining time for each subsequent join.
    """

    async def hanging_task() -> None:
        await asyncio.sleep(100)  # hangs far beyond timeout

    # Use a schedule with a single task that won't be reached (tasks never hit
    # any markers, so each thread simply blocks waiting for the execution lock).
    schedule = Schedule([Step("task1", "never_reached")])
    executor = AsyncTraceExecutor(schedule, deadlock_timeout=100.0)

    timeout = 0.5
    num_threads = 3

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.run(
            {"task1": hanging_task, "task2": hanging_task, "task3": hanging_task},
            timeout=timeout,
        )
    elapsed = time.monotonic() - start

    # With shared deadline: total elapsed ≈ timeout (0.5s)
    # With per-thread timeout (bug): total elapsed ≈ num_threads * timeout (1.5s)
    # We allow up to (N-1)*timeout extra slack to reliably catch the bug.
    assert elapsed < (num_threads - 1) * timeout, (
        f"Elapsed {elapsed:.2f}s exceeds {(num_threads - 1) * timeout:.1f}s: "
        f"run() appears to give each thread its own full timeout instead of "
        f"using a shared deadline (expected ~{timeout:.1f}s total)"
    )


class TestInlineMarkerNoDoubleTrigger:
    """Inline markers (code + comment on same line) must fire exactly once."""

    def test_inline_marker_fires_once(self):
        """An inline marker followed by a non-marker line must not double-trigger.

        When an inline marker (code + comment) is followed by executable code
        without a marker, the "check previous line" logic in the async trace
        function can re-trigger the same marker. This causes a schedule stall
        because the re-triggered marker name doesn't match the next expected step.
        """

        class State:
            def __init__(self):
                self.steps: list[str] = []

        async def task_a(state: State):
            state.steps.append("a1")  # frontrun: a_step1
            state.steps.append("a_between")  # no marker — would re-trigger a_step1
            state.steps.append("a2")  # frontrun: a_step2

        async def task_b(state: State):
            state.steps.append("b1")  # frontrun: b_step1

        schedule = Schedule(
            [
                Step("task_a", "a_step1"),
                Step("task_b", "b_step1"),
                Step("task_a", "a_step2"),
            ]
        )

        state = State()
        executor = AsyncTraceExecutor(schedule, deadlock_timeout=3.0)
        executor.run(
            {"task_a": lambda: task_a(state), "task_b": lambda: task_b(state)},
            timeout=5.0,
        )

        # If the previous-line check double-triggers, the coordinator stalls
        # for deadlock_timeout seconds. Verify it completed quickly.
        assert "a1" in state.steps and "b1" in state.steps and "a2" in state.steps, (
            f"All markers should fire without stall, got steps: {state.steps}"
        )
        assert executor.coordinator.error is None, (
            f"Coordinator should not have errors (double-trigger causes stall): {executor.coordinator.error}"
        )


class TestAsyncScheduleIncompleteness:
    def test_incomplete_schedule_reports_unreached_steps(self):
        """When an async schedule is only partially consumed, the error message
        should list the specific steps that were never reached — matching the
        sync _ThreadTraceExecutor.wait() behavior."""

        async def task_a():
            # frontrun: step_a1
            pass

        async def task_b():
            # frontrun: step_b1
            pass

        schedule = Schedule(
            [
                Step("task_a", "step_a1"),
                Step("task_b", "step_b1"),
                Step("task_a", "step_a_never"),  # never reached
            ]
        )

        executor = AsyncTraceExecutor(schedule, deadlock_timeout=2.0)
        with pytest.raises(TimeoutError, match="Schedule incomplete"):
            executor.run(
                {"task_a": task_a, "task_b": task_b},
                timeout=5.0,
            )
