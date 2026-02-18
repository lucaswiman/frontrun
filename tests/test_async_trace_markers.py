"""Tests for the interlace async_trace_markers module."""

import asyncio

import pytest

from interlace.async_trace_markers import AsyncTraceExecutor, async_interlace
from interlace.common import Schedule, Step


class BankAccount:
    """A simple bank account class with a race condition vulnerability."""

    def __init__(self, balance=0):
        self.balance = balance

    async def get_balance(self):
        return self.balance

    async def set_balance(self, new_balance):
        self.balance = new_balance

    async def transfer(self, amount):
        # interlace: read_balance
        current = await self.get_balance()
        # interlace: write_balance
        await self.set_balance(current + amount)


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
        # interlace: step1
        await asyncio.sleep(0)
        results.append("step1")
        # interlace: step2
        await asyncio.sleep(0)
        results.append("step2")
        # interlace: step3
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
        # interlace: marker_a
        await asyncio.sleep(0)
        results.append("t1_a")
        # interlace: marker_b
        await asyncio.sleep(0)
        results.append("t1_b")

    async def worker2():
        # interlace: marker_a
        await asyncio.sleep(0)
        results.append("t2_a")
        # interlace: marker_b
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
    """The async_interlace() convenience function."""
    results = []

    async def worker1():
        # interlace: mark
        await asyncio.sleep(0)
        results.append("t1")

    async def worker2():
        # interlace: mark
        await asyncio.sleep(0)
        results.append("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)

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
            # interlace: read_counter
            temp = await self.get_value()
            # interlace: write_counter
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
            # interlace: read_value
            current = await self.get_value()
            # interlace: write_value
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
        # interlace: marker1
        await asyncio.sleep(0)
        await asyncio.sleep(10)
        # interlace: marker2
        await asyncio.sleep(0)

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)
        # interlace: marker2
        await asyncio.sleep(0)

    with pytest.raises(TimeoutError):
        async_interlace(
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
        # interlace: marker1
        await asyncio.sleep(0)
        raise ValueError("Intentional error in task1")

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)

    with pytest.raises(ValueError, match="Intentional error in task1"):
        async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)


def test_task_errors_tracked():
    """task_errors is properly populated when tasks raise exceptions."""
    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
        ]
    )

    async def worker1():
        # interlace: marker1
        await asyncio.sleep(0)
        raise ValueError("Error in task1")

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)

    executor = AsyncTraceExecutor(schedule)

    with pytest.raises(ValueError, match="Error in task1"):
        executor.run({"t1": worker1, "t2": worker2})

    assert "t1" in executor.task_errors
    assert isinstance(executor.task_errors["t1"], ValueError)
