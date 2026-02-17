"""
Tests for the interlace async_trace_markers module.

Demonstrates deterministic async task interleaving using comment-based markers.

These tests mirror the structure of test_trace_markers.py but adapted for async/await.
Each test uses asyncio.run() to execute the async test logic.
"""

import asyncio
import os
import sys

# Add parent directory to path so we can import interlace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interlace.async_trace_markers import AsyncTaskCoordinator, AsyncTraceExecutor, async_interlace
from interlace.common import Schedule, Step


class BankAccount:
    """A simple bank account class with a race condition vulnerability.

    This async version demonstrates how race conditions occur at await points.
    Uses # interlace: comments to mark synchronization points.
    """

    def __init__(self, balance=0):
        self.balance = balance

    async def transfer(self, amount):
        """Transfer money to this account (intentionally racy).

        This method has a race condition: it reads the balance, then writes
        a new balance without locking. The # interlace: comments mark where
        the race can occur.

        Marker pattern: the marker comment and ``await`` come BEFORE the
        operation they gate. The marker is detected when the trace function
        fires on the ``await`` line (via the previous-line check). After
        the coordinator approves this task, the operation runs in the next
        ``send()`` call, with no intervening yield to the event loop.

        Args:
            amount: The amount to transfer
        """
        # interlace: read_balance
        await asyncio.sleep(0)
        current = self.balance
        new_balance = current + amount
        # interlace: write_balance
        await asyncio.sleep(0)
        self.balance = new_balance


def test_race_condition_buggy_schedule():
    """Test that demonstrates the race condition bug with an unsafe schedule.

    Schedule: t1 reads, t2 reads, t1 writes, t2 writes
    This creates a lost update: both tasks read the same initial value,
    so the final balance is incorrect.
    """
    print("\n=== Test: Race Condition (Buggy Schedule) ===")

    async def run_test():
        account = BankAccount(balance=100)

        # Define the buggy schedule: both tasks read before either writes
        schedule = Schedule(
            [
                Step("task1", "read_balance"),
                Step("task2", "read_balance"),
                Step("task1", "write_balance"),
                Step("task2", "write_balance"),
            ]
        )

        executor = AsyncTraceExecutor(schedule)

        await executor.run(
            {
                "task1": lambda: account.transfer(50),
                "task2": lambda: account.transfer(50),
            }
        )

        print("Initial balance: 100")
        print("Task 1 transfer: +50")
        print("Task 2 transfer: +50")
        print("Expected (buggy): 150")
        print(f"Actual balance: {account.balance}")

        # With the buggy schedule, we expect a lost update
        assert account.balance == 150, f"Expected 150 (lost update), got {account.balance}"
        print("✓ Race condition successfully reproduced!")

    asyncio.run(run_test())


def test_race_condition_correct_schedule():
    """Test that demonstrates correct execution with a safe schedule.

    Schedule: t1 reads, t1 writes, t2 reads, t2 writes
    This ensures proper serialization: each task completes its transaction
    before the next one starts.
    """
    print("\n=== Test: Race Condition (Correct Schedule) ===")

    async def run_test():
        account = BankAccount(balance=100)

        # Define the correct schedule: each task completes before the next starts
        schedule = Schedule(
            [
                Step("task1", "read_balance"),
                Step("task1", "write_balance"),
                Step("task2", "read_balance"),
                Step("task2", "write_balance"),
            ]
        )

        executor = AsyncTraceExecutor(schedule)

        await executor.run(
            {
                "task1": lambda: account.transfer(50),
                "task2": lambda: account.transfer(50),
            }
        )

        print("Initial balance: 100")
        print("Task 1 transfer: +50")
        print("Task 2 transfer: +50")
        print("Expected (correct): 200")
        print(f"Actual balance: {account.balance}")

        # With the correct schedule, we expect the right result
        assert account.balance == 200, f"Expected 200, got {account.balance}"
        print("✓ Correct execution verified!")

    asyncio.run(run_test())


def test_multiple_markers_same_task():
    """Test a task hitting multiple markers in sequence."""
    print("\n=== Test: Multiple Markers Same Task ===")

    async def run_test():
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

        await executor.run(
            {
                "main": worker_with_markers,
            }
        )

        print(f"Results: {results}")
        assert results == ["step1", "step2", "step3"]
        print("✓ Multiple markers executed in order!")

    asyncio.run(run_test())


def test_alternating_execution():
    """Test alternating execution between two tasks."""
    print("\n=== Test: Alternating Execution ===")

    async def run_test():
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

        # Alternate between tasks at each marker
        schedule = Schedule(
            [
                Step("task1", "marker_a"),
                Step("task2", "marker_a"),
                Step("task1", "marker_b"),
                Step("task2", "marker_b"),
            ]
        )

        executor = AsyncTraceExecutor(schedule)

        await executor.run(
            {
                "task1": worker1,
                "task2": worker2,
            }
        )

        print(f"Execution order: {results}")
        expected = ["t1_a", "t2_a", "t1_b", "t2_b"]
        assert results == expected, f"Expected {expected}, got {results}"
        print("✓ Alternating execution verified!")

    asyncio.run(run_test())


def test_convenience_function():
    """Test the convenience async_interlace() function."""
    print("\n=== Test: Convenience Function ===")

    async def run_test():
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

        await async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)

        print(f"Results: {results}")
        assert results == ["t1", "t2"]
        print("✓ Convenience function works!")

    asyncio.run(run_test())


def test_async_task_coordinator():
    """Test the AsyncTaskCoordinator synchronization logic directly."""
    print("\n=== Test: AsyncTaskCoordinator ===")

    async def run_test():
        schedule = Schedule(
            [
                Step("t1", "m1"),
                Step("t2", "m1"),
                Step("t1", "m2"),
            ]
        )

        coordinator = AsyncTaskCoordinator(schedule)
        results = []

        async def task1_work():
            results.append("t1_start")
            await coordinator.pause("t1", "m1")
            results.append("t1_m1")
            await coordinator.pause("t1", "m2")
            results.append("t1_m2")

        async def task2_work():
            results.append("t2_start")
            await coordinator.pause("t2", "m1")
            results.append("t2_m1")

        # Run both tasks concurrently
        await asyncio.gather(task1_work(), task2_work())

        print(f"Results: {results}")

        # The order should be: both start (unordered), then t1_m1, t2_m1, t1_m2
        assert "t1_m1" in results
        assert "t2_m1" in results
        assert "t1_m2" in results

        # Check the marker order is correct
        m1_index_t1 = results.index("t1_m1")
        m1_index_t2 = results.index("t2_m1")
        m2_index_t1 = results.index("t1_m2")

        assert m1_index_t1 < m1_index_t2, "t1 should hit m1 before t2"
        assert m1_index_t2 < m2_index_t1, "t2 should hit m1 before t1 hits m2"

        print("✓ AsyncTaskCoordinator synchronizes correctly!")

    asyncio.run(run_test())


def test_complex_race_scenario():
    """Test a more complex scenario with multiple shared resources.

    This test demonstrates how three tasks can all experience lost updates
    when they interleave their reads and writes in a maximally racy way.
    """
    print("\n=== Test: Complex Race Scenario ===")

    async def run_test():
        class SharedCounter:
            def __init__(self):
                self.value = 0

            async def increment_racy(self):
                # interlace: read_counter
                await asyncio.sleep(0)
                temp = self.value
                temp = temp + 1
                # interlace: write_counter
                await asyncio.sleep(0)
                self.value = temp

        counter = SharedCounter()

        # Three tasks, each incrementing once
        # We'll interleave them to maximize the race condition
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

        await executor.run(
            {
                "t1": counter.increment_racy,
                "t2": counter.increment_racy,
                "t3": counter.increment_racy,
            }
        )

        print("Initial counter: 0")
        print("Three tasks each increment once")
        print("Expected (with race): 1")
        print(f"Actual counter: {counter.value}")

        # With this schedule, all three tasks read 0, then all write 1
        assert counter.value == 1, f"Expected 1 (lost updates), got {counter.value}"
        print("✓ Complex race condition reproduced!")

    asyncio.run(run_test())


def test_timeout():
    """Test that timeout works correctly."""
    print("\n=== Test: Timeout ===")

    async def run_test():
        # Create a schedule where t2 is waiting but will never get its turn
        schedule = Schedule(
            [
                Step("t1", "marker1"),
                Step("t2", "marker1"),
                # t1 needs to hit marker2, but it comes after t2's marker2 in the schedule
                Step("t2", "marker2"),
                Step("t1", "marker2"),
            ]
        )

        async def worker1():
            # interlace: marker1
            await asyncio.sleep(0)
            # This task will sleep and delay hitting marker2
            await asyncio.sleep(10)
            # interlace: marker2
            await asyncio.sleep(0)

        async def worker2():
            # interlace: marker1
            await asyncio.sleep(0)
            # interlace: marker2
            await asyncio.sleep(0)

        try:
            await async_interlace(
                schedule=schedule,
                tasks={"t1": worker1, "t2": worker2},
                timeout=0.5,  # 500ms timeout
            )
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            print("✓ Timeout correctly raised!")

    asyncio.run(run_test())


def test_exception_propagation():
    """Test that exceptions in tasks are properly propagated."""
    print("\n=== Test: Exception Propagation ===")

    async def run_test():
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

        try:
            await async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert str(e) == "Intentional error in task1"
            print("✓ Exception correctly propagated!")

    asyncio.run(run_test())


def test_task_errors_tracked():
    """Test that task_errors is properly populated when tasks raise exceptions."""
    print("\n=== Test: Task Errors Tracked ===")

    async def run_test():
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

        try:
            await executor.run(
                {
                    "t1": worker1,
                    "t2": worker2,
                }
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert str(e) == "Error in task1"
            assert "t1" in executor.task_errors
            assert isinstance(executor.task_errors["t1"], ValueError)
            print("✓ Task errors properly tracked!")

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("INTERLACE ASYNC TRACE MARKERS - TEST SUITE")
    print("=" * 60)

    tests = [
        test_race_condition_buggy_schedule,
        test_race_condition_correct_schedule,
        test_multiple_markers_same_task,
        test_alternating_execution,
        test_convenience_function,
        test_async_task_coordinator,
        test_complex_race_scenario,
        test_timeout,
        test_exception_propagation,
        test_task_errors_tracked,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
