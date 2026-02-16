"""
Pytest plugin for interlace - deterministic concurrency testing.

This plugin provides fixtures and hooks for using interlace with pytest,
enabling monkey-patching of threading.Lock when tests need deterministic
thread scheduling.

Usage:
    - Place this conftest.py in your tests directory
    - Use interlace fixtures directly, or mark tests with @pytest.mark.interlace
    - Fixtures will automatically patch threading.Lock for those tests
"""

import sys
import os
import threading
import pytest

# Add parent directory to path so we can import interlace
_interlace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _interlace_path not in sys.path:
    sys.path.insert(0, _interlace_path)

from interlace.bytecode import (
    BytecodeInterlace,
    OpcodeScheduler,
    controlled_interleaving,
    run_with_schedule,
    explore_interleavings,
    _real_lock,  # Import the real lock before any patching
    _CooperativeLock,  # Import cooperative lock for patching
)
from interlace.trace_markers import (
    TraceExecutor,
    Schedule,
    Step,
    interlace,
)


def pytest_configure(config):
    """Hook that runs before test collection.

    Registers pytest markers for interlace tests.
    """
    config.addinivalue_line(
        "markers", "interlace: mark test as using interlace concurrency testing"
    )


@pytest.fixture
def _interlace_locks():
    """Internal fixture that patches/unpatches threading.Lock for interlace tests.

    This ensures cooperative locks are used during interlace-specific tests,
    allowing the scheduler to control thread execution at a fine-grained level.
    """
    original_lock = threading.Lock
    threading.Lock = _CooperativeLock
    try:
        yield
    finally:
        threading.Lock = original_lock


@pytest.fixture(autouse=True)
def _patch_locks_for_marked_tests(request):
    """Auto-used fixture that patches threading.Lock for tests marked with @pytest.mark.interlace.

    This allows tests that directly call interlace functions (without using fixtures)
    to still have cooperative lock behavior.
    """
    if request.node.get_closest_marker("interlace"):
        original_lock = threading.Lock
        threading.Lock = _CooperativeLock
        try:
            yield
        finally:
            threading.Lock = original_lock
    else:
        yield


@pytest.fixture
def interlace_bytecode(_interlace_locks):
    """Fixture providing bytecode-level interleaving control.

    Yields a context manager that can be used like:
        with interlace_bytecode(schedule, num_threads) as runner:
            runner.run([thread1_func, thread2_func])

    Example:
        def test_with_bytecode_interlace(interlace_bytecode):
            counter = Counter(0)

            def increment():
                counter.increment()

            schedule = [0, 1] * 50  # Alternate rapidly
            with interlace_bytecode(schedule, num_threads=2) as runner:
                runner.run([increment, increment])

            # May have race condition due to schedule
            assert counter.value <= 2
    """
    return controlled_interleaving


@pytest.fixture
def interlace_bytecode_explore(_interlace_locks):
    """Fixture for property-based exploration of interleavings.

    Yields the explore_interleavings function that can find race conditions
    by exploring multiple random interleavings.

    Example:
        def test_explore_for_races(interlace_bytecode_explore):
            class Counter:
                def __init__(self):
                    self.value = 0
                def increment(self):
                    temp = self.value
                    self.value = temp + 1

            result = interlace_bytecode_explore(
                setup=lambda: Counter(),
                threads=[
                    lambda c: c.increment(),
                    lambda c: c.increment(),
                ],
                invariant=lambda c: c.value == 2,
                max_attempts=100,
            )

            if not result.property_holds:
                print(f"Race condition found after {result.num_explored} attempts")
    """
    return explore_interleavings


@pytest.fixture
def interlace_trace_markers(_interlace_locks):
    """Fixture providing trace marker-based interleaving control.

    Returns a TraceExecutor factory that can be used like:
        executor = TraceExecutor(schedule)
        executor.run("thread1", func1)
        executor.run("thread2", func2)
        executor.wait()

    Example:
        def test_with_trace_markers(interlace_trace_markers):
            account = BankAccount(100)

            def worker1():
                account.transfer(50)  # Has interlace markers

            def worker2():
                account.transfer(50)

            schedule = Schedule([
                Step("worker1", "read_balance"),
                Step("worker2", "read_balance"),
                Step("worker1", "write_balance"),
                Step("worker2", "write_balance"),
            ])

            executor = interlace_trace_markers(schedule)
            executor.run("worker1", worker1)
            executor.run("worker2", worker2)
            executor.wait(timeout=5.0)

            assert account.balance == 150  # Race condition: lost update
    """
    return TraceExecutor


@pytest.fixture
def interlace_schedule_and_step(_interlace_locks):
    """Fixture providing Schedule and Step classes for trace marker tests.

    Returns a tuple of (Schedule, Step) for building execution schedules.

    Example:
        def test_with_custom_schedule(interlace_schedule_and_step):
            Schedule, Step = interlace_schedule_and_step

            schedule = Schedule([
                Step("thread1", "checkpoint1"),
                Step("thread2", "checkpoint1"),
                Step("thread1", "checkpoint2"),
                Step("thread2", "checkpoint2"),
            ])

            # Use with TraceExecutor...
    """
    return (Schedule, Step)


@pytest.fixture
def interlace_controlled_interleaving(_interlace_locks):
    """Fixture for controlled bytecode-level interleaving.

    Provides the controlled_interleaving context manager for fine-grained
    control over thread execution order at the opcode level.

    Example:
        def test_controlled_schedule(interlace_controlled_interleaving):
            results = []

            def task1():
                results.append(1)

            def task2():
                results.append(2)

            # Sequential: thread 1 completes before thread 2 starts
            schedule = [0] * 100 + [1] * 100

            with interlace_controlled_interleaving(schedule, num_threads=2) as runner:
                runner.run([task1, task2])

            # Results will be [1, 2] because task1 runs entirely first
            assert results == [1, 2]
    """
    return controlled_interleaving
