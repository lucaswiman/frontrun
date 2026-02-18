"""
Pytest plugin for frontrun - deterministic concurrency testing.

This plugin provides fixtures and hooks for using frontrun with pytest,
enabling monkey-patching of threading.Lock when tests need deterministic
thread scheduling.

Usage:
    - Place this conftest.py in your tests directory
    - Use frontrun fixtures directly, or mark tests with @pytest.mark.frontrun
    - Fixtures will automatically patch threading.Lock for those tests
"""

import os
import sys
import threading

import pytest

# Add parent directory to path so we can import frontrun
_frontrun_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _frontrun_path not in sys.path:
    sys.path.insert(0, _frontrun_path)

from frontrun.bytecode import (
    _CooperativeLock,  # Import cooperative lock for patching
    controlled_interleaving,
    explore_interleavings,
)
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor


def pytest_configure(config):
    """Hook that runs before test collection.

    Registers pytest markers for frontrun tests.
    """
    config.addinivalue_line("markers", "frontrun: mark test as using frontrun concurrency testing")
    config.addinivalue_line(
        "markers",
        "intentionally_leaves_dangling_threads: mark test as intentionally leaving threads alive (e.g., deadlock tests that cannot be cleaned up)",
    )


@pytest.fixture
def _frontrun_locks():
    """Internal fixture that patches/unpatches threading.Lock for frontrun tests.

    This ensures cooperative locks are used during frontrun-specific tests,
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
    """Auto-used fixture that patches threading.Lock for tests marked with @pytest.mark.frontrun.

    This allows tests that directly call frontrun functions (without using fixtures)
    to still have cooperative lock behavior.
    """
    if request.node.get_closest_marker("frontrun"):
        original_lock = threading.Lock
        threading.Lock = _CooperativeLock
        try:
            yield
        finally:
            threading.Lock = original_lock
    else:
        yield


@pytest.fixture(autouse=True)
def _check_thread_cleanup(request):
    """Auto-used fixture that verifies all non-daemon threads are cleaned up after each test.

    This catches tests that start threads but don't properly join them, which would
    otherwise cause pytest to hang at exit. By checking explicitly, we get clear
    error messages instead of silent hangs.

    Also prints diagnostic info about daemon threads that are still alive (these won't
    block exit but may indicate tests that create intentionally deadlocked threads).
    """
    # Record threads before test
    initial_threads = set(threading.enumerate())

    yield

    # Check for lingering threads after test
    final_threads = set(threading.enumerate())
    new_threads = final_threads - initial_threads

    # Separate into daemon and non-daemon threads (excluding main thread)
    main_thread = threading.main_thread()
    alive_threads = [t for t in new_threads if t != main_thread and t.is_alive()]

    daemon_threads = [t for t in alive_threads if t.daemon]
    non_daemon_threads = [t for t in alive_threads if not t.daemon]

    # Print diagnostic info for all alive threads
    if alive_threads:
        print(f"\n[THREAD CLEANUP] Test: {request.node.nodeid}")
        for t in alive_threads:
            status = "daemon" if t.daemon else "NON-DAEMON"
            print(f"  - {t.name} (ident={t.ident}, {status}, alive={t.is_alive()})")

    # Fail on ANY lingering threads unless explicitly marked as intentional
    if alive_threads and not request.node.get_closest_marker("intentionally_leaves_dangling_threads"):
        thread_info = ", ".join(
            f"{t.name} ({'daemon' if t.daemon else 'NON-DAEMON'}, ident={t.ident})" for t in alive_threads
        )
        pytest.fail(
            f"Test {request.node.nodeid} left {len(alive_threads)} thread(s) running: {thread_info}. "
            f"All threads must be joined before test completion. "
            f"If this is intentional (e.g., testing deadlocks that cannot be cleaned up), "
            f"mark the test with @pytest.mark.intentionally_leaves_dangling_threads"
        )
    # Fail on non-daemon threads (these would block pytest exit)
    if non_daemon_threads:
        thread_info = ", ".join(f"{t.name} (ident={t.ident})" for t in non_daemon_threads)
        pytest.fail(
            f"Test {request.node.nodeid} left {len(non_daemon_threads)} non-daemon thread(s) running: {thread_info}. "
            f"All threads must be joined before test completion."
        )


@pytest.fixture
def frontrun_bytecode(_frontrun_locks):
    """Fixture providing bytecode-level interleaving control.

    Yields a context manager that can be used like:
        with frontrun_bytecode(schedule, num_threads) as runner:
            runner.run([thread1_func, thread2_func])

    Example:
        def test_with_bytecode_frontrun(frontrun_bytecode):
            counter = Counter(0)

            def increment():
                counter.increment()

            schedule = [0, 1] * 50  # Alternate rapidly
            with frontrun_bytecode(schedule, num_threads=2) as runner:
                runner.run([increment, increment])

            # May have race condition due to schedule
            assert counter.value <= 2
    """
    return controlled_interleaving


@pytest.fixture
def frontrun_bytecode_explore(_frontrun_locks):
    """Fixture for property-based exploration of interleavings.

    Yields the explore_interleavings function that can find race conditions
    by exploring multiple random interleavings.

    Example:
        def test_explore_for_races(frontrun_bytecode_explore):
            class Counter:
                def __init__(self):
                    self.value = 0
                def increment(self):
                    temp = self.value
                    self.value = temp + 1

            result = frontrun_bytecode_explore(
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
def frontrun_trace_markers(_frontrun_locks):
    """Fixture providing trace marker-based interleaving control.

    Returns a TraceExecutor factory that can be used like:
        executor = TraceExecutor(schedule)
        executor.run("thread1", func1)
        executor.run("thread2", func2)
        executor.wait()

    Example:
        def test_with_trace_markers(frontrun_trace_markers):
            account = BankAccount(100)

            def worker1():
                account.transfer(50)  # Has frontrun markers

            def worker2():
                account.transfer(50)

            schedule = Schedule([
                Step("worker1", "read_balance"),
                Step("worker2", "read_balance"),
                Step("worker1", "write_balance"),
                Step("worker2", "write_balance"),
            ])

            executor = frontrun_trace_markers(schedule)
            executor.run("worker1", worker1)
            executor.run("worker2", worker2)
            executor.wait(timeout=5.0)

            assert account.balance == 150  # Race condition: lost update
    """
    return TraceExecutor


@pytest.fixture
def frontrun_schedule_and_step(_frontrun_locks):
    """Fixture providing Schedule and Step classes for trace marker tests.

    Returns a tuple of (Schedule, Step) for building execution schedules.

    Example:
        def test_with_custom_schedule(frontrun_schedule_and_step):
            Schedule, Step = frontrun_schedule_and_step

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
def frontrun_controlled_interleaving(_frontrun_locks):
    """Fixture for controlled bytecode-level interleaving.

    Provides the controlled_interleaving context manager for fine-grained
    control over thread execution order at the opcode level.

    Example:
        def test_controlled_schedule(frontrun_controlled_interleaving):
            results = []

            def task1():
                results.append(1)

            def task2():
                results.append(2)

            # Sequential: thread 1 completes before thread 2 starts
            schedule = [0] * 100 + [1] * 100

            with frontrun_controlled_interleaving(schedule, num_threads=2) as runner:
                runner.run([task1, task2])

            # Results will be [1, 2] because task1 runs entirely first
            assert results == [1, 2]
    """
    return controlled_interleaving
